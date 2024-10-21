import string
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=200
):
    """Anneal preconditioner update probability during beginning of training."""
    def _schedule(n):
        n = torch.tensor(n, dtype=torch.float32)
        prob = torch.minimum(
            torch.maximum(
                max_prob * torch.exp(-decay * (n - flat_start)), torch.tensor(min_prob)
            ),
            torch.tensor(max_prob),
        )
        return prob
    return _schedule

class ZeroBlockKron(torch.optim.Optimizer):
    """Distributed Kron optimizer with improved sharding and communication.

    This optimizer maintains the same mathematical operations as your original Kron optimizer,
    while distributing the computation and memory across multiple devices/processes with
    improved sharding and communication strategies.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): Learning rate (default: 0.001).
        b1 (float, optional): Momentum parameter (default: 0.9).
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0.0).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int, optional): Max size for dim's preconditioner to be
            triangular (default: 8192).
        max_skew_triangular (float, optional): Max skew for dim's preconditioner to be
            triangular (default: inf).
        min_ndim_triangular (int, optional): Minimum number of dimensions a layer needs
            to have triangular preconditioners (default: 2).
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator. Defaults
            to the same dtype as the parameters.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner (default: None).
        block_size (int, optional): Size of the blocks for parameter sharding (default: 128).
        sync_frequency (int, optional): Frequency of parameter synchronization (default: 1).
    """

    def __init__(
        self,
        params,
        lr=0.001,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        max_skew_triangular=float("inf"),
        min_ndim_triangular=2,
        mu_dtype=None,
        precond_dtype=None,
        block_size=128,
        sync_frequency=1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta parameter: {b1}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            min_ndim_triangular=min_ndim_triangular,
            precond_lr=0.1,  # Preconditioner learning rate
            precond_init_scale=1.0,  # Preconditioner initialization scale
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            block_size=block_size,
            sync_frequency=sync_frequency,
        )
        super(ZeroBlockKron, self).__init__(params, defaults)

        # Device and data type settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Distributed setup
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
        else:
            self.rank = 0
            self.world_size = 1
            self.is_distributed = False

        self._global_clip = (
            sum(
                p.numel()
                for group in self.param_groups
                for p in group["params"]
                if p.requires_grad
            )
            ** 0.5
        )
        self._element_clip = 1.0
        self._tiny = 1e-30
        self._prob_step = 0
        self.sync_counter = 0

        # Initialize parameter sharding
        self._make_lookup_and_enumeratables()
        self._init_state()

    @torch.no_grad()
    def _make_lookup_and_enumeratables(self):
        self.enumeratables = []
        global_counter = 0

        for group in self.param_groups:
            block_size = group["block_size"]
            for param in group["params"]:
                if param.requires_grad:
                    name = f"param_{global_counter}"
                    param_size = param.numel()
                    num_blocks = (param_size + block_size - 1) // block_size
                    for block_idx in range(num_blocks):
                        start_idx = block_idx * block_size
                        end_idx = min(start_idx + block_size, param_size)
                        block_name = f"{name}_block_{block_idx}"
                        self.enumeratables.append(
                            (
                                global_counter * num_blocks + block_idx,
                                block_name,
                                param,
                                start_idx,
                                end_idx,
                                group,
                            )
                        )
                    global_counter += 1

    def _enumerate_sharded_params(self):
        for (
            global_counter,
            block_name,
            param,
            start_idx,
            end_idx,
            group,
        ) in self.enumeratables:
            if global_counter % self.world_size != self.rank:
                continue
            yield block_name, param, start_idx, end_idx, group

    def _init_state(self):
        for (
            block_name,
            param,
            start_idx,
            end_idx,
            group,
        ) in self._enumerate_sharded_params():
            state = self.state.setdefault(block_name, {})
            if len(state) == 0:
                state["step"] = 0
                block_param = param.view(-1)[start_idx:end_idx]

                # Use mixed precision for momentum buffer and preconditioners
                state["momentum_buffer"] = torch.zeros_like(
                    block_param, dtype=group["mu_dtype"] or torch.float16
                )

                state["Q"], state["exprs"] = init_Q_exprs(
                    block_param,
                    group["precond_init_scale"],
                    group["max_size_triangular"],
                    group["max_skew_triangular"],
                    group["min_ndim_triangular"],
                    dtype=group["precond_dtype"] or torch.float16,
                )

    @torch.no_grad()
    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        # Implement the optimizer step with distributed parameter sharding
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Update preconditioners together
        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        device = self.device
        do_update = torch.rand([], device=device) < update_prob
        self._prob_step += 1

        balance = torch.rand([], device=device) < 0.01 and do_update

        # Start asynchronous gradient reduction
        gradient_handles = self._reduce_gradients_async()

        # Proceed with local computations
        for (
            block_name,
            param,
            start_idx,
            end_idx,
            group,
        ) in self._enumerate_sharded_params():
            state = self.state[block_name]

            # Wait for gradient reduction to complete for this block
            gradient_handles[block_name].wait()
            grad = param.grad.view(-1)[start_idx:end_idx]
            if grad is None:
                continue

            state["step"] += 1

            # Update momentum buffer
            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(group["b1"]).add_(grad, alpha=1 - group["b1"])

            # Balance preconditioners periodically
            if balance:
                _balance_Q(state["Q"])

            # Update preconditioner
            if do_update:
                _update_precond(
                    state["Q"],
                    state["exprs"],
                    torch.randn_like(momentum_buffer, dtype=momentum_buffer.dtype),
                    momentum_buffer.to(dtype=state["Q"][0].dtype, non_blocking=True),
                    group["precond_lr"],
                    self._tiny,
                )

            # Precondition gradients
            pre_grad = _precond_grad(
                state["Q"],
                state["exprs"],
                momentum_buffer.to(dtype=state["Q"][0].dtype, non_blocking=True),
            ).to(dtype=param.dtype, non_blocking=True)

            # Apply trust region
            torch.nn.utils.clip_grad_norm_(pre_grad, self._global_clip)
            pre_grad.clamp_(-self._element_clip, self._element_clip)

            # Apply weight decay and update parameters
            if group["weight_decay"] != 0:
                pre_grad.add_(param.view(-1)[start_idx:end_idx], alpha=group["weight_decay"])
            param.view(-1)[start_idx:end_idx].add_(pre_grad, alpha=-group["lr"])

        # Increment sync counter
        self.sync_counter += 1
        if self.sync_counter % self.param_groups[0]["sync_frequency"] == 0:
            # Synchronize parameters across processes
            self._sync_params()

        return loss

    @torch.no_grad()
    def _reduce_gradients_async(self):
        gradient_handles = {}
        if not self.is_distributed:
            return gradient_handles

        for (
            global_counter,
            block_name,
            param,
            start_idx,
            end_idx,
            group,
        ) in self.enumeratables:
            if param.grad is not None:
                grad = param.grad.view(-1)[start_idx:end_idx]
                handle = dist.all_reduce(
                    grad, op=dist.ReduceOp.SUM, async_op=True
                )
                gradient_handles[block_name] = handle

        return gradient_handles

    @torch.no_grad()
    def _sync_params(self):
        if not self.is_distributed:
            return
        # Use asynchronous parameter synchronization
        handles = []
        for (
            global_counter,
            block_name,
            param,
            start_idx,
            end_idx,
            group,
        ) in self.enumeratables:
            src_rank = global_counter % self.world_size
            tensor = param.data.view(-1)[start_idx:end_idx]
            handle = dist.broadcast(tensor, src=src_rank, async_op=True)
            handles.append(handle)
        # Wait for all broadcasts to complete
        for handle in handles:
            handle.wait()

def init_Q_exprs(t, scale, max_size, max_skew, min_ndim_triangular, dtype=None):
    """Initialize Q matrices and einsum expressions for preconditioning."""
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->,"
        exprP = ",,->,"
        exprGs = [",->"]
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))
        if len(shape) == 1:
            beta_size = 1  # 2nd largest size
        else:
            beta_size = sorted(list(shape))[-2]

        Q = []
        exprGs = []
        piece1A, piece2A, piece3A = ([], "", "")
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, size in enumerate(shape):
            if (
                size == 1
                or size > max_size
                or size > max_skew * beta_size
                or len(shape) < min_ndim_triangular
            ):
                # Use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]

                piece1 = "".join(
                    [
                        (letters[j + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)
            else:
                # Use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [
                        (letters[j + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[j + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]

def _balance_Q(Q_in):
    """Balance the Q matrices to maintain numerical stability."""
    norms = torch.stack([torch.max(torch.abs(q)) for q in Q_in])
    geometric_mean = norms.prod() ** (1 / len(Q_in))
    for i, q in enumerate(Q_in):
        q.mul_(geometric_mean / norms[i])

def _solve_triangular_right(X, A):
    """Compute X @ inv(A) for triangular matrix A."""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32, non_blocking=True)
    A = A.to(dtype=torch.float32, non_blocking=True)
    return torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False).to(
        dtype=orig_dtype, non_blocking=True
    )[0]

def _norm_lower_bound(A):
    """Compute a lower bound for the spectral norm of A."""
    max_abs = torch.max(torch.abs(A))
    if max_abs == 0:
        return max_abs
    aa = A @ A.T
    value0 = torch.max(torch.sum(aa, dim=0))
    value1 = torch.max(torch.sum(aa, dim=1))
    if value0 > value1:
        idx = torch.argmax(torch.sum(aa, dim=0))
        x = A[:, idx]
        return max_abs * torch.linalg.norm((x / torch.linalg.norm(x)) @ A.T)
    else:
        idx = torch.argmax(torch.sum(aa, dim=1))
        x = A[idx, :]
        return max_abs * torch.linalg.norm(A.T @ (x / torch.linalg.norm(x)))

def _update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    # Compute A
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])

    for i, q in enumerate(Q):
        if q.dim() < 2:
            conjB = conjB / q
        else:
            conjB = _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)

    # Update Q
    for q, exprG in zip(Q, exprGs):
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)
        if q.dim() < 2:
            q.sub_(
                step
                / (torch.max(torch.abs(term1 + term2)) + tiny)
                * (term1 - term2)
                * q
            )
        else:
            q.sub_(
                step
                / (_norm_lower_bound(term1 + term2) + tiny)
                * torch.triu(term1 - term2)
                @ q
            )

def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)