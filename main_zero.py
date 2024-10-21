import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.optim import ZeroRedundancyOptimizer


from kron_torch import Kron

class LargerConvNet(nn.Module):
    def __init__(self):
        super(LargerConvNet, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([64, 28, 28])
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([128, 28, 28])
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([256, 14, 14])
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.ln4 = nn.LayerNorm([512, 7, 7])
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.ln5 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 10)
        self.static_param = nn.Parameter(torch.randn(1), requires_grad=False)

    def forward(self, x):
        x = self.scalar * x
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(F.max_pool2d(x, 2))))
        x = F.relu(self.ln4(self.conv4(F.max_pool2d(x, 2))))
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.ln5(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def print_model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, requires_grad=True")
            total_params += param.numel()
        else:
            print(f"{name}: {param.shape}, requires_grad=False")
    print(f"Total trainable parameters: {total_params}")


def train(model, device, train_loader, optimizer, scheduler, epoch):
    initial_static_param = model.module.static_param.clone()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move both data and target to the device
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    final_static_param = model.module.static_param.clone()
    print(f'Static param change: {(final_static_param - initial_static_param).item():.4f}')


def setup():
    print("Setting up distributed environment...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=0,
                world_size=1
            )
            print("Distributed environment initialized successfully.")
            return True
        except Exception as e:
            print(f"Failed to initialize distributed environment: {e}")
            return False
    else:
        print("Distributed environment already initialized.")
        return True


def cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


def main():
    try:
        print("Starting main function...")
        is_distributed = setup()

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            "../data", train=True, download=True, transform=transform
        )

        num_workers = 4

        # Use DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=128,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Only use pin_memory if CUDA is available
        )

        # Add test dataset and loader
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )

        model_kron = LargerConvNet().to(device)
        model_sgd = LargerConvNet().to(device)

        if is_distributed:
            print("Wrapping models with DDP...")
            model_kron = DDP(model_kron, device_ids=[device.index])
            model_sgd = DDP(model_sgd, device_ids=[device.index])
        else:
            print("Running in non-distributed mode.")

        model_kron = torch.compile(model_kron)
        model_sgd = torch.compile(model_sgd)

        print_model_summary(model_kron)

        print(f"Initial static param (Kron): {model_kron.module.static_param.item():.4f}")
        print(f"Initial static param (SGD): {model_sgd.module.static_param.item():.4f}")

        optimizer_kron = ZeroRedundancyOptimizer(
            model_kron.parameters(),
            optimizer_class=Kron,
            lr=0.001,
            weight_decay=0.0001,
            preconditioner_update_probability=1.0,
        )
        optimizer_sgd = ZeroRedundancyOptimizer(
            model_sgd.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=0.01,
        )
        num_epochs = 3
        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch

        scheduler_kron = torch.optim.lr_scheduler.LinearLR(
            optimizer_kron, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )
        scheduler_sgd = torch.optim.lr_scheduler.LinearLR(
            optimizer_sgd, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )

        print("Training with Kron optimizer:")
        for epoch in range(1, num_epochs + 1):
            train_sampler.set_epoch(epoch)
            train(model_kron, device, train_loader, optimizer_kron, scheduler_kron, epoch)
        kron_accuracy = test(model_kron, device, test_loader)

        print("\nKron optimizer states:")
        for group_idx, group in enumerate(optimizer_kron.param_groups):
            print(f"Parameter group {group_idx}:")
            for p in group["params"]:
                if p.requires_grad:
                    state = optimizer_kron.state[p]
                    print(f"  Parameter: shape={p.shape}, dtype={p.dtype}")
                    if state:
                        for key, value in state.items():
                            if key == "Q":
                                print(f"    Q: list of {len(value)} tensors")
                                for i, q_tensor in enumerate(value):
                                    print(
                                        f"      Q[{i}]: shape={q_tensor.shape}, dtype={q_tensor.dtype}"
                                    )
                            elif key == "exprs":
                                print(f"    exprs: tuple of {len(value)} expressions")
                            elif isinstance(value, torch.Tensor):
                                print(
                                    f"    {key}: shape={value.shape}, dtype={value.dtype}"
                                )
                            else:
                                print(f"    {key}: {type(value)}")
                    else:
                        print("    No state")

        print("\nOptimizer attributes:")
        for attr_name in dir(optimizer_kron):
            if not attr_name.startswith("_") and attr_name not in ["state", "param_groups"]:
                attr_value = getattr(optimizer_kron, attr_name)
                if not callable(attr_value):
                    if isinstance(attr_value, torch.Tensor):
                        print(
                            f"  {attr_name}: shape={attr_value.shape}, dtype={attr_value.dtype}"
                        )
                    else:
                        print(f"  {attr_name}: {type(attr_value)}")

        print("\nTraining with SGD optimizer:")
        for epoch in range(1, num_epochs + 1):
            train(model_sgd, device, train_loader, optimizer_sgd, scheduler_sgd, epoch)
        sgd_accuracy = test(model_sgd, device, test_loader)

        print(f"\nFinal results:")
        print(f"Scalar value (Kron): {model_kron.module.scalar.item():.4f}")
        print(f"Scalar value (SGD): {model_sgd.module.scalar.item():.4f}")
        print(f"Final static param (Kron): {model_kron.module.static_param.item():.4f}")
        print(f"Final static param (SGD): {model_sgd.module.static_param.item():.4f}")
        print(f"Test Accuracy (Kron): {kron_accuracy:.2f}%")
        print(f"Test Accuracy (SGD): {sgd_accuracy:.2f}%")

    finally:
        if is_distributed:
            cleanup()

if __name__ == "__main__":
    main()
