# distributed_kron_torch
A distributed version of PSGD Kron

Right now lazy updates happen every 10 intervals. we should make sure we trigger a communication when we have a new prcond. might need to do more but at least should communicate when we have a fresh curvature
