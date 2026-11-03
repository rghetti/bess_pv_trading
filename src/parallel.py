"""Process initializers for bounded native-library and inference parallelism."""

import os


def configure_worker_threads(threads, configure_torch=True):
    """Initialize a fresh worker's BLAS and PyTorch pools before model inference.

    Gurobi's solver thread count remains a separate YAML setting. Call once per
    spawned process: PyTorch inter-op parallelism cannot be reset after use.
    """
    from threadpoolctl import threadpool_limits

    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Worker threads must be a positive integer")
    # Configure future imports as well as numerical pools already loaded at spawn.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    threadpool_limits(limits=threads)
    # Pure statistical workers do not need to import the inference framework.
    if not configure_torch:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
