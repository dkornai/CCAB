"""
End-to-end comparison of the Python and C++ ideal observers on the demo experiment: behaviour (mean reward per block, averaged
over replicates) and speed. As the two backends draw different random numbers, behaviour agrees only statistically.

Usage: python tests/compare_backends.py [n_replicates] [n_workers] [N]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import models_context as MC
import models_obs_rew as M
from experiment import ExperimentalEnv, run_replicate, run_replicates

COV = np.eye(2) * 0.02
ENV = ExperimentalEnv(
    context_o_params={
        0: {0: {'loc': np.array([1.0, 0.0]), 'cov': COV}, 1: {'loc': np.array([-1.0, 0.0]), 'cov': COV}},
        1: {0: {'loc': np.array([0.0, 1.0]), 'cov': COV}, 1: {'loc': np.array([0.0, -1.0]), 'cov': COV}},
    },
    context_r_params={
        0: {0: {'p_rew': np.array([1.0, 0.0, 0.0, 0.0])}, 1: {'p_rew': np.array([0.0, 1.0, 0.0, 0.0])}},
        1: {0: {'p_rew': np.array([0.0, 0.0, 1.0, 0.0])}, 1: {'p_rew': np.array([0.0, 0.0, 0.0, 1.0])}},
    },
)
AGENT = {
    "type_context": MC.CjCRP, "type_obs": M.ConjugateGaussianInvWish, "type_rew": M.ConjugateBernoulli,
    "hyp_context": {'gamma': 0.1, 'alpha_o': 0.5, 'alpha_r': 0.5},
    "hyp_obs": {'mu0': np.array([0.0, 0.0]), 'kappa0': 0.1, 'nu0': 4.0, 'Lambda0': np.eye(2)},
    "hyp_rew": {'alpha0': np.array([0.05] * 4), 'beta0': np.array([0.5] * 4)},
}
SCHEDULE = [(50, 0, 0), (50, 1, 1), (50, 0, 0), (50, 0, 1)]


if __name__ == "__main__":
    n_replicates = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    # Behaviour, from replicates run in parallel
    print(f"{n_replicates} replicates, N={N} particles, {n_workers} workers")
    print(f"{'backend':8s} {'time':>8s}   mean reward per block (± standard error)")
    for backend in ("python", "cpp"):
        t = time.perf_counter()
        results = run_replicates(ENV, AGENT, N, SCHEDULE, n_replicates=n_replicates, n_workers=n_workers, seed=0, backend=backend)
        dt = time.perf_counter() - t
        rewards = np.array([r.experiment_history.reward for r in results]).reshape(n_replicates, len(SCHEDULE), -1).mean(axis=2)
        blocks = "  ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(rewards.mean(axis=0), rewards.std(axis=0, ddof=1) / np.sqrt(n_replicates)))
        print(f"{backend:8s} {dt:7.1f}s   {blocks}")

    # Speed of a single replicate with the demo's 250 particles, in this process
    print("\nSingle replicate, N=250 particles, 200 trials:")
    timings = {}
    for backend in ("cpp", "python"):
        t = time.perf_counter()
        run_replicate(ENV, AGENT, 250, SCHEDULE, seed=1, backend=backend)
        timings[backend] = time.perf_counter() - t
        print(f"  {backend:8s} {timings[backend]:7.2f}s")
    print(f"  speedup  {timings['python'] / timings['cpp']:7.0f}x")
