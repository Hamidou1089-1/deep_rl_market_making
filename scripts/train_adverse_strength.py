"""Train the viable learned agents across the *strength* of adverse selection.

This is the sweep the project is actually about. `alpha_liq` prices the inventory left at the horizon and lives in
the criterion; `sensitivity` (gamma) is the tilt the latent factor applies to the two arrival intensities and
lives in the environment. Sweeping the first asks how the scoring changes; sweeping the second asks how the
*world* changes, and whether a learned quoting policy still holds up when being picked off gets harder.

    lambda_sell(alpha) = lambda_0 exp(-gamma alpha),   lambda_buy(alpha) = lambda_0 exp(+gamma alpha)

`gamma = 0` reproduces the Poisson environment bit for bit, so the sweep starts from a world with no adverse
selection at all and ends at twice the value used elsewhere in the project.

Only PPO, TD3 and SAC are trained. A2C diverges with more data at every setting measured in Test 5, and
RecurrentPPO costs about twenty-seven times PPO per step, which would put this sweep beyond a day; both are
reported there rather than repeated here.

    python scripts/train_adverse_strength.py [--steps N] [--workers K]
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runs" / "adverse_strength"

SENSITIVITIES = (0.0, 0.1, 0.2, 0.4)
ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (1, 2, 3)
ALPHA_LIQ = 1.0          # the regime in which the criterion is a market making problem
CHECKPOINTS = (1_000_000, 4_000_000)
FEATURES = "proxy"


def model_stem(algorithm: str, sensitivity: float, seed: int) -> str:
    return f"{algorithm}_gamma{sensitivity:g}".replace(".", "p") + f"_seed{seed}"


def checkpoint_path(out_dir, algorithm: str, sensitivity: float, seed: int, steps: int) -> Path:
    return Path(out_dir) / f"{model_stem(algorithm, sensitivity, seed)}_at{steps}.zip"


def run_one(job):
    algorithm, sensitivity, seed, total_steps, out_dir, n_train = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from agents.LearnedAgents import make_model
    from gym_local.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
    from train_robustness import build_env

    checkpoints = [c for c in CHECKPOINTS if c <= total_steps] or [total_steps]
    if all(checkpoint_path(out_dir, algorithm, sensitivity, seed, c).exists() for c in checkpoints):
        return f"{model_stem(algorithm, sensitivity, seed)}: deja present"

    wrapped = build_env(n_train, seed, ALPHA_LIQ, features=FEATURES, normalise_action=True,
                        sensitivity=sensitivity)
    model = make_model(algorithm, StableBaselinesTradingEnvironment(wrapped), seed=seed,
                       n_steps=wrapped.n_steps)
    stem = Path(out_dir) / model_stem(algorithm, sensitivity, seed)
    started, done = time.time(), 0
    for checkpoint in checkpoints:
        model.learn(total_timesteps=checkpoint - done, reset_num_timesteps=False, progress_bar=False)
        done = checkpoint
        model.save(f"{stem}_at{checkpoint}")
    return f"{stem.name}: {done:,} pas en {time.time() - started:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=max(CHECKPOINTS))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-train", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(a, g, s, args.steps, str(args.out), args.n_train)
            for a, g, s in itertools.product(ALGORITHMS, SENSITIVITIES, SEEDS)]
    order = {"SAC": 0, "TD3": 1, "PPO": 2}
    jobs.sort(key=lambda job: order[job[0]])
    print(f"{len(jobs)} entrainements, {args.workers} en parallele, {args.steps:,} pas, "
          f"gamma dans {SENSITIVITIES}", flush=True)

    from multiprocessing import Pool
    started = time.time()
    with Pool(args.workers) as pool:
        for done, message in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {message}", flush=True)
    print(f"\ntermine en {(time.time() - started) / 60:.0f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
