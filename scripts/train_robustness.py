"""Train every (algorithm, terminal liquidation cost, seed) combination and cache the models.

Training is separated from analysis on purpose. A notebook that retrains five algorithms across a parameter grid
takes hours to execute and cannot be re-run while reading it; one that loads cached models executes in minutes and
stays honest, because the models it reads are the artefacts this script produced and not something recomputed
slightly differently each time.

Checkpoints are written along the way, so a single run yields a learning curve rather than a single point. That
matters here: the five algorithms have very different sample efficiency -- RecurrentPPO costs about twenty-seven
times PPO per environment step on this machine -- and a comparison at one arbitrary budget would mostly report
which of them happens to be fastest at that budget.

    python scripts/train_robustness.py [--steps N] [--workers K] [--out DIR]
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "runs" / "robustness"

# The axis the sensitivity is about: what it costs to be left holding inventory at the horizon.
ALPHA_LIQS = (0.0, 0.1, 1.0)
PER_STEP_INVENTORY_AVERSION = 0.01
ALGORITHMS = ("PPO", "A2C", "SAC", "TD3", "RecurrentPPO")
SEEDS = (1, 2, 3)
FEATURES = "proxy"          # observable features only: the latent factor has to be inferred
CHECKPOINTS = (200_000, 500_000, 1_000_000)


def model_stem(algorithm: str, alpha_liq: float, seed: int) -> str:
    """File stem for one trained model.

    The decimal point is replaced because `Path.suffix` would otherwise read `.0_seed1_at200000` as an extension:
    Stable-Baselines then declines to append `.zip`, the file lands without one, and the resume check that looks
    for `.zip` never matches, so every run is repeated from scratch on restart.
    """
    return f"{algorithm}_alpha{alpha_liq:g}".replace(".", "p") + f"_seed{seed}"


def checkpoint_path(out_dir, algorithm: str, alpha_liq: float, seed: int, steps: int) -> Path:
    return Path(out_dir) / f"{model_stem(algorithm, alpha_liq, seed)}_at{steps}.zip"


def build_env(num_trajectories, seed, alpha_liq, features=FEATURES, normalise_action=True,
              sensitivity=0.2):
    """The environment, optionally behind an information-set wrapper.

    `features=None` returns the bare environment, whose observation is the raw state. That is what the closed
    forms need: they read inventory and time out of fixed positions in the state vector and would silently
    receive a feature vector otherwise -- `CarteaJaimungalMmAgent` catches this with an assertion on the time
    column, but only because the feature vector happens not to be time-uniform.
    """
    from gym_local.ModelDynamics import AdverseFillModelDynamics
    from gym_local.TradingEnvironment import TradingEnvironment
    from gym_local.observation import InformationSet
    from rewards.RewardFunctions import CjMmCriterion
    from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
    from stochastic_processes.fill_probability_models import ExponentialFillFunction
    from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

    terminal_time, n_steps = 1.0, 200
    step_size = terminal_time / n_steps
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=2.0,
        ou_process=OuMidpriceModel(mean_reversion_level=0.0, mean_reversion_speed=0.05, volatility=20.0,
                                   initial_price=0.0, terminal_time=terminal_time, step_size=step_size,
                                   num_trajectories=num_trajectories),
        initial_price=100.0, terminal_time=terminal_time, step_size=step_size,
        num_trajectories=num_trajectories)
    dynamics = AdverseFillModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([140.0, 140.0]), sensitivity, 4, step_size, num_trajectories),
        fill_probability_model=ExponentialFillFunction(1.5, step_size, num_trajectories),
        num_trajectories=num_trajectories, track_adverse_fills=False, queue_probability=1.0)
    env = TradingEnvironment(
        terminal_time=terminal_time, n_steps=n_steps, model_dynamics=dynamics,
        reward_function=CjMmCriterion(per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
                                      terminal_inventory_aversion=alpha_liq, terminal_time=terminal_time),
        max_inventory=20, num_trajectories=num_trajectories,
        normalise_action_space=normalise_action, normalise_observation_space=False)
    env.seed(seed)
    return InformationSet(env, features=features, alpha_index=4) if features is not None else env


def run_one(job):
    """One (algorithm, alpha_liq, seed) training run, checkpointing as it goes."""
    algorithm, alpha_liq, seed, total_steps, out_dir, n_train = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)

    from agents.LearnedAgents import make_model
    from gym_local.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment

    stem = Path(out_dir) / model_stem(algorithm, alpha_liq, seed)
    checkpoints = [c for c in CHECKPOINTS if c <= total_steps] or [total_steps]
    if all((Path(f"{stem}_at{c}.zip")).exists() for c in checkpoints):
        return f"{stem.name}: deja present, ignore"

    wrapped = build_env(n_train, seed, alpha_liq)
    model = make_model(algorithm, StableBaselinesTradingEnvironment(wrapped), seed=seed, n_steps=wrapped.n_steps)

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
    parser.add_argument("--n-train", type=int, default=100, help="parallel trajectories per training env")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(algorithm, alpha_liq, seed, args.steps, str(args.out), args.n_train)
            for algorithm, alpha_liq, seed in itertools.product(ALGORITHMS, ALPHA_LIQS, SEEDS)]
    # Slowest first, so the long pole is not left to run alone at the end.
    order = {"RecurrentPPO": 0, "SAC": 1, "TD3": 2, "PPO": 3, "A2C": 4}
    jobs.sort(key=lambda job: order[job[0]])
    print(f"{len(jobs)} entrainements, {args.workers} en parallele, {args.steps:,} pas chacun", flush=True)

    from multiprocessing import Pool
    started = time.time()
    with Pool(args.workers) as pool:
        for done, message in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {message}", flush=True)
    print(f"\ntermine en {(time.time() - started) / 60:.0f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
