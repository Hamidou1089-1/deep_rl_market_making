"""Train the learned agents across the strength of *informational* adverse selection.

The environment has been reduced to one swept axis and two fixed ingredients, on the principle that an axis is
only worth sweeping if what it varies is something an agent could plausibly learn about.

    swept    gamma          the latent factor tilts the two arrival intensities. This is the mechanism the
                            project is about: it is predictable from the observable order flow, so it is the only
                            one against which a policy can do something other than quote wider.

    fixed    p = 0.95       queue position. Left slightly below one for realism rather than as an axis: what an
                            agent would learn from it is to post earlier to gain priority, which this action
                            space cannot express, so sweeping it would measure a capability the agent does not
                            have.
             zeta = 0.1     the agent's own executions move the price against its new position. Kept because it
                            is a real feature of trading and costs a visible 5.8 % of profit, but not swept: it
                            is an optimal-execution effect, and the agent shows no sign of exploiting it.
             crossing off   the mechanical channel of Lalor and Swishchuk needs rho near one to fire, and at
                            rho = 4.7 it is inert. Reaching that regime means a different market, not a flag.
             sigma constant volatility is not stochastic here. `HestonMidpriceModel` exists in the package but
                            carries no alpha, so combining the two would need new code.

    python scripts/train_signal_strength.py [--steps N] [--workers K]
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
DEFAULT_OUT = REPO / "runs" / "signal_strength"

SENSITIVITIES = (0.0, 0.1, 0.2, 0.4, 0.8)
QUEUE_PROBABILITY = 0.95
SELF_IMPACT = 0.1
VOLATILITY = 2.0
TRACK_ADVERSE_FILLS = False

ALGORITHMS = ("PPO", "TD3", "SAC")
SEEDS = (1, 2, 3)
CHECKPOINTS = (1_000_000, 2_000_000, 4_000_000)
TERMINAL_TIME, N_STEPS = 1.0, 200
ALPHA_LIQ, PHI, MAX_INVENTORY = 1.0, 0.01, 20


def checkpoint_path(out_dir, algorithm: str, gamma: float, seed: int, steps: int) -> Path:
    stem = f"{algorithm}_gamma{gamma:g}".replace(".", "p") + f"_seed{seed}_at{steps}"
    return Path(out_dir) / f"{stem}.zip"


def build_env(num_trajectories, seed, gamma, features="proxy", normalise_action=True):
    """`features=None` returns the bare environment, whose observation is the raw state, for the closed forms."""
    from gym_local.ModelDynamics import AdverseFillModelDynamics
    from gym_local.TradingEnvironment import TradingEnvironment
    from gym_local.observation import InformationSet
    from rewards.RewardFunctions import CjMmCriterion
    from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
    from stochastic_processes.fill_probability_models import ExponentialFillFunction
    from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

    step_size = TERMINAL_TIME / N_STEPS
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=VOLATILITY,
        ou_process=OuMidpriceModel(mean_reversion_level=0.0, mean_reversion_speed=0.05, volatility=20.0,
                                   initial_price=0.0, terminal_time=TERMINAL_TIME, step_size=step_size,
                                   num_trajectories=num_trajectories),
        initial_price=100.0, terminal_time=TERMINAL_TIME, step_size=step_size,
        num_trajectories=num_trajectories)
    dynamics = AdverseFillModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([140.0, 140.0]), gamma, 4, step_size, num_trajectories),
        fill_probability_model=ExponentialFillFunction(1.5, step_size, num_trajectories),
        num_trajectories=num_trajectories, track_adverse_fills=TRACK_ADVERSE_FILLS,
        queue_probability=QUEUE_PROBABILITY, self_impact=SELF_IMPACT)
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS, model_dynamics=dynamics,
        reward_function=CjMmCriterion(per_step_inventory_aversion=PHI,
                                      terminal_inventory_aversion=ALPHA_LIQ, terminal_time=TERMINAL_TIME),
        max_inventory=MAX_INVENTORY, num_trajectories=num_trajectories,
        normalise_action_space=normalise_action, normalise_observation_space=False)
    env.seed(seed)
    return InformationSet(env, features=features, alpha_index=4) if features is not None else env


def run_one(job):
    algorithm, gamma, seed, total_steps, out_dir, n_train = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from agents.LearnedAgents import make_model
    from gym_local.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment

    checkpoints = [c for c in CHECKPOINTS if c <= total_steps] or [total_steps]
    if all(checkpoint_path(out_dir, algorithm, gamma, seed, c).exists() for c in checkpoints):
        return f"{algorithm}/gamma{gamma}/seed{seed}: deja present"
    wrapped = build_env(n_train, seed, gamma)
    model = make_model(algorithm, StableBaselinesTradingEnvironment(wrapped), seed=seed, n_steps=wrapped.n_steps)
    started, done = time.time(), 0
    for checkpoint in checkpoints:
        model.learn(total_timesteps=checkpoint - done, reset_num_timesteps=False, progress_bar=False)
        done = checkpoint
        model.save(str(checkpoint_path(out_dir, algorithm, gamma, seed, checkpoint))[:-4])
    return f"{algorithm}/gamma{gamma}/seed{seed}: {done:,} pas en {time.time() - started:.0f}s"


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
    print(f"{len(jobs)} entrainements, {args.workers} en parallele, {args.steps:,} pas\n"
          f"  balaye  : gamma dans {SENSITIVITIES}\n"
          f"  fixe    : p={QUEUE_PROBABILITY}, zeta={SELF_IMPACT}, traversee={TRACK_ADVERSE_FILLS}, "
          f"sigma={VOLATILITY}", flush=True)

    from multiprocessing import Pool
    started = time.time()
    with Pool(args.workers) as pool:
        for done, message in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            print(f"[{done}/{len(jobs)}] {message}", flush=True)
    print(f"\ntermine en {(time.time() - started) / 60:.0f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
