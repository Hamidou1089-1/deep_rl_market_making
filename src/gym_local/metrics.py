"""Metrics for judging a market making policy, and specifically for telling skill from passivity.

Mean PnL alone cannot answer the question this project asks. A policy that quotes very wide suffers almost no
adverse selection, carries almost no inventory and earns almost nothing; on a criterion that penalises inventory
it can look respectable while having stopped making a market. The metrics below are chosen so that this failure
mode is visible rather than flattering.

The decomposition that does the work is

    markout_total  =  n_fills  x  markout_per_fill

Both factors must be read together.

    passive        n_fills falls, markout_total improves, markout_per_fill roughly unchanged
    skilled        n_fills roughly held, markout_per_fill improves
    reckless       n_fills rises, markout_per_fill worsens

Only the middle row is what a learned agent is being asked for: pick the *same* number of fights and lose fewer
of them, or equivalently, quote where the flow is benign rather than quoting away from all of it.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gym_local.ModelDynamics import AdverseFillModelDynamics
from gym_local.index_names import ASK_INDEX, ASSET_PRICE_INDEX, BID_INDEX, CASH_INDEX, INVENTORY_INDEX

DEFAULT_MARKOUT_HORIZONS = (1, 5, 20)


def _partial_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Correlation between x and y once a linear fit on `control` has been removed from both."""
    design = np.stack([control, np.ones_like(control)], axis=1)
    residuals = []
    for values in (x, y):
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residuals.append(values - design @ coefficients)
    if residuals[0].std() < 1e-12 or residuals[1].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def run_episode(env, agent, observation_is_raw_state: bool = True) -> dict:
    """One vectorised episode, collecting everything the metrics below need.

    `agent.get_action` is called on whatever the environment returns, so this works both for the closed forms,
    which read the raw state, and for a learned policy behind an `InformationSet` wrapper. When the environment
    is wrapped, `observation_is_raw_state=False` makes the closed forms read `env.raw_state` instead.
    """
    observation = env.reset()
    raw = env.raw_state if hasattr(env, "raw_state") else observation
    states, actions, objective = [raw.copy()], [], np.zeros(env.num_trajectories)
    adverse = np.zeros(env.num_trajectories)
    non_adverse = np.zeros(env.num_trajectories)
    executions = np.zeros(env.num_trajectories)
    for _ in range(env.n_steps):
        action = agent.get_action(observation if observation_is_raw_state else env.raw_state)
        observation, reward, done, _ = env.step(action)
        raw = env.raw_state if hasattr(env, "raw_state") else observation
        states.append(raw.copy())
        # Record depths, never the normalised action. A learned policy is trained on a [-1, 1] action space and
        # a closed form is not, so storing the raw action would make `median_depth` incomparable across arms.
        # The normalisation is a linear reparameterisation applied inside `step`; inverting it recovers the
        # depth the environment actually used.
        depths = np.asarray(action, dtype=float)
        if getattr(env, "normalise_action_space_", False):
            depths = env.normalise_action(depths, inverse=True)
        actions.append(np.asarray(depths, dtype=float).copy())
        objective += reward
        if env.last_fills is not None:
            executions += env.last_fills.sum(axis=1)
        dynamics = env.model_dynamics
        if isinstance(dynamics, AdverseFillModelDynamics) and dynamics.track_adverse_fills:
            adverse += dynamics.last_adverse_fills.sum(axis=1)
            non_adverse += dynamics.last_non_adverse_fills.sum(axis=1)
        if done[0]:
            break
    return dict(states=np.stack(states, axis=-1), actions=np.stack(actions, axis=-1), objective=objective,
                executions=executions, adverse=adverse, non_adverse=non_adverse)


def episode_metrics(episode: dict, horizons: Sequence[int] = DEFAULT_MARKOUT_HORIZONS,
                    alpha_index: Optional[int] = None) -> dict:
    """Reduce one episode to the quantities that separate skill from passivity."""
    states, actions = episode["states"], episode["actions"]
    inventory = states[:, INVENTORY_INDEX, :]
    price = states[:, ASSET_PRICE_INDEX, :]
    inventory_change = np.diff(inventory, axis=1)
    terminal_pnl = states[:, CASH_INDEX, -1] + inventory[:, -1] * price[:, -1]
    fills = episode["executions"]
    depths = actions[:, [BID_INDEX, ASK_INDEX], :]

    metrics = {
        "J": float(episode["objective"].mean()),
        "pnl": float(terminal_pnl.mean()),
        "pnl_sd": float(terminal_pnl.std()),
        # Per-episode reward-to-risk, the criterion-free counterpart of J. J embeds one particular price for
        # inventory risk; this one embeds none, so the two disagreeing is informative rather than a problem.
        "pnl_per_risk": float(terminal_pnl.mean() / terminal_pnl.std()) if terminal_pnl.std() > 0 else np.nan,
        # Activity-normalised profit: separates "earns more" from "trades more".
        "pnl_per_fill": float(terminal_pnl.mean() / fills.mean()) if fills.mean() > 0 else np.nan,
        # activity: the number of times the policy actually traded
        "fills": float(fills.mean()),
        "mean_abs_inventory": float(np.abs(inventory).mean()),
        "abs_terminal_inventory": float(np.abs(inventory[:, -1]).mean()),
        # The median, not the mean. At an inventory bound the closed form refuses to trade by quoting an
        # enormous depth -- of order 1e4 -- and a handful of those steps move a mean by more than the policy
        # difference being measured. The share of negative depths is reported alongside because
        # `ExponentialFillFunction` returns exp(-kappa delta) > 1 there, i.e. a certain fill: it is an artefact
        # of quoting through the mid, not a choice, and it should be watched rather than averaged away.
        "median_depth": float(np.nanmedian(depths)),
        "depth_iqr": float(np.nanquantile(depths, 0.75) - np.nanquantile(depths, 0.25)),
        "negative_depth_share": float((depths < 0).mean()),
    }
    total = episode["adverse"] + episode["non_adverse"]
    metrics["adverse_share"] = float(episode["adverse"].sum() / total.sum()) if total.sum() > 0 else np.nan

    usable = inventory_change.shape[1]
    for h in horizons:
        if h >= usable:
            continue
        moves = price[:, h : h + usable - h] - price[:, : usable - h]
        per_trajectory = (inventory_change[:, : usable - h] * moves).sum(axis=1)
        metrics[f"markout_total_h{h}"] = float(per_trajectory.mean())
        # Per fill: the toxicity of the flow the policy chose to take, net of how much of it it took. This is a
        # ratio of means, not a mean of ratios. The difference matters: only the ratio of means satisfies
        # markout_total = fills x markout_per_fill exactly, which is what makes `passivity_decomposition` an
        # identity rather than an approximation.
        mean_fills = float(fills.mean())
        metrics[f"markout_per_fill_h{h}"] = float(per_trajectory.mean() / mean_fills) if mean_fills > 0 else np.nan

    if alpha_index is not None:
        alpha = states[:, alpha_index, :-1]
        skew = actions[:, BID_INDEX, :] - actions[:, ASK_INDEX, :]
        skew = skew[:, : alpha.shape[1]]
        # Skew is bid minus ask, so leaning *against* the signal -- widening the ask when alpha is positive,
        # because that is the side about to be run through -- shows up as a NEGATIVE correlation.
        metrics["skew_alpha_corr"] = float(np.corrcoef(skew.ravel(), alpha.ravel())[0, 1])
        held = inventory[:, : alpha.shape[1]]
        metrics["inventory_alpha_corr"] = float(np.corrcoef(held.ravel(), alpha.ravel())[0, 1])
        # The raw correlation is not evidence that a policy uses the signal. Adverse selection makes inventory
        # itself correlate with alpha, and every inventory-aware policy skews on inventory -- so the closed
        # form, which cannot see alpha at all, still shows a non-zero raw correlation. Partialling inventory out
        # is what isolates the part of the skew that alpha explains and inventory does not.
        metrics["skew_alpha_partial_corr"] = _partial_correlation(
            skew.ravel(), alpha.ravel(), held.ravel())
    return metrics


def evaluate(env_factory, agent_factory, seeds: Sequence[int], horizons=DEFAULT_MARKOUT_HORIZONS,
             alpha_index: Optional[int] = None, observation_is_raw_state: bool = True) -> pd.DataFrame:
    """One row per seed. Seeds are the unit of replication throughout this project: every arm sees the same
    seeds, so any two arms can be compared pairwise, and the spread across seeds is the error bar that matters."""
    rows = []
    for seed in seeds:
        env = env_factory(seed)
        episode = run_episode(env, agent_factory(env), observation_is_raw_state=observation_is_raw_state)
        rows.append({"seed": seed, **episode_metrics(episode, horizons, alpha_index)})
    return pd.DataFrame(rows).set_index("seed")


def summarise(frames: dict) -> pd.DataFrame:
    """Mean and standard error across seeds, one block per arm."""
    rows = {}
    for name, frame in frames.items():
        mean = frame.mean()
        se = frame.std(ddof=1) / np.sqrt(len(frame))
        rows[name] = pd.concat([mean, se.add_suffix("_se")])
    return pd.DataFrame(rows).T


def passivity_decomposition(baseline: pd.DataFrame, arm: pd.DataFrame, horizon: int = 5) -> dict:
    """Split a change in total markout into the part explained by trading less and the part explained by trading
    better, paired seed by seed.

        d(markout_total) = markout_per_fill x d(fills)  +  fills x d(markout_per_fill)

    evaluated at the baseline for the first term and at the arm for the second, so the two sum exactly to the
    change. A policy whose improvement lives entirely in the first term has gone passive.
    """
    total, per_fill, fills = f"markout_total_h{horizon}", f"markout_per_fill_h{horizon}", "fills"
    d_total = (arm[total] - baseline[total])
    activity = baseline[per_fill] * (arm[fills] - baseline[fills])
    quality = arm[fills] * (arm[per_fill] - baseline[per_fill])
    n = len(d_total)
    return {
        "d_markout_total": float(d_total.mean()),
        "d_markout_total_se": float(d_total.std(ddof=1) / np.sqrt(n)),
        "from_trading_less": float(activity.mean()),
        "from_trading_better": float(quality.mean()),
        "from_trading_better_se": float(quality.std(ddof=1) / np.sqrt(n)),
        "d_fills_pct": float(100 * (arm[fills] / baseline[fills] - 1).mean()),
    }
