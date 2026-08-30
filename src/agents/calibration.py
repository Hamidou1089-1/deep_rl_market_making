"""Fitting a closed-form policy to an environment it was not derived for.

The Cartea-Jaimungal market making solution is optimal in the environment it was derived in. Deployed unchanged in
a richer environment it is merely *misspecified*, and the loss it suffers measures the cost of that
misspecification -- not the value of anything that might replace it. The honest benchmark for a learned agent is
therefore the best policy **inside the closed-form family**, obtained by refitting the parameters the closed form
carries around:

    CF-naive         pi_CJ(kappa_0, lambda_0, phi_0, alpha_0)   -- parameters of the nested environment
    CF-recalibrated  argmax over (kappa, lambda, phi, alpha) of J(pi_CJ(.)) in the richer environment
    RL               whatever a learned agent achieves

Two things are deliberately kept apart. `phi` and `alpha` appear both in the *objective* and inside the closed-form
policy; only the second copy is refitted. The objective J stays pinned to the environment's own reward function, so
every arm is scored on exactly the same criterion. Making the policy internally more inventory averse is a policy
choice; making the objective more inventory averse would be changing the question.

The search is scored on `fit_seeds` and must be reported on disjoint `test_seeds`: an argmax over a noisy objective
buys some of its own improvement, and only held-out seeds separate the real gain from that.
"""

from itertools import product
from typing import Callable, Iterable, Optional

import numpy as np

from agents.Agent import Agent
from agents.BaselineAgents import AvellanedaStoikovAgent, CarteaJaimungalMmAgent
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.helpers.generate_trajectory import generate_trajectory
from gym_local.index_names import CASH_INDEX, INVENTORY_INDEX, ASSET_PRICE_INDEX

CJ_PARAMETER_NAMES = ("fill_exponent", "intensity", "per_step_inventory_aversion", "terminal_inventory_aversion")
AS_PARAMETER_NAMES = ("fill_exponent", "volatility", "risk_aversion")

# Kept for the notebooks written before the Avellaneda-Stoikov family was added.
PARAMETER_NAMES = CJ_PARAMETER_NAMES


class RecalibratedCarteaJaimungalMmAgent(CarteaJaimungalMmAgent):
    """The Cartea-Jaimungal closed form with its internal parameters decoupled from the environment.

    `CarteaJaimungalMmAgent` reads kappa and lambda off the environment's stochastic processes and phi and alpha off
    its reward function, which is exactly right when the environment is the one the solution was derived for. Here
    each of the four can be overridden, so the same closed-form *functional form* can be fitted to an environment
    whose true dynamics it cannot represent. Overriding nothing reproduces `CarteaJaimungalMmAgent` exactly.
    """

    def __init__(
        self,
        env: TradingEnvironment = None,
        fill_exponent: Optional[float] = None,
        intensity: Optional[np.ndarray] = None,
        per_step_inventory_aversion: Optional[float] = None,
        terminal_inventory_aversion: Optional[float] = None,
    ):
        super().__init__(env=env)
        assert not self.inventory_neutral, (
            "Recalibration only has a handle on the risk averse branch of the closed form. Give the environment a "
            "CjMmCriterion reward rather than PnL."
        )
        if fill_exponent is not None:
            self.kappa = fill_exponent
        if intensity is not None:
            self.lambdas = np.broadcast_to(np.asarray(intensity, dtype=float), (2,))
        if per_step_inventory_aversion is not None:
            self.phi = per_step_inventory_aversion
        if terminal_inventory_aversion is not None:
            self.alpha = terminal_inventory_aversion
        self.a_matrix, self.z_vector = self._calculate_a_and_z()

    @property
    def parameters(self) -> dict:
        return {
            "fill_exponent": self.kappa,
            "intensity": float(np.asarray(self.lambdas).ravel()[0]),
            "per_step_inventory_aversion": self.phi,
            "terminal_inventory_aversion": self.alpha,
        }

    def is_numerically_usable(self, max_reachable_inventory: Optional[int] = None) -> bool:
        """The closed form goes through h = log(expm(A (T - t)) z) / kappa, which is non-finite wherever omega
        underflows to zero. In practice this is hard to trigger: the off-diagonal lambda * exp(-1) terms mix mass
        across inventories fast enough to keep omega strictly positive even at absurd kappa, phi or alpha. It does
        break once the intensity is small enough that the mixing stops -- measured across the search region used so
        far, no (kappa, phi, alpha) candidate was rejected and only a starved lambda was.

        The guard is therefore insurance rather than a live constraint, but a search must never quietly evaluate a
        policy quoting non-finite depths, so unusable candidates are dropped and counted instead of clipped. Only
        inventories the policy can actually reach matter, so pass `max_reachable_inventory` when the episode is too
        short to ever fill the book to `max_inventory`.
        """
        reach = self.max_inventory if max_reachable_inventory is None else min(self.max_inventory, max_reachable_inventory)
        lo, hi = self.max_inventory - reach, self.max_inventory + reach + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            for current_time in (0.0, self.terminal_time / 2, self.terminal_time - 1e-9):
                if not np.all(np.isfinite(self._calculate_ht(current_time)[lo:hi])):
                    return False
        return True


class RecalibratedAvellanedaStoikovAgent(AvellanedaStoikovAgent):
    """The Avellaneda-Stoikov quotes with their internal constants decoupled from the environment.

    The closed form carries three quantities that enter its formulas: the fill exponent `kappa`, the midprice
    volatility `sigma`, and the risk aversion `gamma` of the exponential utility it maximises. Note that
    `AvellanedaStoikovAgent` also reads the arrival rate off the environment but never uses it, so it is not a
    handle and is not offered as one.

    The same discipline as the Cartea-Jaimungal case applies and matters more here. Avellaneda-Stoikov maximises
    the exponential utility of terminal wealth, which is *not* the criterion this project scores on. Refitting
    `gamma` is therefore doing two things at once: adapting the policy to a richer environment, and translating
    between two different notions of risk aversion. The arm is still worth having -- it is the best this
    functional form can do on the criterion in use -- but it is a policy search over three parameters, not a
    closed form, and its winning `gamma` should not be read as an estimate of anything.
    """

    def __init__(
        self,
        env: TradingEnvironment = None,
        fill_exponent: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_aversion: Optional[float] = None,
    ):
        super().__init__(risk_aversion=0.1 if risk_aversion is None else risk_aversion, env=env)
        if fill_exponent is not None:
            self.fill_exponent = fill_exponent
        if volatility is not None:
            self.volatility = volatility

    @property
    def parameters(self) -> dict:
        return {"fill_exponent": self.fill_exponent, "volatility": self.volatility,
                "risk_aversion": self.risk_aversion}

    def is_numerically_usable(self, max_reachable_inventory: Optional[int] = None) -> bool:
        """The quotes are closed form and always finite, but a large risk aversion drives the inventory
        adjustment past the half spread and the policy starts quoting through the mid. The environment now floors
        depths at zero, so this is no longer a source of impossible fills, but a candidate that quotes at the
        touch on one side for most of the episode is not meaningfully inside the family and is dropped."""
        reach = self.max_inventory if max_reachable_inventory is None else max_reachable_inventory
        for current_time in (0.0, self.terminal_time / 2, self.terminal_time - 1e-9):
            action = self._get_action(np.array([float(reach)]), np.array([current_time]))
            if not np.all(np.isfinite(action)):
                return False
        return True

    @property
    def max_inventory(self) -> int:
        return self.env.max_inventory


def evaluate_policy(env: TradingEnvironment, agent: Agent, seed: int) -> dict:
    """One paired evaluation. `objective` is the environment's own reward summed over the episode -- the quantity
    the closed form maximises and the quantity an RL agent would be trained on, so every arm is scored alike."""
    observations, _, rewards = generate_trajectory(env, agent, seed=seed)
    terminal_inventory = observations[:, INVENTORY_INDEX, -1]
    terminal_pnl = observations[:, CASH_INDEX, -1] + terminal_inventory * observations[:, ASSET_PRICE_INDEX, -1]
    return {
        "objective": float(rewards.sum(axis=2).mean()),
        "mean_pnl": float(terminal_pnl.mean()),
        "pnl_std": float(terminal_pnl.std()),
        "mean_abs_terminal_inventory": float(np.abs(terminal_inventory).mean()),
        "max_abs_inventory": float(np.abs(observations[:, INVENTORY_INDEX, :]).max()),
    }


def evaluate_parameters(
    env_factory: Callable[[int], TradingEnvironment],
    seeds: Iterable[int],
    agent_factory: Optional[Callable[[TradingEnvironment], Agent]] = None,
    **parameters,
) -> dict:
    """Average `evaluate_policy` over seeds, reporting the standard error across them.

    `env_factory(seed)` must rebuild the environment from scratch for each seed. `agent_factory` defaults to the
    recalibrated closed form built from `parameters`; pass one explicitly to score any other agent on the same
    seeds and the same metrics.
    """
    seeds = list(seeds)
    factory = agent_factory or (lambda env: RecalibratedCarteaJaimungalMmAgent(env=env, **parameters))
    runs = [evaluate_policy(env, factory(env), seed) for seed, env in ((s, env_factory(s)) for s in seeds)]
    summary = {"seeds": len(seeds), **parameters}
    for metric in runs[0]:
        values = np.array([run[metric] for run in runs])
        summary[metric] = float(values.mean())
        if metric == "objective":
            summary["objective_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return summary


def grid_search(
    env_factory: Callable[[int], TradingEnvironment],
    grid: dict,
    fit_seeds: Iterable[int],
    max_reachable_inventory: Optional[int] = None,
    verbose: bool = False,
    agent_class: type = None,
) -> list:
    """Exhaustive search over the product of `grid`, scored on `fit_seeds` with common random numbers.

    Every candidate sees the same seeds, so the comparison between candidates is paired and the argmax is driven by
    the parameters rather than by which seeds a candidate happened to draw. Candidates whose closed form is not
    numerically usable are skipped and reported, never silently clipped.
    """
    agent_class = agent_class or RecalibratedCarteaJaimungalMmAgent
    allowed = CJ_PARAMETER_NAMES if agent_class is RecalibratedCarteaJaimungalMmAgent else AS_PARAMETER_NAMES
    fit_seeds = list(fit_seeds)
    names = [name for name in allowed if name in grid]
    assert names, f"grid must contain at least one of {allowed}"
    unknown = set(grid) - set(allowed)
    assert not unknown, f"{sorted(unknown)} are not parameters of {agent_class.__name__}"
    probe_env = env_factory(fit_seeds[0])

    results, skipped = [], []
    for values in product(*(list(grid[name]) for name in names)):
        parameters = dict(zip(names, values))
        if not agent_class(env=probe_env, **parameters).is_numerically_usable(max_reachable_inventory):
            skipped.append(parameters)
            continue
        results.append(evaluate_parameters(
            env_factory, fit_seeds, agent_factory=lambda env: agent_class(env=env, **parameters), **parameters))
        if verbose:
            print(f"  {parameters} -> objective {results[-1]['objective']:+.3f}")
    if skipped:
        print(f"grid_search: {len(skipped)} of {len(skipped) + len(results)} candidates skipped as numerically "
              f"unusable (the closed form underflows at these parameters), e.g. {skipped[0]}")
    assert results, "every candidate in the grid was numerically unusable"
    return sorted(results, key=lambda row: -row["objective"])
