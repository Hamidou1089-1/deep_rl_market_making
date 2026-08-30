"""Tests for the recalibrated closed-form arm.

The point of `RecalibratedCarteaJaimungalMmAgent` is to give a learned agent an honest opponent: not the closed form
carrying the parameters of an environment it is no longer in, but the best policy the closed-form family can
produce in the environment it is actually being scored in. Three things have to hold:

  * overriding nothing must reproduce `CarteaJaimungalMmAgent` exactly, so the naive arm is genuinely nested;
  * the overrides must move the policy in the direction they claim to;
  * in the nested environment, where the closed form is the derived optimum, recalibration must *not* find a
    materially better parameterisation. If it does, the search is fitting evaluation noise rather than dynamics,
    and every gain it reports elsewhere is suspect.
"""

from unittest import TestCase, main

import numpy as np

from agents.BaselineAgents import AvellanedaStoikovAgent, CarteaJaimungalMmAgent
from agents.calibration import (
    RecalibratedAvellanedaStoikovAgent,
    RecalibratedCarteaJaimungalMmAgent,
    evaluate_parameters,
    evaluate_policy,
    grid_search,
)
from gym_local.ModelDynamics import LimitOrderModelDynamics
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.index_names import BID_INDEX, ASK_INDEX, TIME_INDEX, INVENTORY_INDEX
from rewards.RewardFunctions import CjMmCriterion, PnL
from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

TERMINAL_TIME, N_STEPS = 1.0, 200
STEP_SIZE = TERMINAL_TIME / N_STEPS
FILL_EXPONENT, ARRIVAL_RATE = 1.5, 140.0
PER_STEP_INVENTORY_AVERSION, TERMINAL_INVENTORY_AVERSION = 0.01, 1.0
MAX_INVENTORY = 20
ALPHA_INDEX = 4

TRUE_PARAMETERS = dict(
    fill_exponent=FILL_EXPONENT,
    per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
    terminal_inventory_aversion=TERMINAL_INVENTORY_AVERSION,
)


def get_env(sensitivity, num_trajectories, seed=0, reward_function=None):
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=2.0,
        ou_process=OuMidpriceModel(
            mean_reversion_level=0.0, mean_reversion_speed=0.05, volatility=20.0, initial_price=0.0,
            terminal_time=TERMINAL_TIME, step_size=STEP_SIZE, num_trajectories=num_trajectories,
        ),
        initial_price=100.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE,
        num_trajectories=num_trajectories,
    )
    model_dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([ARRIVAL_RATE, ARRIVAL_RATE]), sensitivity, ALPHA_INDEX, STEP_SIZE, num_trajectories),
        fill_probability_model=ExponentialFillFunction(FILL_EXPONENT, STEP_SIZE, num_trajectories),
        num_trajectories=num_trajectories,
    )
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS, model_dynamics=model_dynamics,
        num_trajectories=num_trajectories, max_inventory=MAX_INVENTORY,
        reward_function=reward_function or CjMmCriterion(
            per_step_inventory_aversion=PER_STEP_INVENTORY_AVERSION,
            terminal_inventory_aversion=TERMINAL_INVENTORY_AVERSION, terminal_time=TERMINAL_TIME),
        normalise_action_space=False, normalise_observation_space=False,
    )
    env.seed(seed)
    return env


def get_state(env, inventories, current_time=0.5):
    state = np.zeros((len(inventories), 5))
    state[:, TIME_INDEX] = current_time
    state[:, INVENTORY_INDEX] = inventories
    return state


class testRecalibratedCarteaJaimungalMmAgent(TestCase):
    def test_overriding_nothing_reproduces_the_naive_closed_form(self):
        """The naive arm has to be nested in the recalibrated family, otherwise the two arms are not comparable."""
        env = get_env(sensitivity=0.2, num_trajectories=9)
        state = get_state(env, np.arange(-4, 5))
        np.testing.assert_array_equal(
            RecalibratedCarteaJaimungalMmAgent(env=env).get_action(state),
            CarteaJaimungalMmAgent(env=env).get_action(state),
        )

    def test_parameters_property_reports_what_the_policy_actually_uses(self):
        env = get_env(sensitivity=0.2, num_trajectories=3)
        agent = RecalibratedCarteaJaimungalMmAgent(
            env=env, fill_exponent=1.1, per_step_inventory_aversion=0.7, terminal_inventory_aversion=2.5)
        self.assertEqual(agent.parameters,
                         {"fill_exponent": 1.1, "intensity": ARRIVAL_RATE,
                          "per_step_inventory_aversion": 0.7, "terminal_inventory_aversion": 2.5})
        # Defaults still come from the environment when they are not overridden.
        self.assertEqual(RecalibratedCarteaJaimungalMmAgent(env=env).parameters, {
            "fill_exponent": FILL_EXPONENT, "intensity": ARRIVAL_RATE,
            "per_step_inventory_aversion": PER_STEP_INVENTORY_AVERSION,
            "terminal_inventory_aversion": TERMINAL_INVENTORY_AVERSION})

    def test_a_smaller_fill_exponent_widens_the_quoted_spread(self):
        env = get_env(sensitivity=0.2, num_trajectories=1)
        state = get_state(env, [0])
        spreads = [RecalibratedCarteaJaimungalMmAgent(env=env, fill_exponent=k).get_action(state).sum()
                   for k in (0.75, 1.0, 1.5, 2.0)]
        self.assertEqual(spreads, sorted(spreads, reverse=True), "spread should shrink as kappa grows")

    def test_a_larger_inventory_aversion_skews_harder(self):
        """The skew is what the closed form uses to shed inventory. Being internally more averse must make the two
        sides of the quote diverge faster as inventory builds up."""
        env = get_env(sensitivity=0.2, num_trajectories=1)
        state = get_state(env, [3])
        skews = []
        for phi in (0.01, 0.1, 1.0, 5.0):
            action = RecalibratedCarteaJaimungalMmAgent(env=env, per_step_inventory_aversion=phi).get_action(state)
            skews.append(float(action[0, BID_INDEX] - action[0, ASK_INDEX]))
        self.assertEqual(skews, sorted(skews), "a long inventory should push the bid away faster as phi grows")
        self.assertGreater(skews[-1], skews[0])

    def test_numerically_unusable_parameters_are_detected(self):
        env = get_env(sensitivity=0.2, num_trajectories=1)
        self.assertTrue(RecalibratedCarteaJaimungalMmAgent(env=env, **TRUE_PARAMETERS).is_numerically_usable())
        # The lambda * exp(-1) off-diagonals mix mass across inventories, so even absurd aversions stay finite.
        self.assertTrue(
            RecalibratedCarteaJaimungalMmAgent(env=env, terminal_inventory_aversion=1e4).is_numerically_usable())
        # Starve that mixing and omega underflows at the inventories the terminal penalty has already zeroed.
        self.assertFalse(RecalibratedCarteaJaimungalMmAgent(
            env=env, intensity=np.array([1e-8, 1e-8]), terminal_inventory_aversion=50.0).is_numerically_usable())

    def test_recalibration_requires_the_risk_averse_branch(self):
        env = get_env(sensitivity=0.2, num_trajectories=1, reward_function=PnL())
        with self.assertRaises(AssertionError):
            RecalibratedCarteaJaimungalMmAgent(env=env, per_step_inventory_aversion=1.0)


class testSearchMachinery(TestCase):
    def test_evaluate_policy_reports_the_environment_own_objective(self):
        env = get_env(sensitivity=0.2, num_trajectories=200, seed=5)
        metrics = evaluate_policy(env, RecalibratedCarteaJaimungalMmAgent(env=env), seed=5)
        self.assertEqual(set(metrics), {"objective", "mean_pnl", "pnl_std",
                                        "mean_abs_terminal_inventory", "max_abs_inventory"})
        # The criterion is PnL minus inventory penalties, so it can never exceed the PnL it is built from.
        self.assertLess(metrics["objective"], metrics["mean_pnl"])

    def test_evaluate_parameters_accepts_an_explicit_agent_factory(self):
        env_factory = lambda seed: get_env(sensitivity=0.2, num_trajectories=200, seed=seed)
        by_parameters = evaluate_parameters(env_factory, [3, 4], **TRUE_PARAMETERS)
        by_factory = evaluate_parameters(env_factory, [3, 4], agent_factory=CarteaJaimungalMmAgent)
        self.assertAlmostEqual(by_parameters["objective"], by_factory["objective"], places=10)

    def test_grid_search_is_sorted_and_skips_unusable_candidates(self):
        env_factory = lambda seed: get_env(sensitivity=0.2, num_trajectories=200, seed=seed)
        grid = {"intensity": [ARRIVAL_RATE, 1e-8], "terminal_inventory_aversion": [1.0, 50.0]}
        rows = grid_search(env_factory, grid, fit_seeds=[3, 4])
        self.assertEqual(len(rows), 3, "the starved-intensity, high-alpha candidate should have been skipped")
        objectives = [row["objective"] for row in rows]
        self.assertEqual(objectives, sorted(objectives, reverse=True))


# The winner of the grid search run in Test_4, pinned here so these tests do not have to redo the search.
RICH_WINNER = dict(fill_exponent=1.375, intensity=55.0,
                   per_step_inventory_aversion=0.5, terminal_inventory_aversion=3.0)
FALSIFICATION_SEEDS = [7, 17, 27]  # disjoint from the fit seeds the winner was found on


class testFalsification(TestCase):
    """The two claims the recalibrated arm rests on. Both are statistical, both are measured on seeds the search
    never saw, and both have effects far larger than the noise: the margins asserted here are roughly a tenth of
    the measured gaps, so they pin the sign and the order of magnitude without being flaky."""

    @staticmethod
    def _objective(sensitivity, parameters, num_trajectories=800):
        return evaluate_parameters(
            lambda seed: get_env(sensitivity, num_trajectories, seed), FALSIFICATION_SEEDS, **parameters
        )["objective"]

    def test_recalibration_beats_the_naive_arm_in_the_richer_environment(self):
        naive = self._objective(0.2, TRUE_PARAMETERS)
        recalibrated = self._objective(0.2, RICH_WINNER)
        self.assertGreater(recalibrated - naive, 1.0,
                           f"recalibration should recover a large part of the loss (measured ~ +12.6), "
                           f"got {recalibrated - naive:+.2f}")

    def test_the_recalibrated_parameters_lose_in_the_nested_environment(self):
        """If the winner of the richer environment also won in the nested one, the search would have found a
        better prior rather than an adaptation to adverse selection, and the whole comparison would be hollow."""
        true_parameters = self._objective(0.0, TRUE_PARAMETERS)
        rich_winner = self._objective(0.0, RICH_WINNER)
        self.assertGreater(true_parameters - rich_winner, 0.3,
                           f"the closed form is the derived optimum in the nested environment, so its own "
                           f"parameters must win there (measured gap ~ 3.1), got {true_parameters - rich_winner:+.2f}")


class testRecalibratedAvellanedaStoikovAgent(TestCase):
    """The Avellaneda-Stoikov arm of the same discipline.

    One difference from the Cartea-Jaimungal case is worth keeping in view. AS maximises the exponential utility
    of terminal wealth, not the criterion this project scores on, so refitting its risk aversion both adapts the
    policy to a richer environment *and* translates between two notions of risk aversion. The tests below pin
    down what the class does mechanically; they do not license reading the fitted `risk_aversion` as an estimate.
    """

    def test_overriding_nothing_reproduces_the_base_class(self):
        env = get_env(0.2, 200, seed=3)
        state = get_state(env, np.linspace(-8, 8, 200))
        np.testing.assert_allclose(
            RecalibratedAvellanedaStoikovAgent(env=env).get_action(state),
            AvellanedaStoikovAgent(env=env).get_action(state))

    def test_each_of_the_three_handles_moves_the_quotes(self):
        env = get_env(0.2, 200, seed=3)
        state = get_state(env, np.linspace(-8, 8, 200))
        reference = RecalibratedAvellanedaStoikovAgent(env=env).get_action(state)
        for override in (dict(risk_aversion=0.5), dict(fill_exponent=3.0), dict(volatility=5.0)):
            action = RecalibratedAvellanedaStoikovAgent(env=env, **override).get_action(state)
            self.assertFalse(np.allclose(action, reference), f"{override} had no effect")

    def test_the_reported_parameters_are_the_ones_in_use(self):
        env = get_env(0.2, 20, seed=3)
        override = dict(fill_exponent=2.5, volatility=3.0, risk_aversion=0.25)
        self.assertEqual(RecalibratedAvellanedaStoikovAgent(env=env, **override).parameters, override)

    def test_a_larger_risk_aversion_leans_harder_against_inventory(self):
        """The one economic property the family guarantees: the price adjustment is proportional to the risk
        aversion, so the skew against inventory must grow with it."""
        env = get_env(0.2, 200, seed=3)
        inventories = np.linspace(-8, 8, 200)
        state = get_state(env, inventories)
        slopes = []
        for risk_aversion in (0.01, 0.1, 0.5):
            action = RecalibratedAvellanedaStoikovAgent(env=env, risk_aversion=risk_aversion).get_action(state)
            skew = action[:, BID_INDEX] - action[:, ASK_INDEX]
            slopes.append(np.polyfit(inventories, skew, 1)[0])
        self.assertGreater(slopes[1], slopes[0])
        self.assertGreater(slopes[2], slopes[1])


class testTheSearchAcceptsBothFamilies(TestCase):
    def test_a_grid_over_avellaneda_stoikov_parameters_runs_and_ranks(self):
        results = grid_search(
            lambda seed: get_env(0.2, 150, seed=seed), {"risk_aversion": [0.01, 0.1, 0.5]}, [11, 22],
            agent_class=RecalibratedAvellanedaStoikovAgent)
        self.assertEqual(len(results), 3)
        self.assertEqual([row["objective"] for row in results],
                         sorted([row["objective"] for row in results], reverse=True))

    def test_a_parameter_of_the_other_family_is_refused(self):
        """`intensity` is a Cartea-Jaimungal handle and does nothing in the Avellaneda-Stoikov formulas. Silently
        accepting it would produce a grid whose rows differ only by noise."""
        with self.assertRaises(AssertionError):
            grid_search(lambda seed: get_env(0.2, 50, seed=seed), {"intensity": [50.0, 100.0]}, [11],
                        agent_class=RecalibratedAvellanedaStoikovAgent)

    def test_the_default_family_is_still_cartea_jaimungal(self):
        results = grid_search(lambda seed: get_env(0.2, 150, seed=seed),
                              {"per_step_inventory_aversion": [0.01, 0.1]}, [11])
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    main()
