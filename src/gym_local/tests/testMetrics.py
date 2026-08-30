"""Tests for the metrics that separate skill from passivity.

Mean PnL cannot answer the question this project asks: a policy that quotes very wide suffers little adverse
selection, carries little inventory and earns little, and on an inventory-penalised criterion it can look
respectable while having stopped making a market. The decomposition

    d(markout_total) = markout_per_fill x d(fills)  +  fills x d(markout_per_fill)

is what makes that failure visible. These tests pin it down on two policies whose behaviour is known by
construction: a uniform widening, which carries no information and can only trade *less*, and a signal-aware
widening, which can trade *better*.
"""

from unittest import TestCase, main

import numpy as np

from agents.Agent import Agent
from agents.BaselineAgents import CarteaJaimungalMmAgent
from gym_local.ModelDynamics import AdverseFillModelDynamics
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.index_names import ASK_INDEX, BID_INDEX
from gym_local.metrics import episode_metrics, evaluate, passivity_decomposition, run_episode
from rewards.RewardFunctions import CjMmCriterion
from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

TERMINAL_TIME, N_STEPS = 1.0, 200
STEP_SIZE = TERMINAL_TIME / N_STEPS
ARRIVAL_RATE, VOLATILITY = 140.0, 2.0
FILL_EXPONENT = 5.0  # rho ~ 1.4, where the adverse branch is active and the closed form is still profitable
ALPHA_SPEED, ALPHA_VOLATILITY, ALPHA_INDEX = 0.05, 20.0, 4
SEEDS = (7, 17, 27)


def get_env(n, seed, sensitivity=0.2):
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=VOLATILITY,
        ou_process=OuMidpriceModel(
            mean_reversion_level=0.0, mean_reversion_speed=ALPHA_SPEED, volatility=ALPHA_VOLATILITY,
            initial_price=0.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE, num_trajectories=n),
        initial_price=100.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE, num_trajectories=n)
    dynamics = AdverseFillModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([ARRIVAL_RATE, ARRIVAL_RATE]), sensitivity, ALPHA_INDEX, STEP_SIZE, n),
        fill_probability_model=ExponentialFillFunction(FILL_EXPONENT, STEP_SIZE, n),
        num_trajectories=n, track_adverse_fills=True, queue_probability=1.0)
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS, model_dynamics=dynamics,
        reward_function=CjMmCriterion(per_step_inventory_aversion=0.01, terminal_inventory_aversion=1.0,
                                      terminal_time=TERMINAL_TIME),
        max_inventory=20, num_trajectories=n, normalise_action_space=False, normalise_observation_space=False)
    env.seed(seed)
    return env


class WiderCj(Agent):
    """Carries no information: it can only trade less, never better."""

    def __init__(self, env, widening):
        self.inner, self.widening = CarteaJaimungalMmAgent(env=env), widening

    def get_action(self, state):
        return self.inner.get_action(state) + self.widening


class SignalAwareCj(Agent):
    """Widens only the side the drift is about to run into."""

    def __init__(self, env, c):
        self.inner, self.c, self.dt = CarteaJaimungalMmAgent(env=env), c, env.step_size

    def get_action(self, state):
        action = self.inner.get_action(state).copy()
        alpha = state[:, ALPHA_INDEX]
        action[:, ASK_INDEX] += self.c * np.maximum(alpha, 0.0) * self.dt
        action[:, BID_INDEX] += self.c * np.maximum(-alpha, 0.0) * self.dt
        return action


def frames(n=600):
    factory = lambda seed: get_env(n, seed)
    return {
        "closed_form": evaluate(factory, lambda env: CarteaJaimungalMmAgent(env=env), SEEDS, alpha_index=ALPHA_INDEX),
        "wider": evaluate(factory, lambda env: WiderCj(env, 0.05), SEEDS, alpha_index=ALPHA_INDEX),
        "signal_aware": evaluate(factory, lambda env: SignalAwareCj(env, 4.0), SEEDS, alpha_index=ALPHA_INDEX),
    }


class testTheDecompositionIsExact(TestCase):
    def test_the_two_terms_sum_to_the_change(self):
        collected = frames()
        for name in ("wider", "signal_aware"):
            d = passivity_decomposition(collected["closed_form"], collected[name], horizon=5)
            self.assertAlmostEqual(d["from_trading_less"] + d["from_trading_better"], d["d_markout_total"], places=8)

    def test_a_policy_compared_with_itself_moves_nothing(self):
        collected = frames()
        d = passivity_decomposition(collected["closed_form"], collected["closed_form"], horizon=5)
        for key in ("d_markout_total", "from_trading_less", "from_trading_better", "d_fills_pct"):
            self.assertAlmostEqual(d[key], 0.0, places=10)


class testPassivityIsDistinguishedFromSkill(TestCase):
    def setUp(self):
        self.frames = frames()

    def test_both_policies_reduce_total_adverse_selection(self):
        """Total markout is the wrong statistic on its own: quoting wider improves it, and so does using the
        signal. That both do is exactly why the decomposition is needed to tell them apart."""
        for name in ("wider", "signal_aware"):
            d = passivity_decomposition(self.frames["closed_form"], self.frames[name], horizon=5)
            self.assertGreater(d["d_markout_total"], 0.0)
            self.assertLess(d["d_fills_pct"], 0.0)

    def test_the_uniform_widening_owes_its_improvement_to_trading_less(self):
        """It carries no information, so almost all of what it gains must come from taking fewer fills. The
        residual is not asserted to be zero or negative: quoting wider also puts the quote further from the
        price, which removes some adverse fills too, and how much depends on the regime."""
        d = passivity_decomposition(self.frames["closed_form"], self.frames["wider"], horizon=5)
        share = d["from_trading_better"] / d["d_markout_total"]
        self.assertLess(share, 0.30, "an uninformed policy should not be improving the flow it selects")

    def test_the_signal_aware_widening_owes_much_more_to_trading_better(self):
        signal = passivity_decomposition(self.frames["closed_form"], self.frames["signal_aware"], horizon=5)
        uniform = passivity_decomposition(self.frames["closed_form"], self.frames["wider"], horizon=5)
        self.assertGreater(signal["from_trading_better"], 0.0)
        self.assertGreater(signal["from_trading_better"] / signal["d_markout_total"],
                           2 * uniform["from_trading_better"] / uniform["d_markout_total"])

    def test_the_signal_aware_policy_cuts_fewer_fills_and_still_gains_more(self):
        """The signature of skill rather than retreat: a larger improvement for a smaller loss of activity."""
        signal = passivity_decomposition(self.frames["closed_form"], self.frames["signal_aware"], horizon=5)
        uniform = passivity_decomposition(self.frames["closed_form"], self.frames["wider"], horizon=5)
        self.assertGreater(signal["d_markout_total"], uniform["d_markout_total"])
        self.assertGreater(signal["d_fills_pct"], uniform["d_fills_pct"])

    def test_per_fill_toxicity_improves_only_for_the_informed_policy(self):
        per_fill = {name: frame["markout_per_fill_h5"].mean() for name, frame in self.frames.items()}
        self.assertGreater(per_fill["signal_aware"], per_fill["closed_form"])
        self.assertGreater(per_fill["signal_aware"], per_fill["wider"])


class testTheActivityAndDepthMetrics(TestCase):
    def test_widening_raises_the_median_depth_and_lowers_the_fill_count(self):
        """The mean depth cannot be used for this: at an inventory bound the closed form refuses to trade by
        quoting of order 1e4, and a handful of such steps swamp any policy difference."""
        collected = frames()
        self.assertGreater(collected["wider"]["median_depth"].mean(),
                           collected["closed_form"]["median_depth"].mean())
        self.assertLess(collected["wider"]["fills"].mean(), collected["closed_form"]["fills"].mean())

    def test_the_median_moves_by_the_widening_while_the_mean_does_not(self):
        env = get_env(400, 7)
        base = episode_metrics(run_episode(env, CarteaJaimungalMmAgent(env=env)), alpha_index=ALPHA_INDEX)
        env = get_env(400, 7)
        wider = episode_metrics(run_episode(env, WiderCj(env, 0.05)), alpha_index=ALPHA_INDEX)
        self.assertAlmostEqual(wider["median_depth"] - base["median_depth"], 0.05, places=2)

    def test_the_raw_skew_correlation_is_not_evidence_of_using_the_signal(self):
        """Adverse selection makes inventory itself correlate with alpha, and every inventory-aware policy skews
        on inventory. So a policy that cannot see alpha still shows a raw correlation with it, and the raw
        statistic would credit it with information it does not have."""
        collected = frames()
        self.assertLess(collected["closed_form"]["inventory_alpha_corr"].mean(), -0.20,
                        "inventory must be correlated with the signal, or the confound would not exist")
        self.assertGreater(abs(collected["wider"]["skew_alpha_corr"].mean()), 0.10,
                           "an uninformed policy shows a non-trivial raw correlation")

    def test_partialling_out_inventory_isolates_the_use_of_the_signal(self):
        """`skew` is bid minus ask, so leaning against the signal is a negative partial correlation."""
        collected = frames()
        self.assertLess(abs(collected["closed_form"]["skew_alpha_partial_corr"].mean()), 0.05,
                        "the closed form cannot see alpha and must score near zero once inventory is removed")
        self.assertLess(abs(collected["wider"]["skew_alpha_partial_corr"].mean()), 0.20)
        self.assertLess(collected["signal_aware"]["skew_alpha_partial_corr"].mean(), -0.50,
                        "the oracle rule must be clearly identified as using the signal")

    def test_the_negative_depth_artefact_is_reported_and_rare(self):
        """`ExponentialFillFunction` returns exp(-kappa delta) > 1 for a negative depth, i.e. a certain fill. It
        is an artefact of the closed form quoting through the mid at large inventory, and it must be visible."""
        collected = frames()
        share = collected["closed_form"]["negative_depth_share"].mean()
        self.assertGreater(share, 0.0, "the artefact exists and the metric must not hide it")
        self.assertLess(share, 0.05, "if it ever became common the comparison would be measuring the artefact")


class testTheCriterionFreeMetrics(TestCase):
    """`J` prices inventory risk at one particular alpha_liq. These two do not, so they are what makes a
    sensitivity analysis in alpha_liq readable: they change only when behaviour changes, not when the scoring
    changes."""

    def test_reward_to_risk_and_profit_per_fill_are_consistent_with_their_parts(self):
        collected = frames()
        for frame in collected.values():
            np.testing.assert_allclose(frame["pnl_per_risk"], frame["pnl"] / frame["pnl_sd"], rtol=1e-9)
            np.testing.assert_allclose(frame["pnl_per_fill"], frame["pnl"] / frame["fills"], rtol=1e-9)

    def test_a_uniform_widening_raises_profit_per_fill_while_trading_less(self):
        """Quoting wider earns more on each trade it still does. That alone is not skill, which is why it is
        reported next to the fill count rather than instead of it."""
        collected = frames()
        self.assertGreater(collected["wider"]["pnl_per_fill"].mean(),
                           collected["closed_form"]["pnl_per_fill"].mean())
        self.assertLess(collected["wider"]["fills"].mean(), collected["closed_form"]["fills"].mean())


if __name__ == "__main__":
    main()
