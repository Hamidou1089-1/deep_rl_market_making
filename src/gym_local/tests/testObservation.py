"""Tests for the information set wrapper.

Three things must hold for the comparison across information sets to mean anything.

  * **The features are what they claim.** A proxy that does not actually track the latent factor would make the
    "the agent must infer alpha" arm a test of nothing.
  * **The sets are nested and the environment is untouched.** Changing what the agent sees must not change the
    world it acts in, or the arms would not be comparable.
  * **No leakage.** The `proxy` set must not carry alpha in any exactly recoverable form, otherwise it is the
    oracle arm under another name.
"""

from unittest import TestCase, main

import numpy as np

from agents.Agent import Agent
from gym_local.ModelDynamics import LimitOrderModelDynamics
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.observation import FEATURE_SETS, InformationSet
from gym_local.index_names import ASSET_PRICE_INDEX, INVENTORY_INDEX
from rewards.RewardFunctions import CjMmCriterion
from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

TERMINAL_TIME, N_STEPS = 1.0, 200
STEP_SIZE = TERMINAL_TIME / N_STEPS
ARRIVAL_RATE, FILL_EXPONENT, VOLATILITY = 140.0, 1.5, 2.0
ALPHA_SPEED, ALPHA_VOLATILITY, ALPHA_INDEX = 0.05, 20.0, 4
MAX_INVENTORY = 20


class ConstantDepthAgent(Agent):
    def __init__(self, num_trajectories=1, depth=1.0):
        self.action = np.array([[depth, depth]]).repeat(num_trajectories, axis=0)

    def get_action(self, state):
        return self.action


def get_env(num_trajectories, seed, sensitivity=0.2):
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=VOLATILITY,
        ou_process=OuMidpriceModel(
            mean_reversion_level=0.0, mean_reversion_speed=ALPHA_SPEED, volatility=ALPHA_VOLATILITY,
            initial_price=0.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE,
            num_trajectories=num_trajectories),
        initial_price=100.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE,
        num_trajectories=num_trajectories)
    model_dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([ARRIVAL_RATE, ARRIVAL_RATE]), sensitivity, ALPHA_INDEX, STEP_SIZE, num_trajectories),
        fill_probability_model=ExponentialFillFunction(FILL_EXPONENT, STEP_SIZE, num_trajectories),
        num_trajectories=num_trajectories)
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS, model_dynamics=model_dynamics,
        reward_function=CjMmCriterion(per_step_inventory_aversion=0.01, terminal_inventory_aversion=1.0,
                                      terminal_time=TERMINAL_TIME),
        max_inventory=MAX_INVENTORY, num_trajectories=num_trajectories,
        normalise_action_space=False, normalise_observation_space=False)
    env.seed(seed)
    return env


def collect(features, seed=3, n=200, flow_window=5, return_window=20, steps=None):
    env = InformationSet(get_env(n, seed), features=features, flow_window=flow_window,
                         return_window=return_window, alpha_index=ALPHA_INDEX)
    agent = ConstantDepthAgent(n)
    observation = env.reset()
    observations, alphas = [observation], [env.raw_state[:, ALPHA_INDEX].copy()]
    for _ in range(steps or env.n_steps):
        observation, _, done, _ = env.step(agent.get_action(env.raw_state))
        observations.append(observation)
        alphas.append(env.raw_state[:, ALPHA_INDEX].copy())
        if done[0]:
            break
    return env, np.stack(observations), np.stack(alphas)


class testShapeAndSpace(TestCase):
    def test_every_set_matches_its_declared_width(self):
        for features, names in FEATURE_SETS.items():
            env, observations, _ = collect(features, steps=30)
            self.assertEqual(env.observation_space.shape, (len(names),))
            self.assertEqual(observations.shape[2], len(names))

    def test_features_stay_inside_the_declared_box(self):
        for features in FEATURE_SETS:
            _, observations, _ = collect(features, steps=60)
            self.assertTrue(np.isfinite(observations).all())
            self.assertLessEqual(np.abs(observations).max(), 5.0 + 1e-6)

    def test_the_sets_are_nested(self):
        for smaller, larger in (("minimal", "oracle"), ("proxy", "oracle_proxy")):
            self.assertEqual(FEATURE_SETS[smaller][:2], FEATURE_SETS[larger][:2])
            self.assertTrue(set(FEATURE_SETS[smaller]).issubset(FEATURE_SETS[larger]))


class testTheEnvironmentIsUntouched(TestCase):
    def test_the_wrapper_does_not_change_the_world(self):
        """What the agent sees is a modelling decision; it must not alter the dynamics it acts in. The same seed
        and the same actions must produce the same inventory and price path under every information set."""
        reference = None
        for features in FEATURE_SETS:
            env, _, _ = collect(features)
            path = np.stack([env.raw_state[:, INVENTORY_INDEX], env.raw_state[:, ASSET_PRICE_INDEX]])
            if reference is None:
                reference = path
            else:
                np.testing.assert_array_equal(path, reference)


class testTheProxiesTrackTheLatentFactor(TestCase):
    def test_flow_imbalance_is_correlated_with_alpha(self):
        """The arrival intensities are tilted by alpha, so the realised order flow imbalance over a window is a
        noisy measurement of it. If this correlation were absent the `proxy` arm would have nothing to infer."""
        _, observations, alphas = collect("proxy")
        names = FEATURE_SETS["proxy"]
        flow = observations[:, :, names.index("flow_imbalance")].ravel()
        alpha = alphas.ravel()
        self.assertGreater(np.corrcoef(flow, alpha)[0, 1], 0.10)

    def test_recent_return_is_correlated_with_alpha(self):
        _, observations, alphas = collect("proxy")
        names = FEATURE_SETS["proxy"]
        returns = observations[:, :, names.index("recent_return")].ravel()
        self.assertGreater(np.corrcoef(returns, alphas.ravel())[0, 1], 0.10)

    def test_the_proxies_are_noisy_rather_than_exact(self):
        """The arm is only interesting if alpha has to be *inferred*. A proxy that recovered it exactly would make
        `proxy` the oracle arm under another name."""
        _, observations, alphas = collect("proxy")
        names = FEATURE_SETS["proxy"]
        for name in ("flow_imbalance", "recent_return", "fill_imbalance"):
            correlation = np.corrcoef(observations[:, :, names.index(name)].ravel(), alphas.ravel())[0, 1]
            self.assertLess(abs(correlation), 0.95, f"{name} recovers alpha almost exactly")

    def test_alpha_is_absent_from_the_observable_sets(self):
        for features in ("minimal", "proxy"):
            self.assertNotIn("alpha", FEATURE_SETS[features])

    def test_the_oracle_feature_is_alpha_up_to_a_fixed_scale(self):
        env, observations, alphas = collect("oracle", steps=40)
        index = FEATURE_SETS["oracle"].index("alpha")
        unclipped = np.abs(alphas) < 4.9 * env._alpha_scale
        np.testing.assert_allclose(
            observations[:, :, index][unclipped], (alphas / env._alpha_scale)[unclipped], rtol=1e-5)


class testTheRollingFeatures(TestCase):
    def test_the_flow_window_has_an_interior_optimum(self):
        """Two effects fight: a longer window averages away arrival noise, and a longer window averages in stale
        values of a factor whose half-life is about 13.5 steps here. Neither extreme wins, which is what makes
        the window a real design parameter rather than "as long as possible"."""
        index = FEATURE_SETS["proxy"].index("flow_imbalance")
        correlations = {}
        for window in (1, 5, 80):
            _, observations, alphas = collect("proxy", flow_window=window)
            correlations[window] = np.corrcoef(observations[:, :, index].ravel(), alphas.ravel())[0, 1]
        self.assertGreater(correlations[5], correlations[1], "too short: arrival noise dominates")
        self.assertGreater(correlations[5], correlations[80], "too long: the factor has moved")

    def test_the_return_window_wants_to_be_longer_than_the_flow_window(self):
        """The return integrates the drift against a diffusion growing only like sqrt(W), so its optimum sits
        further out than the count-based proxy's. This is why the two windows are separate parameters."""
        index = FEATURE_SETS["proxy"].index("recent_return")
        correlations = {}
        for window in (5, 20):
            _, observations, alphas = collect("proxy", return_window=window)
            correlations[window] = np.corrcoef(observations[:, :, index].ravel(), alphas.ravel())[0, 1]
        self.assertGreater(correlations[20], correlations[5])

    def test_the_buffers_are_cleared_between_episodes(self):
        env = InformationSet(get_env(50, 3), features="proxy", alpha_index=ALPHA_INDEX)
        agent = ConstantDepthAgent(50)
        env.reset()
        for _ in range(40):
            env.step(agent.get_action(env.raw_state))
        first = env.reset()
        for _ in range(40):
            env.step(agent.get_action(env.raw_state))
        second = env.reset()
        np.testing.assert_array_equal(first, second)
        names = FEATURE_SETS["proxy"]
        for name in ("flow_imbalance", "fill_imbalance", "recent_return"):
            np.testing.assert_array_equal(second[:, names.index(name)], np.zeros(50))

    def test_the_rolling_sum_matches_a_naive_recomputation(self):
        from gym_local.observation import _RollingSum

        rng = np.random.default_rng(0)
        window, n = 7, 4
        rolling, pushed = _RollingSum(n, window), []
        for _ in range(30):
            values = rng.normal(size=n)
            pushed.append(values)
            total = rolling.push(values).copy()
            np.testing.assert_allclose(total, np.sum(pushed[-window:], axis=0), atol=1e-12)


if __name__ == "__main__":
    main()
