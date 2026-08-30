"""Tests for state dependent arrivals.

The point of `StateDependentPoissonArrivalModel` is to make adverse selection *emerge* from a latent factor that
drives both the arrival intensities and the midprice drift, rather than hard-coding a correlation between fills and
subsequent price moves. Two properties matter and are asserted here:

  * nesting -- `sensitivity = 0` must reproduce `PoissonArrivalModel` exactly, so that the closed-form benchmark
    remains a limiting case of the richer environment;
  * the mechanism -- with `sensitivity > 0` the market maker must actually be adversely selected, which is measured
    by the markout E[dq_t * (S_{t+h} - S_t)] being significantly negative.
"""

from unittest import TestCase, main

import numpy as np

from agents.Agent import Agent
from gym_local.ModelDynamics import LimitOrderModelDynamics
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.helpers.generate_trajectory import generate_trajectory
from gym_local.index_names import INVENTORY_INDEX, ASSET_PRICE_INDEX
from stochastic_processes.arrival_models import PoissonArrivalModel, StateDependentPoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

TERMINAL_TIME = 1.0
N_STEPS = 200
STEP_SIZE = TERMINAL_TIME / N_STEPS
ARRIVAL_RATE = 140.0
FILL_EXPONENT = 1.5
INTENSITY = np.array([ARRIVAL_RATE, ARRIVAL_RATE])

# With `ShortTermOuAlphaMidpriceModel` as the (first) stochastic process, the environment state is laid out as
# [cash, inventory, time, price, alpha], so the alpha signal sits at index 4.
ALPHA_INDEX = 4

# `OuMidpriceModel.update` applies its mean reversion without multiplying by the step size, so the default speed of
# 1.0 collapses the alpha to white noise. A small speed is what actually gives a persistent, predictable signal --
# see testMidpriceModels.testOuMidpriceModelDiscretisation.
ALPHA_SPEED = 0.05
ALPHA_VOLATILITY = 20.0


class ConstantDepthAgent(Agent):
    def __init__(self, num_trajectories: int, depth: float = 1 / FILL_EXPONENT):
        self.action = np.ones((num_trajectories, 2)) * depth

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action


def get_env(sensitivity: float, num_trajectories: int, seed: int = 101, state_dependent: bool = True):
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=2.0,
        ou_process=OuMidpriceModel(
            mean_reversion_level=0.0,
            mean_reversion_speed=ALPHA_SPEED,
            volatility=ALPHA_VOLATILITY,
            initial_price=0.0,
            terminal_time=TERMINAL_TIME,
            step_size=STEP_SIZE,
            num_trajectories=num_trajectories,
        ),
        initial_price=100.0,
        terminal_time=TERMINAL_TIME,
        step_size=STEP_SIZE,
        num_trajectories=num_trajectories,
    )
    if state_dependent:
        arrival_model = StateDependentPoissonArrivalModel(
            intensity=INTENSITY,
            sensitivity=sensitivity,
            signal_index=ALPHA_INDEX,
            step_size=STEP_SIZE,
            num_trajectories=num_trajectories,
        )
    else:
        arrival_model = PoissonArrivalModel(
            intensity=INTENSITY, step_size=STEP_SIZE, num_trajectories=num_trajectories
        )
    model_dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=arrival_model,
        fill_probability_model=ExponentialFillFunction(
            fill_exponent=FILL_EXPONENT, step_size=STEP_SIZE, num_trajectories=num_trajectories
        ),
        num_trajectories=num_trajectories,
    )
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME,
        n_steps=N_STEPS,
        model_dynamics=model_dynamics,
        num_trajectories=num_trajectories,
        normalise_action_space=False,
        normalise_observation_space=False,
    )
    env.seed(seed)
    return env


def get_markouts(observations: np.ndarray, horizons) -> dict:
    """Markout of the market maker's own trades: E[dq_t * (S_{t+h} - S_t)]. Negative means the midprice moves
    against the inventory the market maker just took on, i.e. the market maker is adversely selected."""
    inventory = observations[:, INVENTORY_INDEX, :]
    price = observations[:, ASSET_PRICE_INDEX, :]
    inventory_change = np.diff(inventory, axis=1)
    markouts = dict()
    for horizon in horizons:
        n = inventory_change.shape[1] - horizon
        markouts[horizon] = float(
            np.mean(inventory_change[:, :n] * (price[:, horizon : horizon + n] - price[:, :n]))
        )
    return markouts


class testStateDependentPoissonArrivalModel(TestCase):
    def test_zero_sensitivity_reproduces_poisson_arrivals_draw_for_draw(self):
        num_trajectories = 17
        baseline = PoissonArrivalModel(
            intensity=INTENSITY, step_size=STEP_SIZE, num_trajectories=num_trajectories, seed=7
        )
        nested = StateDependentPoissonArrivalModel(
            intensity=INTENSITY, sensitivity=0.0, step_size=STEP_SIZE, num_trajectories=num_trajectories, seed=7
        )
        rng = np.random.default_rng(0)
        for _ in range(50):
            # An arbitrary, non-degenerate state: switching the model off must make it ignore the state entirely.
            state = rng.normal(size=(num_trajectories, 5)) * 5
            np.testing.assert_array_equal(nested.get_arrivals(state), baseline.get_arrivals(state))

    def test_signal_defaults_to_zero_without_a_state(self):
        model = StateDependentPoissonArrivalModel(sensitivity=0.3, num_trajectories=4, step_size=STEP_SIZE)
        np.testing.assert_array_equal(model.get_signal(None), np.zeros((4, 1)))
        np.testing.assert_allclose(model.get_intensities(None), np.tile(INTENSITY, (4, 1)))

    def test_intensities_tilt_towards_buy_orders_when_the_signal_is_positive(self):
        model = StateDependentPoissonArrivalModel(
            sensitivity=0.2, signal_index=ALPHA_INDEX, num_trajectories=3, step_size=STEP_SIZE
        )
        state = np.zeros((3, 5))
        state[:, ALPHA_INDEX] = np.array([-2.0, 0.0, 3.0])
        intensities = model.get_intensities(state)
        sell_side, buy_side = intensities[:, 0], intensities[:, 1]
        # A positive signal means the midprice is drifting up, so exogenous BUY orders (entry 1, which lift the
        # market maker's ask) must become more frequent and exogenous SELL orders less frequent.
        self.assertLess(buy_side[0], sell_side[0])
        self.assertAlmostEqual(buy_side[1], sell_side[1])
        self.assertGreater(buy_side[2], sell_side[2])
        # The tilt is antisymmetric, so the geometric mean of the two intensities is unchanged by the signal.
        np.testing.assert_allclose(sell_side * buy_side, INTENSITY[0] * INTENSITY[1] * np.ones(3))

    def test_arrival_probabilities_are_clamped_to_one(self):
        model = StateDependentPoissonArrivalModel(
            intensity=INTENSITY, sensitivity=5.0, signal_index=ALPHA_INDEX, num_trajectories=2, step_size=STEP_SIZE
        )
        state = np.zeros((2, 5))
        state[:, ALPHA_INDEX] = np.array([-10.0, 10.0])
        probabilities = model.get_arrival_probabilities(state)
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertAlmostEqual(probabilities[1, 1], 1.0)  # saturated by a large positive signal


class testStateDependentArrivalsInTheEnvironment(TestCase):
    def test_zero_sensitivity_gives_the_same_environment_as_poisson_arrivals(self):
        """The strong nesting check: switched off, the richer environment must produce a bit-for-bit identical
        trajectory, including an identical observation space."""
        num_trajectories = 20
        nested_env = get_env(sensitivity=0.0, num_trajectories=num_trajectories, seed=55, state_dependent=True)
        baseline_env = get_env(sensitivity=0.0, num_trajectories=num_trajectories, seed=55, state_dependent=False)
        np.testing.assert_array_equal(nested_env.observation_space.low, baseline_env.observation_space.low)
        np.testing.assert_array_equal(nested_env.observation_space.high, baseline_env.observation_space.high)
        nested = generate_trajectory(nested_env, ConstantDepthAgent(num_trajectories), seed=55)
        baseline = generate_trajectory(baseline_env, ConstantDepthAgent(num_trajectories), seed=55)
        for name, nested_array, baseline_array in zip(("observations", "actions", "rewards"), nested, baseline):
            np.testing.assert_array_equal(nested_array, baseline_array, err_msg=f"{name} differ")

    def test_market_maker_sells_into_a_rising_signal(self):
        """The mechanism, measured directly: conditional on a high alpha the market maker should be accumulating a
        short position, and conditional on a low alpha a long one."""
        num_trajectories = 400
        env = get_env(sensitivity=0.2, num_trajectories=num_trajectories, seed=101)
        observations, _, _ = generate_trajectory(env, ConstantDepthAgent(num_trajectories), seed=101)
        alpha = observations[:, ALPHA_INDEX, :-1]
        inventory_change = np.diff(observations[:, INVENTORY_INDEX, :], axis=1)
        top_decile = alpha > np.quantile(alpha, 0.9)
        bottom_decile = alpha < np.quantile(alpha, 0.1)
        self.assertLess(inventory_change[top_decile].mean(), -0.1)
        self.assertGreater(inventory_change[bottom_decile].mean(), 0.1)

    def test_switching_the_signal_on_creates_adverse_selection(self):
        """The markout is the statistic the environment is meant to reproduce, so it is compared between the two
        regimes rather than against an absolute threshold. Measured over 20 seeds x 500 trajectories the switched
        off markout is indistinguishable from zero at horizons 1, 5 and 20 (|t| < 0.6) but shows a small residual
        at horizon 50 (mean -0.0015, per-seed values reaching -0.006) whose t-statistic is not trustworthy given
        how skewed the estimator is at long horizons. A ratio test is insensitive to that residual: the tilted
        environment produces a markout more than an order of magnitude larger at every horizon.
        """
        num_trajectories = 400
        horizons = (1, 5, 20, 50)
        markouts = dict()
        for sensitivity in (0.0, 0.2):
            env = get_env(sensitivity=sensitivity, num_trajectories=num_trajectories, seed=101)
            observations, _, _ = generate_trajectory(env, ConstantDepthAgent(num_trajectories), seed=101)
            markouts[sensitivity] = get_markouts(observations, horizons)

        for horizon in horizons:
            switched_off, switched_on = markouts[0.0][horizon], markouts[0.2][horizon]
            self.assertLess(switched_on, -2e-3, f"markout should be negative at horizon {horizon}")
            self.assertGreater(
                abs(switched_on),
                5 * abs(switched_off),
                f"at horizon {horizon} the tilted markout ({switched_on:.5f}) is not clearly separated from the "
                f"switched off one ({switched_off:.5f})",
            )

        # Short horizons are where the null holds cleanly, so they also get an absolute bound.
        for horizon in (1, 5, 20):
            self.assertLess(abs(markouts[0.0][horizon]), 5e-3)

        # Adverse selection accumulates: the longer the market maker holds the position, the worse the markout.
        for shorter, longer in zip(horizons, horizons[1:]):
            self.assertLess(markouts[0.2][longer], markouts[0.2][shorter])


if __name__ == "__main__":
    main()
