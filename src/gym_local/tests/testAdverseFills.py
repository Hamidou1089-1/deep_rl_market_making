"""Tests for the Lalor and Swishchuk (2024) fill mechanism.

The paper's claim is that the standard market-making simulator overstates performance because it draws market
orders independently of the price path: the midprice can walk through a resting quote without filling it, so every
fill is non-adverse by construction. `AdverseFillModelDynamics` adds the two corrections of its section 4 -- fills
forced by the price trading through the quote (eq. 16-17), and a queue-position probability on the remaining fills
(eq. 13-15) -- combined as N = max(AF, NF) (eq. 18-19).

Three properties are asserted here.

  * **Nesting** -- with the adverse branch off and `queue_probability = 1` the class must reproduce
    `LimitOrderModelDynamics` path by path, so the closed-form benchmark stays an exact limiting case.
  * **The rule** -- an adverse fill must be a deterministic function of the realised price move and the quoted
    depth, must fire on the side the price ran into, and must be avoidable by quoting past the move.
  * **Settlement** -- an execution must clear at the price the order was quoted from, not at the price the market
    has just moved to; otherwise the adverse fill would be free.
"""

from unittest import TestCase, main

import numpy as np

from agents.Agent import Agent
from gym_local.ModelDynamics import AdverseFillModelDynamics, LimitOrderModelDynamics
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.helpers.generate_trajectory import generate_trajectory
from gym_local.index_names import BID_INDEX, ASK_INDEX, CASH_INDEX, INVENTORY_INDEX
from rewards.RewardFunctions import PnL
from stochastic_processes.arrival_models import PoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import BrownianMotionMidpriceModel

TERMINAL_TIME = 1.0
N_STEPS = 100
STEP_SIZE = TERMINAL_TIME / N_STEPS
ARRIVAL_RATE = 140.0
FILL_EXPONENT = 1.5
VOLATILITY = 2.0
INITIAL_PRICE = 100.0

# The whole mechanism is governed by one dimensionless number: the quoted depth measured in standard deviations of
# the price move over a step,
#
#     rho = delta / (sigma * sqrt(dt)),      P(adverse fill per side per step) = 1 - Phi(rho).
#
# The paper's own parameters (Table 3) sit at rho = 1.00, which predicts an adverse-to-non-adverse fill ratio of
# 5.4 against the 4.8 of its Table 4 -- the mechanism read off first principles reproduces the paper's fill mix.
# Tests that need the adverse branch to fire therefore quote at rho ~ 1, and
# `testTheRegimeWhereTheMechanismBites` asserts that rho is the operative quantity.
PER_STEP_PRICE_MOVE = VOLATILITY * np.sqrt(STEP_SIZE)
ACTIVE_DEPTH = PER_STEP_PRICE_MOVE  # rho = 1


class ConstantDepthAgent(Agent):
    """Quotes a fixed depth on both sides, so that any difference between two runs comes from the environment."""

    def __init__(self, num_trajectories: int = 1, depth: float = ACTIVE_DEPTH):
        self.action = np.array([[depth, depth]]).repeat(num_trajectories, axis=0)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return self.action


def get_dynamics(num_trajectories, adverse=True, queue_probability=1.0, dynamics_class=None):
    dynamics_class = dynamics_class or AdverseFillModelDynamics
    kwargs = dict(
        midprice_model=BrownianMotionMidpriceModel(
            volatility=VOLATILITY, initial_price=INITIAL_PRICE, terminal_time=TERMINAL_TIME,
            step_size=STEP_SIZE, num_trajectories=num_trajectories),
        arrival_model=PoissonArrivalModel(
            np.array([ARRIVAL_RATE, ARRIVAL_RATE]), STEP_SIZE, num_trajectories),
        fill_probability_model=ExponentialFillFunction(FILL_EXPONENT, STEP_SIZE, num_trajectories),
        num_trajectories=num_trajectories,
    )
    if dynamics_class is AdverseFillModelDynamics:
        kwargs.update(track_adverse_fills=adverse, queue_probability=queue_probability)
    return dynamics_class(**kwargs)


def get_env(num_trajectories, seed, **kwargs):
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS,
        model_dynamics=get_dynamics(num_trajectories, **kwargs),
        reward_function=PnL(), max_inventory=100, num_trajectories=num_trajectories,
        normalise_action_space=False, normalise_observation_space=False)
    env.seed(seed)
    return env


class testNesting(TestCase):
    def test_the_benchmark_environment_is_reproduced_path_by_path(self):
        """No adverse fills and a front-of-queue probability is exactly the simulator of Cartea et al. (2015).
        The class must then draw the same random numbers in the same order, not merely match in distribution."""
        for seed in (1, 55, 999, 12345):
            improved = generate_trajectory(
                get_env(20, seed, adverse=False, queue_probability=1.0), ConstantDepthAgent(20), seed=seed)
            benchmark = generate_trajectory(
                get_env(20, seed, dynamics_class=LimitOrderModelDynamics), ConstantDepthAgent(20), seed=seed)
            for improved_array, benchmark_array in zip(improved, benchmark):
                np.testing.assert_array_equal(improved_array, benchmark_array)

    def test_the_adverse_branch_changes_the_paths(self):
        """A guard against the previous test passing because the switch does nothing at all."""
        seed = 7
        with_adverse = generate_trajectory(get_env(20, seed, adverse=True), ConstantDepthAgent(20), seed=seed)
        without = generate_trajectory(get_env(20, seed, adverse=False), ConstantDepthAgent(20), seed=seed)
        self.assertFalse(np.array_equal(with_adverse[0], without[0]))


def resolve(depths, price_move, arrivals=None, fills=None, **kwargs):
    """Drive `resolve_fills` with the price move imposed rather than simulated, so assertions on the rule are
    exact rather than statistical."""
    n = len(depths)
    dynamics = get_dynamics(n, **kwargs)
    midprice_before = np.full((n, 1), INITIAL_PRICE)
    dynamics.midprice_model.current_state[:, 0] = INITIAL_PRICE + np.asarray(price_move)
    arrivals = np.zeros((n, 2)) if arrivals is None else np.asarray(arrivals, dtype=float)
    fills = np.zeros((n, 2)) if fills is None else np.asarray(fills, dtype=float)
    _, executed = dynamics.resolve_fills(arrivals, fills, np.asarray(depths, dtype=float), midprice_before)
    return executed, dynamics


class testTheAdverseFillRule(TestCase):
    """The rule is tested directly on imposed price moves."""

    def test_a_price_move_through_the_quote_fills_that_side_and_only_that_side(self):
        executed, _ = resolve(depths=[[1.0, 1.0], [1.0, 1.0]], price_move=[+2.0, -2.0])
        self.assertEqual(executed[0, ASK_INDEX], 1.0, "an upward move through the ask must fill the ask")
        self.assertEqual(executed[0, BID_INDEX], 0.0, "it must not fill the bid")
        self.assertEqual(executed[1, BID_INDEX], 1.0, "a downward move through the bid must fill the bid")
        self.assertEqual(executed[1, ASK_INDEX], 0.0)

    def test_quoting_past_the_move_avoids_the_fill(self):
        """The depth is the protection. This is what gives the agent something to trade off, rather than a fixed
        tax it can do nothing about."""
        executed, _ = resolve(depths=[[0.5, 0.5], [5.0, 5.0]], price_move=[+2.0, +2.0])
        self.assertEqual(executed[0, ASK_INDEX], 1.0)
        self.assertEqual(executed[1, ASK_INDEX], 0.0)

    def test_at_the_touch_any_move_fills_one_side(self):
        """Depth zero is the paper's own at-the-touch rule, eq. (16)-(17)."""
        executed, _ = resolve(depths=[[0.0, 0.0]] * 3, price_move=[+1e-9, -1e-9, 0.0])
        np.testing.assert_array_equal(executed[0], [0.0, 1.0])
        np.testing.assert_array_equal(executed[1], [1.0, 0.0])
        np.testing.assert_array_equal(executed[2], [0.0, 0.0], "an unchanged price fills neither side")

    def test_the_adverse_branch_is_not_a_draw(self):
        """Twice the same inputs must give the same executions -- the branch is a function of the path."""
        first, _ = resolve(depths=[[1.0, 1.0]] * 4, price_move=[+2.0, -2.0, +0.5, -0.5])
        second, _ = resolve(depths=[[1.0, 1.0]] * 4, price_move=[+2.0, -2.0, +0.5, -0.5])
        np.testing.assert_array_equal(first, second)

    def test_switching_the_branch_off_leaves_only_the_arrival_driven_fills(self):
        executed, _ = resolve(depths=[[1.0, 1.0]], price_move=[+9.0], adverse=False)
        np.testing.assert_array_equal(executed, [[0.0, 0.0]])

    def test_the_two_branches_combine_as_a_maximum(self):
        """Equation (18)-(19): a fill occurs if either branch fires, and a doubly triggered side still fills once."""
        executed, dynamics = resolve(
            depths=[[1.0, 1.0]], price_move=[+2.0], arrivals=[[1.0, 1.0]], fills=[[1.0, 1.0]])
        np.testing.assert_array_equal(executed, [[1.0, 1.0]])
        np.testing.assert_array_equal(
            executed, np.maximum(dynamics.last_adverse_fills, dynamics.last_non_adverse_fills))
        np.testing.assert_array_equal(dynamics.last_adverse_fills, [[0.0, 1.0]])

    def test_a_zero_queue_probability_removes_every_non_adverse_fill(self):
        executed, dynamics = resolve(
            depths=[[1.0, 1.0]] * 50, price_move=[0.0] * 50,
            arrivals=np.ones((50, 2)), fills=np.ones((50, 2)), queue_probability=0.0)
        np.testing.assert_array_equal(executed, np.zeros((50, 2)))

    def test_a_queue_probability_below_one_thins_the_non_adverse_fills(self):
        executed, _ = resolve(
            depths=[[1.0, 1.0]] * 4000, price_move=[0.0] * 4000,
            arrivals=np.ones((4000, 2)), fills=np.ones((4000, 2)), queue_probability=0.2)
        self.assertAlmostEqual(executed.mean(), 0.2, delta=0.02)


class testSettlement(TestCase):
    def test_an_execution_clears_at_the_price_it_was_quoted_from(self):
        """The order was resting at S_t + delta. Settling it at S_{t+dt} instead would hand the agent the price
        move it was just run over by, which is exactly the error the paper is correcting."""
        depth = ACTIVE_DEPTH
        env = get_env(1, seed=3, adverse=True, queue_probability=0.0)
        obs = env.reset()
        midprice_before = float(env.model_dynamics.midprice_model.current_state[0, 0])
        cash_before = float(env.state[0, CASH_INDEX])
        # Step until the adverse branch alone produces a fill, which it must settle at midprice_before + depth.
        for _ in range(N_STEPS):
            obs, _, done, _ = env.step(np.array([[depth, depth]]))
            inventory = float(env.state[0, INVENTORY_INDEX])
            cash = float(env.state[0, CASH_INDEX])
            if inventory != 0:
                expected = cash_before + (-inventory) * (midprice_before + depth * np.sign(-inventory))
                self.assertAlmostEqual(cash, expected, places=9)
                return
            midprice_before = float(env.model_dynamics.midprice_model.current_state[0, 0])
            cash_before = cash
            if done[0]:
                break
        self.fail("no adverse fill occurred in a whole episode; the mechanism is not firing")

    def test_the_settlement_price_override_does_not_leak_out_of_the_step(self):
        env = get_env(4, seed=11)
        env.reset()
        env.step(np.array([[ACTIVE_DEPTH, ACTIVE_DEPTH]]).repeat(4, axis=0))
        self.assertIsNone(env.model_dynamics.execution_midprice)
        np.testing.assert_allclose(
            env.model_dynamics.midprice.ravel(), env.model_dynamics.midprice_model.current_state[:, 0])


class testTheEffectOnTheAgent(TestCase):
    def test_adverse_fills_make_the_market_maker_lose_money_on_its_inventory(self):
        """The signature of adverse selection: the price moves against the position just taken. Measured as the
        one-step markout E[dq_t * (S_{t+1} - S_t)], which must be negative with the branch on and indistinguishable
        from zero with it off."""
        from gym_local.index_names import ASSET_PRICE_INDEX

        def markout(adverse):
            values = []
            for seed in (2, 4, 6, 8, 10):
                env = get_env(400, seed, adverse=adverse)
                observations, _, _ = generate_trajectory(env, ConstantDepthAgent(400), seed=seed)
                inventory_change = np.diff(observations[:, INVENTORY_INDEX, :], axis=1)
                price_change = np.diff(observations[:, ASSET_PRICE_INDEX, :], axis=1)
                values.append(float(np.mean(inventory_change * price_change)))
            return np.array(values)

        with_adverse, without = markout(True), markout(False)
        self.assertLess(with_adverse.mean(), 0.0)
        self.assertLess(with_adverse.max(), without.min(),
                        "the adverse branch must separate the two regimes on every seed")
        self.assertLess(abs(without.mean()), abs(with_adverse.mean()) / 10,
                        "with the branch off the markout must be an order of magnitude closer to zero")


class testTheRegimeWhereTheMechanismBites(TestCase):
    """The correction is not a fixed tax: whether it does anything at all is decided by rho = delta / (sigma sqrt dt).

    This matters for the project rather than for the class. The Cartea-Jaimungal closed form quotes around 1/kappa,
    and with the fill exponent and volatility used elsewhere in this repository that is rho = 4.7, where an adverse
    fill is a five sigma event and the paper's correction is numerically invisible. Reproducing the paper's regime
    is therefore a calibration constraint linking kappa to sigma sqrt(dt), not a switch to turn on.
    """

    def adverse_fill_rate(self, depth, num_trajectories=2000):
        dynamics = get_dynamics(num_trajectories, queue_probability=0.0)
        midprice_before = np.full((num_trajectories, 1), INITIAL_PRICE)
        rng = np.random.default_rng(0)
        dynamics.midprice_model.current_state[:, 0] = INITIAL_PRICE + rng.normal(
            scale=PER_STEP_PRICE_MOVE, size=num_trajectories)
        depths = np.full((num_trajectories, 2), depth)
        _, executed = dynamics.resolve_fills(
            np.zeros((num_trajectories, 2)), np.zeros((num_trajectories, 2)), depths, midprice_before)
        return executed.mean()

    def test_the_rate_follows_the_gaussian_tail_of_rho(self):
        from scipy.stats import norm

        for rho in (0.5, 1.0, 2.0):
            expected = 1 - norm.cdf(rho)  # one side fires; averaged over the two sides this is the mean
            self.assertAlmostEqual(self.adverse_fill_rate(rho * PER_STEP_PRICE_MOVE), expected, delta=0.02)

    def test_at_the_depth_the_closed_form_quotes_the_mechanism_is_inactive(self):
        """The closed form quotes around 1/kappa. Here that is rho = 3.3, and at the dt = 1/200 used by the
        notebooks it is rho = 4.7 -- a four to five sigma move, so the correction is numerically invisible.
        Documented, not endorsed: it is why the paper's environment cannot simply be switched on over this
        repository's parameters, and why reproducing its regime is a calibration constraint on kappa sigma sqrt(dt)."""
        closed_form_depth = 1 / FILL_EXPONENT
        self.assertGreater(closed_form_depth / PER_STEP_PRICE_MOVE, 3.0)
        adverse = self.adverse_fill_rate(closed_form_depth, num_trajectories=40000)
        non_adverse = ARRIVAL_RATE * STEP_SIZE * np.exp(-FILL_EXPONENT * closed_form_depth)
        self.assertLess(adverse, 1e-3)
        self.assertLess(adverse / non_adverse, 0.01,
                        "at the depth the closed form quotes, adverse fills are two orders of magnitude rarer "
                        "than the arrival driven fills, so the correction cannot move any result")


class testDepthsAreFlooredAtZero(TestCase):
    """A limit order cannot be posted through the mid, and the two closed forms do not know that.

    At large inventory Cartea-Jaimungal and Avellaneda-Stoikov both return a negative depth on the side that
    reduces the position. The simulator has no representation for such an order: `ExponentialFillFunction`
    returns exp(-kappa delta) > 1, and "the price moved past the quote" becomes almost always true. Measured on
    the closed form at kappa = 1.5, negative depths were 0.59 % of quotes but carried 53 % of all adverse fills.
    """

    def test_a_negative_quote_is_treated_as_a_quote_at_the_touch(self):
        dynamics = get_dynamics(3)
        floored = dynamics._limit_depths(np.array([[-1.0, -0.5], [0.0, 0.0], [0.2, 0.3]]))
        np.testing.assert_array_equal(floored, [[0.0, 0.0], [0.0, 0.0], [0.2, 0.3]])

    def test_the_adverse_rule_stays_directional_at_a_negative_quote(self):
        """Without the floor, `price_move > -1.0` holds for most downward moves too, so the *ask* would be
        adversely filled by the price going down. The floor keeps each side answering to its own direction."""
        executed, _ = resolve(depths=[[-1.0, -1.0]], price_move=[-0.5])
        np.testing.assert_array_equal(executed, [[1.0, 0.0]], "only the bid, which the price ran through")
        executed, _ = resolve(depths=[[-1.0, -1.0]], price_move=[+0.5])
        np.testing.assert_array_equal(executed, [[0.0, 1.0]])

    def test_the_floor_leaves_ordinary_quotes_untouched(self):
        executed, _ = resolve(depths=[[1.0, 1.0], [1.0, 1.0]], price_move=[+2.0, -2.0])
        np.testing.assert_array_equal(executed, [[0.0, 1.0], [1.0, 0.0]])


if __name__ == "__main__":
    main()
