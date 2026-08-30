from math import sqrt
from typing import Optional

import numpy as np

from stochastic_processes.StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel

from gym_local.index_names import BID_INDEX, ASK_INDEX

class ConstantMidpriceModel(MidpriceModel):
    def __init__(
        self,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price]]),
            max_value=np.array([[initial_price]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        pass


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * np.sqrt(terminal_time)


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.current_state * self.step_size
            + self.volatility
            * self.current_state
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class OuMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state += -self.mean_reversion_speed * (
            self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1))
        ) + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: What should this be?


class ShortTermOuAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_process: OuMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_process = ou_process or OuMidpriceModel(
            initial_price=0.0,
            terminal_time=terminal_time,
            step_size=step_size,
            num_trajectories=num_trajectories,
            seed=seed,
        )
        assert self.ou_process.num_trajectories == num_trajectories, (
            f"The alpha sub-process simulates {self.ou_process.num_trajectories} trajectories but the midprice model "
            f"simulates {num_trajectories}. A mismatch silently broadcasts one alpha path across all trajectories."
        )
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_process.min_value[0, 0],
                    ]
                ]
            ),
            max_value=np.array(
                [[self._get_max_asset_price(initial_price, terminal_time), self.ou_process.max_value[0, 0]]]
            ),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_process.current_state[:, 0] * self.step_size
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories,))
        )
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_process.current_state[:, 0]

    def reset(self):
        super().reset()
        self.ou_process.reset()

    def seed(self, seed: int = None):
        super().seed(seed)
        # Offset by a prime so the alpha stream cannot collide with the seed + i + 1 that TradingEnvironment.seed
        # hands to the other stochastic processes.
        self.ou_process.seed(None if seed is None else seed + 9973)

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @num_trajectories.setter
    def num_trajectories(self, num_trajectories: int):
        self._num_trajectories = num_trajectories
        if self.ou_process.num_trajectories != num_trajectories:
            self.ou_process.num_trajectories = num_trajectories
            self.ou_process.reset()

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class BrownianMotionJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class OuJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            - self.mean_reversion_speed
            * (self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1)))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))            
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class ShortTermJumpAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_jump_process: OuJumpMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_jump_process = ou_jump_process or OuJumpMidpriceModel(
            initial_price=0.0,
            terminal_time=terminal_time,
            step_size=step_size,
            num_trajectories=num_trajectories,
            seed=seed,
        )
        assert self.ou_jump_process.num_trajectories == num_trajectories, (
            f"The alpha sub-process simulates {self.ou_jump_process.num_trajectories} trajectories but the midprice model "
            f"simulates {num_trajectories}. A mismatch silently broadcasts one alpha path across all trajectories."
        )
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_jump_process.min_value[0, 0],
                    ]
                ]
            ),
            max_value=np.array(
                [[self._get_max_asset_price(initial_price, terminal_time), self.ou_jump_process.max_value[0, 0]]]
            ),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_jump_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_jump_process.current_state[:, 0] * self.step_size
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories,))
        )
        self.ou_jump_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_jump_process.current_state[:, 0]

    def reset(self):
        super().reset()
        self.ou_jump_process.reset()

    def seed(self, seed: int = None):
        super().seed(seed)
        # Offset by a prime so the alpha stream cannot collide with the seed + i + 1 that TradingEnvironment.seed
        # hands to the other stochastic processes.
        self.ou_jump_process.seed(None if seed is None else seed + 9973)

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @num_trajectories.setter
    def num_trajectories(self, num_trajectories: int):
        self._num_trajectories = num_trajectories
        if self.ou_jump_process.num_trajectories != num_trajectories:
            self.ou_jump_process.num_trajectories = num_trajectories
            self.ou_jump_process.reset()

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class HestonMidpriceModel(MidpriceModel):
    # Current/Initial State with the Heston model will consist of price AND current variance, not just price
    def __init__(
        self,
        drift: float = 0.05,
        volatility_mean_reversion_rate: float = 3,
        volatility_mean_reversion_level: float = 0.04,
        weiner_correlation: float = -0.8,
        volatility_of_volatility: float = 0.6,
        initial_price: float = 100,
        initial_variance: float = 0.2**2,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility_mean_reversion_rate = volatility_mean_reversion_rate
        self.terminal_time = terminal_time
        self.weiner_correlation = weiner_correlation
        self.volatility_mean_reversion_level = volatility_mean_reversion_level
        self.volatility_of_volatility = volatility_of_volatility
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_variance]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        weiner_means = np.array([0, 0])
        weiner_corr = np.array([[1, self.weiner_correlation], [self.weiner_correlation, 1]])
        # `self.rng`, not `np.random`: every other model in this file draws from its own seeded generator,
        # and drawing from the global one here made Heston the single model whose paths did not reproduce
        # from its seed -- which would silently break the nesting checks and every paired comparison.
        weiners = self.rng.multivariate_normal(weiner_means, cov=weiner_corr, size=self.num_trajectories)
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.drift * self.current_state[:, 0] * self.step_size
            + np.sqrt(self.current_state[:, 1] * self.step_size) * self.current_state[:, 0] * weiners[:, 0]
        )
        self.current_state[:, 1] = np.abs(
            self.current_state[:, 1]
            + self.volatility_mean_reversion_rate
            * (self.volatility_mean_reversion_level - self.current_state[:, 1])
            * self.step_size
            + self.volatility_of_volatility * np.sqrt(self.current_state[:, 1] * self.step_size) * weiners[:, 1]
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility_mean_reversion_level * terminal_time


class ConstantElasticityOfVarianceMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        gamma: float = 1,  # gamma = 1 is just gbm
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.gamma = gamma
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.current_state * self.drift * self.step_size  # *np.ones((self.num_trajectories, 1))
            + self.volatility
            * (self.current_state**self.gamma)
            * np.sqrt(self.step_size)
            # `self.rng`, not `np.random`: same defect as HestonMidpriceModel carried, same consequence.
            * self.rng.normal(size=self.num_trajectories)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time





class ShortTermAlphaStochasticVolMidpriceModel(ShortTermOuAlphaMidpriceModel):
    """The short-term alpha midprice, with the diffusion driven by a mean-reverting stochastic variance.

    A constant volatility has no regimes, so an agent cannot learn the one thing every market maker actually does:
    quote wider when the market turns. The variance follows the Heston / CIR dynamics

        dv = kappa_v (theta_v - v) dt + xi sqrt(v) dW_v,     dS = alpha dt + sqrt(v) dW_S,

    discretised with full truncation, `v <- max(., 0)`, which is the standard fix for Euler stepping a CIR
    process into negative territory. `correlation` is the leverage effect between the two Brownians: negative
    means volatility rises as the price falls.

    The state is `[price, alpha, volatility]`. The third component is the *volatility*, not the variance, because
    that is the quantity comparable to the constant this class replaces and the one a policy would reason in.

    **Nesting.** With `vol_of_vol = 0` and `initial_variance = long_run_variance = volatility^2`, the variance is
    stationary at exactly that level and no random number is drawn for it, so the price path is identical to
    `ShortTermOuAlphaMidpriceModel` trajectory by trajectory. That is asserted in the tests rather than assumed:
    it is what makes the constant-volatility environment a limiting case of this one, and therefore what keeps
    every earlier result comparable.

    `volatility` is kept as the long-run level rather than the running value, because `AvellanedaStoikovAgent`
    reads it off the model as a scalar and would otherwise silently see whatever the last step happened to be.
    """

    def __init__(
        self,
        volatility: float = 2.0,
        vol_mean_reversion_rate: float = 5.0,
        vol_of_vol: float = 0.0,
        correlation: float = 0.0,
        initial_volatility: Optional[float] = None,
        ou_process: OuMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        assert vol_of_vol >= 0.0, "vol_of_vol is a standard deviation."
        assert -1.0 <= correlation <= 1.0, "correlation must be a correlation."
        self.vol_mean_reversion_rate = vol_mean_reversion_rate
        self.vol_of_vol = vol_of_vol
        self.correlation = correlation
        self.long_run_variance = volatility ** 2
        self.initial_variance = (initial_volatility ** 2) if initial_volatility is not None else volatility ** 2
        super().__init__(
            volatility=volatility, ou_process=ou_process, initial_price=initial_price,
            terminal_time=terminal_time, step_size=step_size, num_trajectories=num_trajectories, seed=seed)
        # widen the declared state to carry the volatility path
        self.min_value = np.append(self.min_value, np.array([[0.0]]), axis=1)
        self.max_value = np.append(self.max_value, np.array([[self._max_volatility()]]), axis=1)
        self.initial_state = np.append(self.initial_state, np.array([[np.sqrt(self.initial_variance)]]), axis=1)
        self.reset()

    def _max_volatility(self) -> float:
        """A bound for the declared observation space, not a cap applied to the path."""
        return float(np.sqrt(self.long_run_variance) + 6 * self.vol_of_vol / max(self.vol_mean_reversion_rate, 1e-9)
                     + np.sqrt(self.initial_variance))

    @property
    def variance(self) -> np.ndarray:
        return self.current_state[:, 2] ** 2

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        variance = self.current_state[:, 2] ** 2
        price_shock = self.rng.normal(size=(self.num_trajectories,))
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_process.current_state[:, 0] * self.step_size
            + np.sqrt(variance) * sqrt(self.step_size) * price_shock
        )
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_process.current_state[:, 0]
        if self.vol_of_vol > 0.0:
            # Drawn only when the mechanism is live, so that vol_of_vol = 0 consumes exactly the random numbers
            # the constant-volatility model consumes and reproduces it path by path.
            independent = self.rng.normal(size=(self.num_trajectories,))
            vol_shock = self.correlation * price_shock + np.sqrt(1 - self.correlation ** 2) * independent
            variance = np.maximum(
                variance
                + self.vol_mean_reversion_rate * (self.long_run_variance - variance) * self.step_size
                + self.vol_of_vol * np.sqrt(variance) * sqrt(self.step_size) * vol_shock,
                0.0,
            )
        self.current_state[:, 2] = np.sqrt(variance)

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time
