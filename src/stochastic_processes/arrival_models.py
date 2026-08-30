import abc
from typing import Optional

import numpy as np

from stochastic_processes.StochasticProcessModel import StochasticProcessModel


class ArrivalModel(StochasticProcessModel):
    """ArrivalModel models the arrival of orders to the order book. The first entry of arrivals represents an arrival
    of an exogenous SELL order (arriving on the buy side of the book) and the second entry represents an arrival of an
    exogenous BUY order (arriving on the sell side of the book).
    """

    def __init__(
        self,
        min_value: np.ndarray,
        max_value: np.ndarray,
        step_size: float,
        terminal_time: float,
        initial_state: np.ndarray,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        super().__init__(min_value, max_value, step_size, terminal_time, initial_state, num_trajectories, seed)

    @abc.abstractmethod
    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        """`state` is the full environment state *before* the current step, laid out as
        [cash, inventory, time, <midprice model state>, <arrival model state>, ...]. Models with constant
        intensities ignore it; state dependent models read their driving signal from it. It defaults to None so
        that arrival models remain usable standalone, outside a TradingEnvironment."""
        pass


class PoissonArrivalModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.intensity * self.step_size


class PoissonArrivalNonLinearModel(ArrivalModel):
    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < 1. - np.exp(-self.intensity * self.step_size)


class StateDependentPoissonArrivalModel(ArrivalModel):
    """Poisson arrivals whose intensities are tilted by an exogenous signal read from the environment state --
    typically the mean reverting alpha carried by `ShortTermOuAlphaMidpriceModel`, which also drives the midprice
    drift. Writing alpha for that signal,

        lambda_sell(alpha) = intensity[0] * exp(-sensitivity * alpha)
        lambda_buy(alpha)  = intensity[1] * exp(+sensitivity * alpha)

    Recall the `ArrivalModel` convention: entry 0 is an exogenous SELL market order (it hits the market maker's bid,
    so the market maker BUYS) and entry 1 is an exogenous BUY market order (it lifts the ask, so the market maker
    SELLS). With `sensitivity > 0`, a positive alpha therefore makes the market maker sell precisely when the
    midprice is drifting up. Adverse selection is not hard coded as a correlation between fills and price moves: it
    emerges from the two channels sharing one latent factor. Because that factor is part of the observation, the
    adverse selection is in principle predictable, which is what leaves an agent something to learn.

    `sensitivity = 0` reduces to `PoissonArrivalModel` exactly -- same intensities and, for a given seed, the same
    random draws. That nesting is what makes the closed-form benchmark a limiting case of this environment.

    `signal_index` is the position of the signal in the environment state. The state is laid out as
    [cash, inventory, time, <midprice model state>, <arrival model state>, ...], so with
    `ShortTermOuAlphaMidpriceModel` (state = [price, alpha]) as the midprice model, alpha sits at index 4. Read it
    off `env.stochastic_process_indices` rather than trusting the default when the model stack changes.
    """

    def __init__(
        self,
        intensity: np.ndarray = np.array([140.0, 140.0]),
        sensitivity: float = 0.0,
        signal_index: int = 4,
        step_size: float = 0.001,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.intensity = np.array(intensity)
        self.sensitivity = sensitivity
        self.signal_index = signal_index
        self.tilt = np.array([[-1.0, 1.0]])  # sell side is damped, buy side is amplified by a positive signal
        super().__init__(
            min_value=np.array([[]]),
            max_value=np.array([[]]),
            step_size=step_size,
            terminal_time=0.0,
            initial_state=np.array([[]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None):
        pass

    def get_signal(self, state: np.ndarray = None) -> np.ndarray:
        """Returns a (num_trajectories, 1) column of signal values. Falls back to zero -- i.e. to constant
        intensities -- when there is no state to read, and short circuits when the model is switched off so that
        `sensitivity = 0` is bit-for-bit identical to `PoissonArrivalModel`."""
        if state is None or self.sensitivity == 0.0:
            return np.zeros((self.num_trajectories, 1))
        return state[:, self.signal_index].reshape(-1, 1)

    def get_intensities(self, state: np.ndarray = None) -> np.ndarray:
        return self.intensity * np.exp(self.sensitivity * self.get_signal(state) * self.tilt)

    def get_arrival_probabilities(self, state: np.ndarray = None) -> np.ndarray:
        # A large tilt can push intensity * step_size above one; clamping keeps this a valid Bernoulli probability.
        # The clamp is inactive whenever intensity * step_size <= 1, so it never perturbs the sensitivity = 0 case.
        return np.minimum(self.get_intensities(state) * self.step_size, 1.0)

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.get_arrival_probabilities(state)


class HawkesArrivalModel(ArrivalModel):
    def __init__(
        self,
        baseline_arrival_rate: np.ndarray = np.array([[10.0, 10.0]]),
        step_size: float = 0.01,
        jump_size: float = 40.0,
        mean_reversion_speed: float = 60.0,
        terminal_time: float = 1,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.baseline_arrival_rate = baseline_arrival_rate
        self.jump_size = jump_size  # see https://arxiv.org/pdf/1507.02822.pdf, equation (4).
        self.mean_reversion_speed = mean_reversion_speed
        super().__init__(
            min_value=np.array([[0, 0]]),
            max_value=np.array([[1, 1]]) * self._get_max_arrival_rate(),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=baseline_arrival_rate,
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.mean_reversion_speed
            * (np.ones((self.num_trajectories, 2)) * self.baseline_arrival_rate - self.current_state)
            * self.step_size
            * np.ones((self.num_trajectories, 2))
            + self.jump_size * arrivals
        )
        return self.current_state

    def get_arrivals(self, state: np.ndarray = None) -> np.ndarray:
        unif = self.rng.uniform(size=(self.num_trajectories, 2))
        return unif < self.current_state * self.step_size

    def _get_max_arrival_rate(self):
        return self.baseline_arrival_rate * 10

    # TODO: Improve this with 4*std
    # See: https://math.stackexchange.com/questions/4047342/expectation-of-hawkes-process-with-exponential-kernel
