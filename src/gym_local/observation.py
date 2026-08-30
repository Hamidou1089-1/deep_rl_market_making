"""What the agent is allowed to condition on.

The closed forms of Avellaneda-Stoikov and Cartea-Jaimungal read exactly two things: inventory and time to go.
Their optimal depths are functions of $(t, q)$ and of four constants, and there is no channel through which any
exogenous state could enter. That is the whole reason they degrade when the environment stops satisfying the
assumptions they were derived under -- not a defect of the formulas, but their scope.

A learned agent is interesting here only if it conditions on more. The question is *what* more. Handing it the
latent factor alpha would answer a different and much easier question, because alpha is not observable in a real
book; it is the thing a market maker infers. `InformationSet` therefore builds the features a market maker can
actually construct -- its own executions, the trades that print, and recent returns -- and keeps alpha available
only as a deliberately labelled oracle arm that bounds what any of the honest arms could achieve.

    minimal        (q, time left)                                   the closed forms' own information set
    proxy          + flow imbalance, fill imbalance, recent return  observable; alpha must be inferred
    oracle         minimal + alpha                                  upper bound, not a deployable agent
    oracle_proxy   proxy + alpha                                    how much the proxies were leaving on the table

A volatility regime is deliberately **not** among the features. When the variance is stochastic the agent lives
through the regime and has to approximate it from the returns and the flow it already sees; handing it either the
latent variance or a ready-made realised-volatility statistic would answer a different and much easier question.
The same logic is why alpha appears only in the arms labelled oracle.

The rolling features are running sums over a window, so the cost per step is O(1) per trajectory rather than
O(window).

**The two proxies want different windows, and this is measured rather than assumed.** The flow imbalance is a
count difference: its signal grows like `W * lambda dt * gamma alpha` and its noise like `sqrt(2 lambda dt W)`,
so a longer window helps -- until it exceeds the half-life of alpha itself and starts averaging in stale values.
With the parameters used here (half-life about 13.5 steps) the correlation with alpha peaks near **5 steps**
(0.79, against 0.69 at one step and 0.39 at eighty). The recent return integrates the drift against a diffusion
whose standard deviation grows only like `sqrt(W)`, so it keeps improving for longer and peaks near **20 steps**
(0.38). The defaults below are those two optima; they are separate parameters because a single window is
necessarily wrong for one of the features.
"""

from typing import Optional, Sequence

import gymnasium as gym
import numpy as np

from gym_local.index_names import ASK_INDEX, BID_INDEX, ASSET_PRICE_INDEX, INVENTORY_INDEX, TIME_INDEX
from gym_local.wrappers import ForwardingWrapper

_PROXY = ("inventory", "time_left", "flow_imbalance", "flow_intensity", "fill_imbalance", "recent_return")

FEATURE_SETS = {
    "minimal": ("inventory", "time_left"),
    "proxy": _PROXY,
    "oracle": ("inventory", "time_left", "alpha"),
    "oracle_proxy": _PROXY + ("alpha",),
}

# Standardised features are clipped into this band. It is wide enough that clipping is rare and narrow enough
# that a single outlying step cannot dominate a policy gradient.
FEATURE_BOUND = 5.0


class _RollingSum:
    """Running sum over the last `window` steps, per trajectory, in O(1) per step."""

    def __init__(self, num_trajectories: int, window: int):
        self.buffer = np.zeros((num_trajectories, window))
        self.total = np.zeros(num_trajectories)
        self.position = 0

    def reset(self):
        self.buffer[:] = 0.0
        self.total[:] = 0.0
        self.position = 0

    def push(self, values: np.ndarray) -> np.ndarray:
        self.total += values - self.buffer[:, self.position]
        self.buffer[:, self.position] = values
        self.position = (self.position + 1) % self.buffer.shape[1]
        return self.total


class InformationSet(ForwardingWrapper):
    """Replaces the raw environment state by a named, scaled feature vector.

    The wrapped environment must be built with `normalise_observation_space=False`: this wrapper reads the raw
    state and does its own scaling, and scaling twice would silently distort every feature.

    Scaling constants are read off the environment's own parameters -- the base intensity, the midprice
    volatility, the inventory bound -- so they are quantities the modeller fixes, never estimates of the latent
    process. `alpha` is the one exception and is scaled by its analytic stationary standard deviation.
    """

    def __init__(self, env, features: str = "proxy", flow_window: int = 5, return_window: int = 20,
                 alpha_index: Optional[int] = None):
        super().__init__(env)
        assert not env.normalise_observation_space_, (
            "InformationSet reads the raw state; build the environment with normalise_observation_space=False."
        )
        assert features in FEATURE_SETS, f"features must be one of {sorted(FEATURE_SETS)}"
        assert flow_window >= 1 and return_window >= 2, (
            "a return window of one step compares the price to itself and is identically zero."
        )
        self.features: Sequence[str] = FEATURE_SETS[features]
        self.feature_set, self.alpha_index = features, alpha_index
        self.flow_window, self.return_window = flow_window, return_window
        if "alpha" in self.features:
            assert alpha_index is not None, "the oracle feature sets need alpha_index"

        n = env.num_trajectories
        self.flow_imbalance = _RollingSum(n, flow_window)
        self.flow_count = _RollingSum(n, flow_window)
        self.fill_imbalance = _RollingSum(n, flow_window)
        self.price_history = np.zeros((n, return_window))
        self.history_position = 0
        self.steps_taken = 0

        self._intensity = float(np.mean(env.model_dynamics.arrival_model.intensity))
        self._volatility = float(getattr(env.model_dynamics.midprice_model, "volatility", 1.0))
        self._step_size = env.step_size
        # sd of a window sum of independent Bernoulli arrivals on the two sides
        self._flow_scale = np.sqrt(max(2.0 * self._intensity * self._step_size * flow_window, 1e-12))
        self._return_scale = np.sqrt(max(self._volatility ** 2 * self._step_size * return_window, 1e-12))
        self._alpha_scale = self._stationary_alpha_sd(env)

        self.observation_space = gym.spaces.Box(
            low=np.float32(-FEATURE_BOUND * np.ones(len(self.features))),
            high=np.float32(FEATURE_BOUND * np.ones(len(self.features))),
        )

    @staticmethod
    def _stationary_alpha_sd(env) -> float:
        """Stationary sd of the alpha sub-process, from its own parameters.

        `OuMidpriceModel.update` applies its mean reversion as -speed * (alpha - level) without a factor dt, so
        the discrete recursion is alpha <- (1 - speed) alpha + vol sqrt(dt) Z and the stationary variance is
        vol^2 dt / (1 - (1 - speed)^2). Reproducing the discretisation actually used, rather than the continuous
        time formula, is what keeps the scaling correct.
        """
        ou = getattr(env.model_dynamics.midprice_model, "ou_process", None)
        if ou is None:
            return 1.0
        retention = (1.0 - ou.mean_reversion_speed) ** 2
        variance = ou.volatility ** 2 * ou.step_size / max(1.0 - retention, 1e-12)
        return float(np.sqrt(max(variance, 1e-12)))

    def _reset_buffers(self, state: np.ndarray):
        for rolling in (self.flow_imbalance, self.flow_count, self.fill_imbalance):
            rolling.reset()
        self.price_history[:] = state[:, ASSET_PRICE_INDEX][:, None]
        self.history_position = 0
        self.steps_taken = 0

    def _push_step(self, state: np.ndarray):
        arrivals, fills = self.env.last_arrivals, self.env.last_fills
        zeros = np.zeros(state.shape[0])
        if arrivals is not None:
            # entry 0 is an exogenous SELL market order, entry 1 an exogenous BUY: a positive imbalance means
            # buying pressure, which is what lifts the maker's ask.
            self.flow_imbalance.push(arrivals[:, 1] - arrivals[:, 0])
            self.flow_count.push(arrivals[:, 1] + arrivals[:, 0])
        else:
            self.flow_imbalance.push(zeros)
            self.flow_count.push(zeros)
        if fills is not None:
            # a positive imbalance means the maker sold more than it bought over the window
            self.fill_imbalance.push(fills[:, ASK_INDEX] - fills[:, BID_INDEX])
        else:
            self.fill_imbalance.push(zeros)
        self.price_history[:, self.history_position] = state[:, ASSET_PRICE_INDEX]
        self.history_position = (self.history_position + 1) % self.return_window
        self.steps_taken += 1

    def _observe(self, state: np.ndarray) -> np.ndarray:
        values = {}
        if "inventory" in self.features:
            values["inventory"] = state[:, INVENTORY_INDEX] / self.max_inventory
        if "time_left" in self.features:
            values["time_left"] = (self.terminal_time - state[:, TIME_INDEX]) / self.terminal_time
        if "flow_imbalance" in self.features:
            values["flow_imbalance"] = self.flow_imbalance.total / self._flow_scale
        if "flow_intensity" in self.features:
            expected = 2.0 * self._intensity * self._step_size * self.flow_window
            values["flow_intensity"] = (self.flow_count.total - expected) / self._flow_scale
        if "fill_imbalance" in self.features:
            values["fill_imbalance"] = self.fill_imbalance.total / self._flow_scale
        if "recent_return" in self.features:
            oldest = self.price_history[:, self.history_position]
            values["recent_return"] = (state[:, ASSET_PRICE_INDEX] - oldest) / self._return_scale
        if "alpha" in self.features:
            values["alpha"] = state[:, self.alpha_index] / self._alpha_scale
        stacked = np.stack([values[name] for name in self.features], axis=1)
        return np.clip(stacked, -FEATURE_BOUND, FEATURE_BOUND).astype(np.float32)

    def reset(self):
        state = self.env.reset()
        self._reset_buffers(state)
        return self._observe(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._push_step(state)
        return self._observe(state), reward, done, info

    @property
    def raw_state(self) -> np.ndarray:
        return self.env.state
