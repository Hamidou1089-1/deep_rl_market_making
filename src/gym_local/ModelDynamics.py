import abc
import gymnasium as gym
from copy import copy
from typing import Optional
        
import numpy as np
from numpy.random import default_rng


from gym_local.index_names import ASSET_PRICE_INDEX, CASH_INDEX, INVENTORY_INDEX, BID_INDEX, ASK_INDEX

from stochastic_processes.arrival_models import ArrivalModel
from stochastic_processes.fill_probability_models import FillProbabilityModel
from stochastic_processes.midprice_models import MidpriceModel
from stochastic_processes.price_impact_models import PriceImpactModel


class ModelDynamics(metaclass=abc.ABCMeta):
    # Fills are resolved before the market is advanced, so they cannot depend on the price move over the step.
    # A subclass that needs that move -- an adverse fill mechanism -- sets this to True and implements
    # `resolve_fills`; see `AdverseFillModelDynamics` and `TradingEnvironment._update_state_after_price_move`.
    fills_require_price_move = False

    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
    ):
        self.midprice_model = midprice_model
        self.arrival_model = arrival_model
        self.fill_probability_model = fill_probability_model
        self.price_impact_model = price_impact_model
        self.num_trajectories = num_trajectories
        self.rng = default_rng(seed)
        self.seed_ = seed
        self.fill_multiplier = self._get_fill_multiplier()
        self.round_initial_inventory = False
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.state = None

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        pass
    
    def get_fills(self, action: np.ndarray):
        pass
    
    def get_arrivals_and_fills(self, action: np.ndarray):
        return None, None 

    def apply_self_impact(self, executed: np.ndarray):
        """Let the agent's own executions move the midprice. A no-op unless a subclass implements it.

        Called by the environment after the executions of a step are known. Separate from `update_state` because
        it acts on the market, not on the agent: the cash and inventory of this step have already settled at the
        price the order was resting at, and the impact is what the *next* step inherits.
        """

    def _limit_depths(self, action: np.ndarray):
        """Quoted depths, floored at zero.

        The limit order action space is `Box(low=0, high=max_depth)`, but the closed forms are formulas rather
        than policies and do not respect it: at large inventory Cartea-Jaimungal and Avellaneda-Stoikov both
        return a negative depth on the side that reduces the position, i.e. an order posted *through* the mid.
        That is a market order, not a limit order, and the simulator has no representation for it --
        `ExponentialFillFunction` returns exp(-kappa delta) > 1, which is not a probability, and an adverse fill
        rule of the form "the price moved past the quote" becomes trivially true.

        Measured on the Cartea-Jaimungal control at kappa = 1.5, negative depths are only 0.59 % of quotes but
        carried 53 % of all adverse fills, at a rate of 0.71 against 0.004 elsewhere. Flooring at zero keeps a
        quote at the touch, which is the tightest thing a limit order can be, and stops the two mechanisms from
        being driven by a state neither of them models.
        """
        return np.maximum(action[:, 0:2], 0.0)

    def get_action_space(self) -> gym.spaces.Space:
        pass
    
    def get_required_stochastic_processes(self):
        pass
    
    def _get_max_depth(self) -> Optional[float]:
        if self.fill_probability_model is not None:
            return self.fill_probability_model.max_depth
        else:
            return None

    def _get_max_speed(self) -> float:
        if self.price_impact_model is not None:
            return self.price_impact_model.max_speed
        else:
            return None

    def _get_fill_multiplier(self):
        ones = np.ones((self.num_trajectories, 1))
        return np.append(-ones, ones, axis=1)

    def _check_processes_are_not_none(self, processes):
        for process in processes:
            self._check_process_is_not_none(process)

    def _check_process_is_not_none(self, process: str):
        assert getattr(self, process) is not None, f"This model dynamics cannot have env.{process} to be None."

    @property
    def midprice(self):
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)


class LimitOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
        self_impact: float = 0.0,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.self_impact = self_impact
        self.max_depth = max_depth or self._get_max_depth()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True
        
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
                axis=1,
            )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(low=np.float32(0.0), high=np.float32(self.max_depth), shape=(2,))
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals(self.state)
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths, self.state)
        return arrivals, fills

    def apply_self_impact(self, executed: np.ndarray):
        """Permanent impact of the trades the agent's quotes absorbed, signed by the aggressor.

        A fill on the ask means an aggressive *buy* lifted the quote, and aggressive buying pushes the price up;
        a fill on the bid is an aggressive sell and pushes it down. The market maker is therefore pushed against
        its own new position on every trade it does, which is a third and distinct channel of adverse selection:

          * the informational one comes from a factor that tilts the flow, and is predictable from that factor;
          * the mechanical one comes from the price running through a resting quote, and is predictable only to
            the extent the move itself is;
          * this one is caused by the agent's own trading, so it is perfectly predictable in principle -- the
            agent knows what it just traded -- and is avoidable only by trading less or quoting wider.

        Reported separately for exactly that reason: the three have different ceilings on what any policy can do
        about them. `self_impact = 0` leaves the midprice untouched and draws no random numbers, so the mechanism
        off is bit-for-bit the environment without it.
        """
        if not self.self_impact:
            return
        signed = executed[:, ASK_INDEX] - executed[:, BID_INDEX]
        self.midprice_model.current_state[:, 0] += self.self_impact * signed
        self.state[:, ASSET_PRICE_INDEX] = self.midprice_model.current_state[:, 0]


class AtTheTouchModelDynamics(ModelDynamics):
    """ModelDynamics for 'touch'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        fixed_market_half_spread: float = 0.5,
        seed: int = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.round_initial_inventory = True
        self.fixed_market_half_spread = fixed_market_half_spread
        
    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self.fixed_market_half_spread * self.fill_multiplier),
                axis=1,
            )
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)

    def _post_at_touch(self, action: np.ndarray):
        return action[:, 0:2]

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.MultiBinary(2) 
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals(self.state)
        fills = self._post_at_touch(action)
        return arrivals, fills


class LimitAndMarketOrderModelDynamics(ModelDynamics):
    """ModelDynamics for 'limit_and_market'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        arrival_model : ArrivalModel  = None,
        fill_probability_model : FillProbabilityModel  = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth : float = None,
        fixed_market_half_spread : float = 0.5,
    ):
        super().__init__(midprice_model = midprice_model,
                        arrival_model = arrival_model,
                        fill_probability_model = fill_probability_model, 
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_depth = max_depth or self._get_max_depth()
        self.fixed_market_half_spread = fixed_market_half_spread
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = True

    def _market_order_buy(self, action: np.ndarray):
        return action[:, 2 + BID_INDEX]
        
    def _market_order_sell(self, action: np.ndarray):
        return action[:, 2 + ASK_INDEX]

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        mo_buy = np.single(self._market_order_buy(action) > 0.5)
        mo_sell = np.single(self._market_order_sell(action) > 0.5)
        best_bid = (self.midprice - self.fixed_market_half_spread).reshape(-1,)
        best_ask = (self.midprice + self.fixed_market_half_spread).reshape(-1,)
        self.state[:, CASH_INDEX] += mo_sell * best_bid - mo_buy * best_ask
        self.state[:, INVENTORY_INDEX] += mo_buy - mo_sell
        self.state[:, INVENTORY_INDEX] += np.sum(arrivals * fills * -self.fill_multiplier, axis=1)
        self.state[:, CASH_INDEX] += np.sum(
                self.fill_multiplier
                * arrivals
                * fills
                * (self.midprice + self._limit_depths(action) * self.fill_multiplier),
                axis=1,
            )

    def get_action_space(self) -> gym.spaces.Space:
        assert self.max_depth is not None, "For limit orders max_depth cannot be None."
        # agent chooses spread on bid and ask
        return gym.spaces.Box(
                low=np.zeros(4),
                high=np.array([self.max_depth, self.max_depth, 1, 1], dtype=np.float32),
            )
    
    def get_required_stochastic_processes(self):
        processes = ["arrival_model", "fill_probability_model"]
        return processes

    def get_arrivals_and_fills(self, action: np.ndarray):
        arrivals = self.arrival_model.get_arrivals(self.state)
        depths = self._limit_depths(action)
        fills = self.fill_probability_model.get_fills(depths, self.state)
        return arrivals, fills


class TradinghWithSpeedModelDynamics(ModelDynamics):
    """ModelDynamics for 'speed'."""
    def __init__(
        self,
        midprice_model : MidpriceModel  = None,
        price_impact_model : PriceImpactModel = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_speed : float = None,
    ):
        super().__init__(midprice_model = midprice_model,
                        price_impact_model = price_impact_model,
                        num_trajectories = num_trajectories,
                        seed = seed)
        self.max_speed = max_speed or self._get_max_speed()
        self.required_processes = self.get_required_stochastic_processes()
        self._check_processes_are_not_none(self.required_processes)
        self.round_initial_inventory = False

    def update_state(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray):
        price_impact = self.price_impact_model.get_impact(action)
        execution_price = self.midprice + price_impact
        volume = action * self.midprice_model.step_size
        self.state[:, CASH_INDEX] -= np.squeeze(volume * execution_price)
        self.state[:, INVENTORY_INDEX] += np.squeeze(volume)

    def get_action_space(self) -> gym.spaces.Space:
        # agent chooses speed of trading: positive buys, negative sells
        return gym.spaces.Box(low=np.float32([-self.max_speed]), high=np.float32([self.max_speed]))
    
    def get_required_stochastic_processes(self):
        processes = ["price_impact_model"]
        return processes


class AdverseFillModelDynamics(LimitOrderModelDynamics):
    """The fill mechanism of Lalor and Swishchuk (2024), section 4, generalised to limit orders quoted at a depth.

    In the benchmark simulator of Cartea et al. (2015) -- `LimitOrderModelDynamics` above -- market orders are
    drawn independently of the price path. The midprice can therefore walk straight through a resting quote
    without filling it, and every fill the agent receives is non-adverse by construction. The paper's empirical
    evidence on CME futures (ES, NQ, CL, ZN) is that the overwhelming majority of fills are in fact adverse, and
    it proposes two corrections.

    **Adverse fills** (eq. 16-17), a deterministic function of the realised path rather than a separate draw::

        AF_ask = 1{S_{t+dt} - S_t >  delta_ask}       the ask was traded through
        AF_bid = 1{S_t - S_{t+dt} >  delta_bid}       the bid was traded through

    At the touch (``delta = 0``) this is exactly the paper's own rule -- any upward (downward) move fills the ask
    (bid) -- and it encodes the book's mechanical constraint that the price cannot move past a resting order
    without executing it first. Quoting deeper is what buys protection: the move must clear the depth.

    **Queue position** (eq. 13-15). The benchmark implicitly assumes the agent is at the front of the queue and is
    filled by every matching market order, i.e. ``p = 1``. The paper attaches a Bernoulli probability to the
    non-adverse branch instead, and calibrates ``p = 0.2`` on CME data::

        NF = arrival . fill(delta) . Bernoulli(p)

    **Combination** (eq. 18-19): ``N = max(AF, NF)`` -- an execution occurs if either branch fires.

    The two corrections pull in opposite directions, which is the point: ``p < 1`` removes benign fills the agent
    was being handed for free, while the adverse branch adds fills it cannot refuse. The paper's finding is that
    the second dominates.

    With ``track_adverse_fills=False`` and ``queue_probability=1.0`` this class draws exactly the random numbers
    `LimitOrderModelDynamics` draws and reproduces it path by path, so the benchmark environment is an exact
    nested limit of the improved one.
    """

    fills_require_price_move = True

    def __init__(
        self,
        midprice_model: MidpriceModel = None,
        arrival_model: ArrivalModel = None,
        fill_probability_model: FillProbabilityModel = None,
        num_trajectories: int = 1,
        seed: int = None,
        max_depth: float = None,
        track_adverse_fills: bool = True,
        queue_probability: float = 1.0,
        self_impact: float = 0.0,
    ):
        super().__init__(
            midprice_model=midprice_model,
            arrival_model=arrival_model,
            fill_probability_model=fill_probability_model,
            num_trajectories=num_trajectories,
            seed=seed,
            max_depth=max_depth,
            self_impact=self_impact,
        )
        assert 0.0 <= queue_probability <= 1.0, "queue_probability is a probability."
        self.track_adverse_fills = track_adverse_fills
        self.queue_probability = queue_probability
        # Set by the environment while cash and inventory settle, so that executions clear at the price the
        # resting order was quoted from rather than at the price the market has just moved to.
        self.execution_midprice = None
        self.last_adverse_fills = np.zeros((num_trajectories, 2))
        self.last_non_adverse_fills = np.zeros((num_trajectories, 2))

    @property
    def midprice(self):
        if self.execution_midprice is not None:
            return self.execution_midprice
        return self.midprice_model.current_state[:, 0].reshape(-1, 1)

    def resolve_fills(self, arrivals: np.ndarray, fills: np.ndarray, action: np.ndarray, midprice_before: np.ndarray):
        """Combine the two fill branches once the price move over the step is known.

        Returns `(arrivals, fills)` with the arrivals collapsed to ones, so that the inherited `update_state`,
        which settles the elementwise product of the two, settles exactly the executions of eq. (18)-(19). The
        two branches are also stored on the instance, which is what lets a diagnostic count adverse against
        non-adverse fills the way the paper's Table 4 does.
        """
        depths = self._limit_depths(action)
        non_adverse = np.asarray(arrivals, dtype=float) * np.asarray(fills, dtype=float)
        if self.queue_probability < 1.0:
            non_adverse = non_adverse * (self.rng.random(non_adverse.shape) < self.queue_probability)
        if self.track_adverse_fills:
            price_move = self.midprice_model.current_state[:, 0].reshape(-1, 1) - midprice_before
            adverse = np.concatenate(
                [
                    -price_move > depths[:, BID_INDEX].reshape(-1, 1),  # the bid was traded through
                    price_move > depths[:, ASK_INDEX].reshape(-1, 1),  # the ask was traded through
                ],
                axis=1,
            ).astype(float)
        else:
            adverse = np.zeros_like(non_adverse)
        self.last_non_adverse_fills, self.last_adverse_fills = non_adverse, adverse
        executed = np.maximum(non_adverse, adverse)
        return np.ones_like(executed), executed
