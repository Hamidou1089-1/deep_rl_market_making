"""Learned market making agents, and the harness that makes them comparable.

Five algorithms, chosen to span the families the two surveys organise around rather than to collect the most
recent acronyms. Bai et al. (2024) report policy-gradient and value-based methods as the two dominant families in
financial applications; Ghasemi et al. (2025) separate them further by on- versus off-policy, stochastic versus
deterministic policy, and by whether the network carries state.

    PPO            on-policy policy gradient with a clipped trust region. The default throughout the market
                   making literature, and the reference point here.
    A2C            on-policy actor-critic without the trust region. Its only structural difference from PPO is
                   the absence of clipping, so the pair isolates what the trust region actually buys.
    SAC            off-policy, maximum entropy, stochastic policy. Sample efficient, and its entropy term is a
                   direct answer to a criterion whose optimum is famously flat in market making.
    TD3            off-policy, deterministic policy, twin critics. The DDPG lineage that Bai et al. cite for
                   continuous-action finance, with the overestimation fixes.
    RecurrentPPO   PPO with an LSTM. The principled arm for this environment specifically: the factor driving
                   adverse selection is latent, so the decision problem is partially observed and the sufficient
                   statistic is a function of history. The hand-built rolling features of `InformationSet` are one
                   guess at that statistic; a recurrent policy is asked to find its own.

**What is held equal.** Every algorithm sees the same environment, the same information set, the same seeds and
the same number of environment steps. Nothing else can be held equal without crippling one family or another --
an on-policy method wants wide vectorised rollouts, an off-policy method wants many gradient steps per
transition -- so the per-algorithm settings below differ, are stated explicitly, and are the same across every
arm of a comparison.

Chan and Shelton (2001), the first RL market maker, condition on inventory, order imbalance, the spread and
recent price changes. That is the `proxy` feature set of `gym_local.observation`, arrived at independently here.
"""

from typing import Callable, Optional

import numpy as np
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from sb3_contrib import RecurrentPPO

from agents.Agent import Agent
from gym_local.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment

# Shared where the algorithms admit a shared value. `gamma = 1` is not a choice: the criterion is a finite
# horizon sum with no discounting, so any other value optimises a different objective.
COMMON = dict(gamma=1.0, device="cpu", verbose=0)
NET_ARCH = [64, 64]

ALGORITHMS = {
    "PPO": dict(
        cls=PPO, policy="MlpPolicy", on_policy=True,
        kwargs=dict(n_epochs=10, learning_rate=3e-4, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                                       net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)))),
    "A2C": dict(
        cls=A2C, policy="MlpPolicy", on_policy=True,
        kwargs=dict(learning_rate=7e-4, gae_lambda=1.0, ent_coef=0.0, normalize_advantage=True,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                                       net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)))),
    "SAC": dict(
        cls=SAC, policy="MlpPolicy", on_policy=False,
        kwargs=dict(learning_rate=3e-4, batch_size=256, tau=0.005, ent_coef="auto",
                    learning_starts=10_000, train_freq=(1, "step"), gradient_steps=4,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=NET_ARCH))),
    "TD3": dict(
        cls=TD3, policy="MlpPolicy", on_policy=False,
        kwargs=dict(learning_rate=3e-4, batch_size=256, tau=0.005, policy_delay=2,
                    learning_starts=10_000, train_freq=(1, "step"), gradient_steps=4,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=NET_ARCH))),
    "RecurrentPPO": dict(
        cls=RecurrentPPO, policy="MlpLstmPolicy", on_policy=True,
        kwargs=dict(n_epochs=10, learning_rate=3e-4, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(activation_fn=torch.nn.ReLU, lstm_hidden_size=64, n_lstm_layers=1,
                                       net_arch=dict(pi=NET_ARCH, vf=NET_ARCH)))),
}


class LearnedAgent(Agent):
    """Adapts a trained Stable-Baselines 3 model to this project's `Agent` interface.

    Recurrent policies carry state between steps, so the hidden state and the episode-start flags are tracked
    here. `reset` must be called at the start of every episode; forgetting it would silently carry one episode's
    memory into the next, which is not an error SB3 raises.
    """

    def __init__(self, model, num_trajectories: int, deterministic: bool = True):
        self.model = model
        self.num_trajectories = num_trajectories
        self.deterministic = deterministic
        self.is_recurrent = isinstance(model, RecurrentPPO)
        self.reset()

    def reset(self):
        self.lstm_states = None
        self.episode_starts = np.ones(self.num_trajectories, dtype=bool)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        if not self.is_recurrent:
            action, _ = self.model.predict(observation, deterministic=self.deterministic)
            return action
        action, self.lstm_states = self.model.predict(
            observation, state=self.lstm_states, episode_start=self.episode_starts,
            deterministic=self.deterministic)
        self.episode_starts = np.zeros(self.num_trajectories, dtype=bool)
        return action


def make_model(name: str, env, seed: int, n_steps: int, **overrides):
    """Build an untrained model of the named algorithm on an already-wrapped trading environment.

    `n_steps` is the environment's episode length. For the on-policy methods it is also the rollout length, so a
    policy update sees whole episodes and the terminal inventory penalty is never split across two updates --
    which matters here, because that penalty is most of the signal.
    """
    assert name in ALGORITHMS, f"unknown algorithm {name!r}; choose from {sorted(ALGORITHMS)}"
    spec = ALGORITHMS[name]
    kwargs = dict(COMMON, **spec["kwargs"])
    if spec["on_policy"]:
        kwargs["n_steps"] = n_steps
        if name != "A2C":
            # A2C takes no minibatches; the others split the rollout into ten.
            kwargs["batch_size"] = max(env.num_envs * n_steps // 10, 1)
    kwargs.update(overrides)
    return spec["cls"](spec["policy"], env, seed=seed, **kwargs)


def train(name: str, env_factory: Callable[[], object], seed: int, total_steps: int,
          torch_threads: int = 1, **overrides):
    """Train one algorithm for a fixed number of environment steps.

    `torch_threads = 1` is deliberate and worth keeping: these networks are small enough that thread
    synchronisation costs more than the parallelism gains, by an order of magnitude on this machine.
    """
    torch.set_num_threads(torch_threads)
    wrapped = env_factory()
    env = StableBaselinesTradingEnvironment(wrapped)
    model = make_model(name, env, seed=seed, n_steps=wrapped.n_steps, **overrides)
    model.learn(total_timesteps=total_steps, progress_bar=False)
    return model


def agent_factory(model, num_trajectories: int, deterministic: bool = True) -> Callable[[object], Agent]:
    """A factory of the shape `gym_local.metrics.evaluate` expects."""
    def build(env) -> Agent:
        return LearnedAgent(model, num_trajectories=num_trajectories, deterministic=deterministic)
    return build
