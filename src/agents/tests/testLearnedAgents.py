"""Tests for the learned-agent harness.

The point of the harness is that five algorithms from different families can be compared without any of them
being quietly advantaged. Three things therefore have to hold:

  * every algorithm builds and runs on this environment, including the recurrent one, whose state handling is
    the easiest thing to get silently wrong;
  * the settings that would change *what* is being optimised are shared -- above all `gamma = 1`, since the
    criterion is an undiscounted finite-horizon sum and any other value optimises a different objective;
  * a recurrent policy's memory does not leak from one episode into the next.
"""

from unittest import TestCase, main

import numpy as np

from agents.LearnedAgents import ALGORITHMS, LearnedAgent, make_model, train
from gym_local.ModelDynamics import LimitOrderModelDynamics
from gym_local.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from gym_local.TradingEnvironment import TradingEnvironment
from gym_local.observation import InformationSet
from rewards.RewardFunctions import CjMmCriterion
from stochastic_processes.arrival_models import StateDependentPoissonArrivalModel
from stochastic_processes.fill_probability_models import ExponentialFillFunction
from stochastic_processes.midprice_models import OuMidpriceModel, ShortTermOuAlphaMidpriceModel

TERMINAL_TIME, N_STEPS = 1.0, 20
STEP_SIZE = TERMINAL_TIME / N_STEPS
ALPHA_INDEX = 4


def get_env(n=8, seed=1, features="proxy"):
    midprice_model = ShortTermOuAlphaMidpriceModel(
        volatility=2.0,
        ou_process=OuMidpriceModel(mean_reversion_level=0.0, mean_reversion_speed=0.05, volatility=20.0,
                                   initial_price=0.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE,
                                   num_trajectories=n),
        initial_price=100.0, terminal_time=TERMINAL_TIME, step_size=STEP_SIZE, num_trajectories=n)
    dynamics = LimitOrderModelDynamics(
        midprice_model=midprice_model,
        arrival_model=StateDependentPoissonArrivalModel(
            np.array([140.0, 140.0]), 0.2, ALPHA_INDEX, STEP_SIZE, n),
        fill_probability_model=ExponentialFillFunction(1.5, STEP_SIZE, n), num_trajectories=n)
    env = TradingEnvironment(
        terminal_time=TERMINAL_TIME, n_steps=N_STEPS, model_dynamics=dynamics,
        reward_function=CjMmCriterion(per_step_inventory_aversion=0.01, terminal_inventory_aversion=1.0,
                                      terminal_time=TERMINAL_TIME),
        max_inventory=20, num_trajectories=n,
        normalise_action_space=True, normalise_observation_space=False)
    env.seed(seed)
    return InformationSet(env, features=features, alpha_index=ALPHA_INDEX)


class testTheRegistry(TestCase):
    def test_the_five_families_are_present(self):
        self.assertEqual(sorted(ALGORITHMS), ["A2C", "PPO", "RecurrentPPO", "SAC", "TD3"])

    def test_every_algorithm_builds_on_this_environment(self):
        wrapped = get_env()
        env = StableBaselinesTradingEnvironment(wrapped)
        for name in ALGORITHMS:
            model = make_model(name, env, seed=0, n_steps=wrapped.n_steps)
            self.assertIsInstance(model, ALGORITHMS[name]["cls"], name)

    def test_the_discount_factor_is_one_everywhere(self):
        """The criterion is an undiscounted finite-horizon sum. A discount would silently make each algorithm
        optimise a different objective, and the comparison would be meaningless."""
        wrapped = get_env()
        env = StableBaselinesTradingEnvironment(wrapped)
        for name in ALGORITHMS:
            self.assertEqual(make_model(name, env, seed=0, n_steps=wrapped.n_steps).gamma, 1.0, name)

    def test_an_unknown_algorithm_is_refused(self):
        wrapped = get_env()
        with self.assertRaises(AssertionError):
            make_model("DQN", StableBaselinesTradingEnvironment(wrapped), seed=0, n_steps=wrapped.n_steps)


class testTheAgentAdapter(TestCase):
    def test_actions_are_inside_the_action_space(self):
        for name in ("PPO", "SAC", "RecurrentPPO"):
            model = train(name, get_env, seed=0, total_steps=200)
            env = get_env(n=8, seed=5)
            agent = LearnedAgent(model, num_trajectories=8)
            observation = env.reset()
            for _ in range(5):
                action = agent.get_action(observation)
                self.assertEqual(action.shape, (8, 2), name)
                self.assertTrue(np.all(action >= env.action_space.low - 1e-6), name)
                self.assertTrue(np.all(action <= env.action_space.high + 1e-6), name)
                observation, _, _, _ = env.step(action)

    def test_a_recurrent_agent_starts_each_episode_without_memory(self):
        """`reset` clears the hidden state and re-arms the episode-start flags. Forgetting it carries one
        episode's memory into the next, which SB3 does not raise on."""
        model = train("RecurrentPPO", get_env, seed=0, total_steps=200)
        env = get_env(n=8, seed=5)
        agent = LearnedAgent(model, num_trajectories=8)

        observation = env.reset()
        first = agent.get_action(observation)
        for _ in range(5):  # advance the hidden state
            observation, _, _, _ = env.step(agent.get_action(observation))
        drifted = agent.get_action(env.reset())

        agent.reset()
        again = agent.get_action(env.reset())
        np.testing.assert_allclose(first, again, rtol=1e-5,
                                   err_msg="reset must restore the action taken on a fresh episode")
        self.assertFalse(np.allclose(first, drifted, rtol=1e-5),
                         "without reset the carried hidden state must still be visible in the action")

    def test_a_feedforward_agent_is_memoryless_by_construction(self):
        model = train("PPO", get_env, seed=0, total_steps=200)
        env = get_env(n=8, seed=5)
        agent = LearnedAgent(model, num_trajectories=8)
        observation = env.reset()
        first = agent.get_action(observation)
        for _ in range(3):
            agent.get_action(env.step(first)[0])
        np.testing.assert_allclose(first, agent.get_action(observation), rtol=1e-6,
                                   err_msg="a feedforward policy must depend on the observation alone")


if __name__ == "__main__":
    main()
