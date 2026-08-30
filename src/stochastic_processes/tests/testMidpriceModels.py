"""Regression tests for the two short term alpha midprice models.

Both were unusable before: their `min_value` / `max_value` embedded the sub-process bounds as (1, 1) arrays inside a
row vector, which raised on construction, and their `update` wrote (n, 1) columns into (n,) slices, which broadcast
to (n, n) as soon as more than one trajectory was simulated. They are the carrier of the latent alpha that
`StateDependentPoissonArrivalModel` reads, so they are pinned down here.
"""

from unittest import TestCase, main

import numpy as np

from stochastic_processes.midprice_models import (
    OuJumpMidpriceModel,
    OuMidpriceModel,
    ShortTermJumpAlphaMidpriceModel,
    ShortTermOuAlphaMidpriceModel,
)

STEP_SIZE = 0.005
NO_ARRIVALS = np.zeros((1, 2))


def empty_inputs(num_trajectories: int):
    zeros = np.zeros((num_trajectories, 2))
    return zeros, zeros, zeros


class testShortTermAlphaMidpriceModels(TestCase):
    def test_models_are_constructible_with_well_shaped_bounds(self):
        for model_class in (ShortTermOuAlphaMidpriceModel, ShortTermJumpAlphaMidpriceModel):
            model = model_class(step_size=STEP_SIZE)
            for name in ("min_value", "max_value", "initial_state"):
                bound = getattr(model, name)
                self.assertEqual(bound.dtype.kind, "f", f"{model_class.__name__}.{name} is not numeric")
                self.assertEqual(bound.shape, (1, 2), f"{model_class.__name__}.{name} has the wrong shape")

    def test_update_is_vectorised_over_trajectories(self):
        for model_class in (ShortTermOuAlphaMidpriceModel, ShortTermJumpAlphaMidpriceModel):
            for num_trajectories in (1, 5):
                model = model_class(step_size=STEP_SIZE, num_trajectories=num_trajectories)
                for _ in range(10):
                    model.update(*empty_inputs(num_trajectories))
                self.assertEqual(
                    model.current_state.shape,
                    (num_trajectories, 2),
                    f"{model_class.__name__} lost its shape with {num_trajectories} trajectories",
                )
                self.assertTrue(np.all(np.isfinite(model.current_state)))

    def test_trajectories_are_independent(self):
        num_trajectories = 5
        model = ShortTermOuAlphaMidpriceModel(step_size=STEP_SIZE, num_trajectories=num_trajectories, seed=3)
        for _ in range(20):
            model.update(*empty_inputs(num_trajectories))
        prices = model.current_state[:, 0]
        self.assertEqual(len(np.unique(prices)), num_trajectories, "trajectories should not be identical")

    def test_the_second_state_component_is_the_alpha_signal(self):
        model = ShortTermOuAlphaMidpriceModel(step_size=STEP_SIZE, num_trajectories=4, seed=3)
        for _ in range(20):
            model.update(*empty_inputs(4))
        np.testing.assert_array_equal(model.current_state[:, 1], model.ou_process.current_state[:, 0])


class testOuMidpriceModelDiscretisation(TestCase):
    def test_mean_reversion_is_applied_without_the_step_size(self):
        """Documents a discretisation quirk rather than endorsing it: `OuMidpriceModel.update` applies
        `-speed * (state - level)` with no `* step_size` factor. The default speed of 1.0 therefore wipes the state
        out at every step, leaving white noise instead of a persistent signal. Anything relying on a *predictable*
        alpha must set the speed to roughly kappa * step_size itself."""
        num_trajectories = 500
        white_noise = OuMidpriceModel(
            mean_reversion_level=0.0,
            mean_reversion_speed=1.0,
            volatility=20.0,
            initial_price=0.0,
            step_size=STEP_SIZE,
            num_trajectories=num_trajectories,
            seed=5,
        )
        persistent = OuMidpriceModel(
            mean_reversion_level=0.0,
            mean_reversion_speed=0.05,
            volatility=20.0,
            initial_price=0.0,
            step_size=STEP_SIZE,
            num_trajectories=num_trajectories,
            seed=5,
        )
        for model, expected_autocorrelation in ((white_noise, 0.0), (persistent, 0.95)):
            path = []
            for _ in range(400):
                model.update(*empty_inputs(num_trajectories))
                path.append(model.current_state[:, 0].copy())
            path = np.array(path).T[:, 100:]  # burn in
            autocorrelation = np.corrcoef(path[:, :-1].ravel(), path[:, 1:].ravel())[0, 1]
            self.assertAlmostEqual(autocorrelation, expected_autocorrelation, delta=0.05)


class testEveryMidpriceModelIsReproducible(TestCase):
    """A model that draws from the global numpy generator instead of its own is not seedable, and nothing in the
    package's interface reveals that. `HestonMidpriceModel` was the one model in this state; the check is kept
    over the whole family so a future addition cannot reintroduce it silently."""

    def test_the_same_seed_gives_the_same_path(self):
        from stochastic_processes.midprice_models import (BrownianMotionMidpriceModel,
                                                          ConstantElasticityOfVarianceMidpriceModel,
                                                          GeometricBrownianMotionMidpriceModel,
                                                          HestonMidpriceModel, OuMidpriceModel)
        for model_class in (BrownianMotionMidpriceModel, GeometricBrownianMotionMidpriceModel,
                            OuMidpriceModel, HestonMidpriceModel,
                            ConstantElasticityOfVarianceMidpriceModel):
            paths = []
            for _ in range(2):
                model = model_class(num_trajectories=5, seed=42)
                model.reset()
                for _ in range(10):
                    model.update(None, None, None)
                paths.append(model.current_state[:, 0].copy())
            np.testing.assert_array_equal(paths[0], paths[1], err_msg=model_class.__name__)

    def test_different_seeds_give_different_paths(self):
        """A guard against the previous test passing because a model is deterministic."""
        from stochastic_processes.midprice_models import HestonMidpriceModel

        paths = []
        for seed in (1, 2):
            model = HestonMidpriceModel(num_trajectories=5, seed=seed)
            model.reset()
            for _ in range(10):
                model.update(None, None, None)
            paths.append(model.current_state[:, 0].copy())
        self.assertFalse(np.array_equal(paths[0], paths[1]))


if __name__ == "__main__":
    main()
