# deep rl market making

Measuring how far a closed-form market-making control degrades as the assumptions
it was derived under are relaxed, and whether a learned agent recovers what is
lost.

Built on **`mbt_gym`** — Jerome, Sánchez-Betancourt, Savani and Herdegen (2022),
*mbt-gym: Reinforcement learning for model-based limit order book trading*
([repo](https://github.com/JJJerome/mbt_gym), [paper](doc/mbt_gym%20paper.pdf)).
Their code is the simulator here. It is BSD-3-Clause and the licence travels
with it in [`src/LICENSE.mbt_gym`](src/LICENSE.mbt_gym).

## Layout

```
src/gym_local/            their gym/, renamed to stop shadowing the `gym` package
src/agents/               their agents/, plus calibration.py
src/rewards/              their rewards/
src/stochastic_processes/ their stochastic_processes/
notebooks/                AS and CJP replications, and the adverse selection tests
doc/                      the papers and the specification
data/                     market data (gitignored)
```

### Added to their simulator

| Module | What it adds |
|---|---|
| `stochastic_processes/arrival_models.py` | `StateDependentPoissonArrivalModel` — intensities tilted by a latent factor, so adverse selection *emerges* from one factor driving both order flow and the midprice drift rather than being imposed as a correlation. `sensitivity = 0` reproduces `PoissonArrivalModel` bit for bit. |
| `gym_local/ModelDynamics.py` | `AdverseFillModelDynamics` — the fill mechanism of Lalor and Swishchuk (2024): fills forced when the price trades through a resting quote, plus a queue-position probability on the rest. Switching both off reproduces `LimitOrderModelDynamics` bit for bit. |
| `gym_local/observation.py` | `InformationSet` — controls what the agent may condition on: the closed forms' own `(q, t)`, observable proxies for the latent factor, or the factor itself as a labelled oracle bound. |
| `gym_local/metrics.py` | Metrics that separate skill from passivity: total markout splits into "traded less" and "traded better", and the skew-signal correlation is partialled on inventory. |
| `agents/calibration.py` | The closed form with its four internal constants refitted by grid search. Not a closed form: a four-parameter policy search, and it belongs on the learned side of the ledger. |

Every relaxation is **nested**: with its parameter at zero the enriched
environment must reproduce the closed-form one trajectory by trajectory, which is
asserted rather than assumed. Without that, evaluating the closed form in the
enriched world is out of distribution and any gap is uninterpretable.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Defects found in the upstream simulator

`mbt_gym` is research code that does what its own paper needed. The defects below surface once it is pushed in
directions that paper did not go — a seeded paired comparison, a stochastic-volatility midprice, a per-trajectory
reward, an action space other than a `Box`. Each was found by a test in this repository, and each is listed with
the symptom that would have been observed rather than just the line that was wrong, because most of them fail
silently.

| Defect | What breaks | Status |
|---|---|---|
| `HestonMidpriceModel.update` draws from `np.random`, not `self.rng` | The one midprice model whose paths do not reproduce from their seed. Nesting checks and every paired comparison silently lose their pairing. | **fixed** |
| `ConstantElasticityOfVarianceMidpriceModel.update` does the same | Same. Found by the test written for the previous line — which is the argument for testing the family rather than the instance. | **fixed** |
| `TradingEnvironment.seed()` never seeds `model_dynamics.rng` | It seeds the stochastic processes only. Any mechanism drawing from the dynamics' own generator is seeded from entropy: two runs of one seed diverge. Nothing in the interface reveals it. | **fixed** |
| `RunningInventoryPenalty.calculate` casts with `int(is_terminal_step)` | Raises under numpy 2 when the caller passes a per-trajectory boolean array, which is the correct thing to pass when trajectories terminate separately. | **fixed** |
| The closed forms quote negative depths, and the fill model accepts them | At large inventory both Cartea–Jaimungal and Avellaneda–Stoikov return a depth below zero — a limit order posted *through* the mid, which is a market order and has no representation here. `ExponentialFillFunction` then returns `exp(-kappa delta) > 1`, which is not a probability. Measured at `kappa = 1.5`: 0.59 % of quotes, carrying 53 % of all adverse fills. | **fixed** (`_limit_depths` floors at zero) |
| `OuMidpriceModel.update` applies `-speed * (alpha - level)` with no factor `dt` | At the default speed of 1.0 the state is erased every step, so the "short-term alpha" is white noise. Measured lag-one autocorrelation: −0.001. | **documented, behaviour left intact** |
| `TradingEnvironment._get_normalised_action_space` reads `.low`/`.high` | `gym.spaces.MultiBinary` has neither, so `AtTheTouchModelDynamics` cannot be used with a normalised action space at all — which is the action space of the at-the-touch control in Cartea et al. (2015). | **open** |
| Calling `get_arrivals_and_fills()` to observe a step draws a second time | `TradingEnvironment.step()` calls it itself. An outside call returns counts with the right marginal law but independent of the ones the processes saw, and shifts the generator. | **worked around**: the environment now stores `last_arrivals` and `last_fills` |

The tests are also not collected by `pytest` out of the box — the repository names its modules `testXxx.py`. Fixed
here by adding `python_files` and `pythonpath` to `pyproject.toml`.

## Citation

```bibtex
@article{jerome2022mbtgym,
  title  = {mbt-gym: Reinforcement learning for model-based limit order book trading},
  author = {Jerome, Joseph and S{\'a}nchez-Betancourt, Leandro and Savani, Rahul
            and Herdegen, Martin},
  year   = {2022},
  url    = {https://github.com/JJJerome/mbt_gym}
}
```
