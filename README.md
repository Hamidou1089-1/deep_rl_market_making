# deep rl market making

Built on **`mbt_gym`** — Jerome, Sánchez-Betancourt, Savani and Herdegen (2022),
*mbt-gym: Reinforcement learning for model-based limit order book trading*
([repo](https://github.com/JJJerome/mbt_gym), [paper](doc/mbt_gym%20paper.pdf)).
Their code is the simulator here. It is BSD-3-Clause and the licence travels
with it in [`src/LICENSE.mbt_gym`](src/LICENSE.mbt_gym).

## Layout

```
src/gym/                  their gym/                   → import gym.*
src/agents/               their agents/                → import agents.*
src/rewards/              their rewards/               → import rewards.*
src/stochastic_processes/ their stochastic_processes/  → import stochastic_processes.*
notebooks/                their notebooks: baseline agents, SB3, AS and CJP replications
doc/                      the papers and the specification
data/binance/             market data (gitignored)
```

`src/` is the source root, not a package. The four packages come straight out of
`mbt_gym/`, so the imports read `from gym.TradingEnvironment import ...`.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

The editable install is what makes `import gym` resolve to `src/gym`. The
notebooks also add `../src` to the path, so they run either way.

## What was changed in their code, and why

Not byte-identical to upstream. Taking the four packages out of `mbt_gym/`
forces an import rewrite, and one of them is named `gym`, which collides with
the library of the same name. Everything changed is listed here so the
provenance stays auditable.

| change | files | why |
|---|---|---|
| `from mbt_gym.X import` → `from X import` | 12 | the parent package is gone |
| `import gym` → `import gymnasium as gym`, and the two `from gym…` variants | 11 | `src/gym` takes the name, so the library has to be reached under another. Only `gym.Env`, `gym.Wrapper` and `gym.spaces.Box` were used, all present in gymnasium |
| `_flatten_obs` → `_stack_obs as _flatten_obs` | 1 | SB3 v2 renamed it; identical signature |

The legacy `gym` package is **uninstalled on purpose**: while it was present,
`import gym` was ambiguous between the library and `src/gym`, and resolution
depended on path order. Do not reinstall it.

Side effect worth having: the spaces are now gymnasium-native, which is what
stable-baselines3 v2 requires.

## Two upstream issues that remain, and are not layout-related

**`StableBaselinesTradingEnvironment` needs one method to work with SB3 v2.**
Its `VecEnv.__init__` calls `get_attr("render_mode")`, and the upstream stub
returns `None`, so `all(... in None)` raises. Verified against SB3 2.9.0:

```python
from gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment

class SbBridge(StableBaselinesTradingEnvironment):
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name, None)] * self.num_envs
```

**`get_sharpe_ratio`, `get_sortino_ratio` and `get_maximum_drawdown` divide by
portfolio value**, which starts at `initial_cash`, default `0.0`. At the default
they return `nan` or a silently wrong number; pass `initial_cash > 0`, and note
that the value it takes changes the answer.

## One trap in their simulation

`TradingEnvironment.step()` calls `model_dynamics.get_arrivals_and_fills()`
itself. Calling it beforehand to record what happened **draws a second time**:
the counts follow the right marginal law but are independent of the ones the
processes actually saw, and the extra draw shifts the generator. Wrap the method
and stash its result instead of calling it.

## Verified

All 27 modules import. Their four notebooks' import chains resolve. A
Cartea–Jaimungal agent quotes `0.672 ≈ 1/κ` at zero inventory, and
`PPO.learn` runs through the bridge above.

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
