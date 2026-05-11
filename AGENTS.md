# AGENTS.md

## Cursor Cloud specific instructions

### Overview

MVPinn is a pure-Python scientific computing library implementing Physics-Informed Neural Networks for Milne-Eddington inversion of solar Stokes profiles. There are no web services, databases, or Docker containers—just a single Python package.

### Quick reference

| Action | Command |
|---|---|
| Install deps | `pip install -r requirements.txt -e ".[dev]"` |
| Lint (critical) | `python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` |
| Lint (warnings) | `python3 -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics` |
| Import smoke test | `python3 -c "from src import MEInversionPINN; print('OK')"` |

### Caveats

- **No test suite exists.** The `tests/` directory mentioned in `README.md` is absent. CI only runs flake8 lint + a single import check (`python -c "from src import MEInversionPINN"`).
- **No sample FITS data in repo.** All `.fts`/`.fits` files are gitignored. Full end-to-end training/inference requires obtaining solar observation data externally. You can still validate the model with synthetic data (see below).
- **CPU-only in Cloud Agent VMs.** CUDA is not available; PyTorch falls back to CPU automatically via `torch.cuda.is_available()` checks throughout the codebase.
- **Existing lint issue.** `src/ME_utils.py` line 796 has an unused `global b_shared` that triggers F824 in the critical flake8 check. This is pre-existing in the codebase.
- **Use `python3` not `python`.** The VM does not have a `python` symlink; always invoke `python3` (or `python3 -m <module>`).

### Synthetic validation

To verify the model and physics engine work without FITS data:

```python
import torch
from src import MEInversionPINN, MEPhysicsLoss

model = MEInversionPINN(n_wavelengths=50)
model.eval()
with torch.no_grad():
    params = model(torch.randn(8, 200))

physics = MEPhysicsLoss()
_, stokes = physics(params, torch.linspace(-0.5, 0.5, 50))
print(f"Output: params {params.shape}, stokes {stokes.shape}")
```
