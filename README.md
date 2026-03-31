# Anonymous ICML 2026 Submission

## Requirements

```bash
pip install --upgrade "jax[cpu]"
pip install dm-haiku optax chex
pip install git+https://github.com/deepmind/enn
```

## Usage

```bash
python epipivot.py
```

This will:
- Generate 100 training and 50 test LP instances
- Train the model for 180 epochs
- Evaluate against classical baselines (LC, Steepest Edge, Bland)
- Save results to `epipivot_results.csv`

## Output

```
=== Results: (m,n)=(25,50) | dist=base ===
  Method        Cost (mean)   Cost (std)   Steps  Success
  --------------------------------------------------------
  EpiPivot            xx.xx        xx.xx    xx.x    xx.x%
  LC                  xx.xx        xx.xx    xx.x    xx.x%
  Steepest            xx.xx        xx.xx    xx.x    xx.x%
  Bland               xx.xx        xx.xx    xx.x    xx.x%
```

## Reproducibility

The random seed is controlled by `GLOBAL_SEED` in the script. Changing this value will generate a different set of LP instances.
