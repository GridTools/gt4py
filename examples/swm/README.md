# Shallow water model

The NCAR [SWM](https://github.com/NCAR/SWM) shallow water benchmark written with `gt4py.next`.
Start with the notebook; the other two files only support it.

- `swm.ipynb`: the model, a choice of backends, its validation, and an opt-in performance
  comparison across all backends.
- `swm_numpy.py`: a NumPy implementation of the same scheme, which the notebook validates
  against.
- `swm_reference.npz`: `u`, `v` and `p` after the full 4000-step benchmark on the 16x16 grid,
  from NCAR/SWM `ref/16x16`.
