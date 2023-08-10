# RSSA-and-abstract-safe-control

Codes for the following two projects:
- Robust Safe Control
- Abstract Safe Control

## Robust Safe Control
See `src/pybullet-dynamics/RSSA_convex.py` and `src/pybullet-dynamics/RSSA_gaussian.py`.

## Abstract Safe Control
See `src/pybullet-dynamics/panda_rod_env/panda_abstract_safe_control.py`

Note: when using `PandaEnv`, please install [pytorch-kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) package.

## Multi-Modal Robust Safe Control
See `src/pybullet-dynamics/MMRSSA_*`
- SegWayAdditiveNoiseEnv: add noise on $f(x)$
- SegWayMultiplicativeNoiseEnv: $K_m$
- SegWayMultiplicativeAllNoiseEnv: add noise on $f$ and $g$

