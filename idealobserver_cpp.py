"""
Wrapper around the C++ port of the ideal observer (module pf_cpp, sources in cpp/), with the same interface as IdealObsPF.

The module has to be built for each machine, with:
    python cpp/build_cpp.py
"""
import numpy as np

import pf_cpp
from subject import Subject, IdealObsParams
from utils_sample import rng


def to_cpp_value(value):
    """
    Convert a hyperparameter or observation to the type expected by pf_cpp (arrays become float64)
    """
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    return value


def to_cpp_params(params: IdealObsParams) -> pf_cpp.IdealObsParams:
    """
    Build the C++ IdealObsParams from the Python one. \\
    The C++ model classes have the same names and constructor arguments as the Python ones, so each model is constructed from its hyperparameters as in Python.
    """
    def build(model_type: type, hyp: dict):
        return getattr(pf_cpp, model_type.__name__)(**{k: to_cpp_value(v) for k, v in hyp.items()})

    return pf_cpp.IdealObsParams(
        context=build(params["type_context"], params["hyp_context"]),
        obs=build(params["type_obs"], params["hyp_obs"]),
        rew=build(params["type_rew"], params["hyp_rew"]),
    )


class IdealObsPFCpp(Subject):
    """
    Particle filtering implementation of the ideal observer, running in C++.
    """
    def __init__(self, N: int, hyp_params: IdealObsParams, seed: int | None = None):
        """
        Differs from IdealObsPF: the C++ ensemble has its own random number generator, rather than using the one in utils_sample.

        :param int seed: seed for the C++ random number generator. If None, it is drawn from utils_sample.rng, so that seeding that generator makes runs reproducible.
        """
        if seed is None:
            seed = int(rng.integers(0, 2**63 - 1))

        self.N = N
        self.hyp_params = hyp_params
        self._pf = pf_cpp.IdealObsPF(N, to_cpp_params(hyp_params), seed)

    def before_action(self, o_t: np.ndarray):
        """
        Process the incoming observation o_t
        """
        self._pf.before_action(to_cpp_value(o_t))

    def select_action(self) -> int:
        """
        Select an action a_t
        """
        return self._pf.select_action()

    def after_action(self, o_t: np.ndarray, a_t: int, r_t: float):
        """
        Process the resulting o_t, a_t, r_t emissions at time t
        """
        self._pf.after_action(to_cpp_value(o_t), int(a_t), float(r_t))

    def resample(self):
        """
        Resample particles according to weights
        """
        self._pf.resample()

    # Tracked quantities, as the same types as in IdealObsPF
    @property
    def weights(self) -> np.ndarray:
        return self._pf.weights

    @property
    def log_weights(self) -> np.ndarray:
        return self._pf.log_weights

    @property
    def vec_s_t(self) -> np.ndarray:
        return self._pf.vec_s_t.astype(int)

    @property
    def vec_c_o_t(self) -> np.ndarray:
        return self._pf.vec_c_o_t.astype(int)

    @property
    def vec_c_r_t(self) -> np.ndarray:
        return self._pf.vec_c_r_t.astype(int)

    @property
    def vec_j_t(self) -> np.ndarray:
        return self._pf.vec_j_t.astype(int)

    @property
    def p_action(self) -> np.ndarray:
        return self._pf.p_action

    @property
    def p_state(self) -> np.ndarray:
        return self._pf.p_state

    @property
    def p_jump(self) -> float:
        return self._pf.p_jump
