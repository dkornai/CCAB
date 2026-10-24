"""
Component tests of the C++ port (module pf_cpp, sources in cpp/) against the Python implementation.

Deterministic computations (posterior updates, predictive likelihoods, context hypotheses, beliefs) must agree up to rounding,
as both implementations compute in float64. Samplers are compared statistically, with fixed seeds.

Run with: python -m pytest tests
"""
import numpy as np
import pytest

import pf_cpp
import utils_sample
import models_context as MC
import models_obs_rew as M
from idealobserver_cpp import to_cpp_params, to_cpp_value
from particle import Particle

# Tolerances for float64 in both implementations, which differ only in the order of operations
RTOL = 1e-9
ATOL = 1e-12


# ----------------------------------------
#   Helpers
# ----------------------------------------

def to_np(x):
    return np.asarray(x, dtype=np.float64)


def assert_close(cpp, py, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(to_np(cpp), to_np(py), rtol=rtol, atol=atol)


def assert_dicts_close(cpp: dict, py: dict, rtol=RTOL, atol=ATOL):
    assert cpp.keys() == py.keys()
    for key in py:
        assert_close(cpp[key], py[key], rtol=rtol, atol=atol)


def cpp_model(py_cls: type, hyp: dict):
    """C++ counterpart of the Python model py_cls(**hyp)"""
    return getattr(pf_cpp, py_cls.__name__)(**{k: to_cpp_value(v) for k, v in hyp.items()})


HYP_NIW = {'mu0': np.array([0.0, 0.0]), 'kappa0': 0.1, 'nu0': 4.0, 'Lambda0': np.eye(2)}
HYP_NIW_3D = {'mu0': np.array([0.5, -0.2, 0.0]), 'kappa0': 0.5, 'nu0': 5.0,
              'Lambda0': np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 0.5]])}
HYP_NIG = {'mu0': np.array([0.0, 0.0]), 'kappa0': 0.1, 'alpha0': 2.0, 'beta0': 1.0}
HYP_BB = {'alpha0': np.array([0.05, 0.05, 0.05, 0.05]), 'beta0': np.array([0.5, 0.5, 0.5, 0.5])}
HYP_CAT = {'rho_c': 0.75, 'rho_i': 0.25, 'beta0': np.array([0.25, 0.25, 0.25, 0.25])}
HYP_CJCRP = {'gamma': 0.1, 'alpha_o': 0.5, 'alpha_r': 0.5}
HYP_JCRP = {'gamma': 0.1, 'alpha': 0.5}

AGENTS = {
    "CjCRP_NIW_BB": {"type_context": MC.CjCRP, "type_obs": M.ConjugateGaussianInvWish, "type_rew": M.ConjugateBernoulli,
                     "hyp_context": HYP_CJCRP, "hyp_obs": HYP_NIW, "hyp_rew": HYP_BB},
    "CjCRP_NIG_CAT": {"type_context": MC.CjCRP, "type_obs": M.ConjugateGaussianInvGam, "type_rew": M.ConjugateOptimalArm,
                      "hyp_context": HYP_CJCRP, "hyp_obs": HYP_NIG, "hyp_rew": HYP_CAT},
    "jCRP_NIW_BB": {"type_context": MC.jCRP, "type_obs": M.ConjugateGaussianInvWish, "type_rew": M.ConjugateBernoulli,
                    "hyp_context": HYP_JCRP, "hyp_obs": HYP_NIW, "hyp_rew": HYP_BB},
}


# ----------------------------------------
#   utils_sample
# ----------------------------------------

def test_cat_sample_frequencies():
    rng = pf_cpp.Rng(0)
    probs = np.array([0.2, 0.0, 0.5, 0.3]) * 7.0  # unnormalised, with a zero entry
    n = 40000
    counts = np.bincount([pf_cpp.cat_sample(probs, rng) for _ in range(n)], minlength=4)
    assert counts[1] == 0
    assert_close(counts / n, probs / probs.sum(), rtol=0, atol=0.01)


def test_cat_sample_n_frequencies():
    rng = pf_cpp.Rng(1)
    probs = np.array([0.0, 0.1, 0.6, 0.0, 0.3])
    samples = np.array(pf_cpp.cat_sample_n(probs, 100000, rng))
    counts = np.bincount(samples, minlength=5)
    assert counts[0] == 0 and counts[3] == 0
    assert_close(counts / len(samples), probs, rtol=0, atol=0.01)


def test_cat2D_sample_frequencies():
    """Indices (i, j) must index probs, as in Python's Categorical2D"""
    rng = pf_cpp.Rng(2)
    probs = np.array([[0.1, 0.0, 0.2], [0.3, 0.25, 0.15]])
    n = 40000
    counts = np.zeros_like(probs)
    for _ in range(n):
        i, j = pf_cpp.cat2D_sample(probs, rng)
        counts[i, j] += 1
    assert counts[0, 1] == 0
    assert_close(counts / n, probs, rtol=0, atol=0.01)


@pytest.mark.parametrize("alpha, beta", [(0.05, 0.5), (0.5, 0.05), (3.0, 7.0)])
def test_beta_sample_moments(alpha, beta):
    rng = pf_cpp.Rng(3)
    n = 50000
    samples = pf_cpp.beta_sample(np.full(n, alpha), np.full(n, beta), rng)
    assert np.all((samples >= 0) & (samples <= 1))
    mean = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    assert abs(samples.mean() - mean) < 5 * np.sqrt(var / n)
    assert abs(samples.var() - var) < 0.05 * var


# ----------------------------------------
#   models_obs_rew
# ----------------------------------------

@pytest.mark.parametrize("py_cls, hyp", [
    (M.ConjugateGaussianInvWish, HYP_NIW),
    (M.ConjugateGaussianInvWish, HYP_NIW_3D),
    (M.ConjugateGaussianInvGam, HYP_NIG),
])
def test_obs_model(py_cls, hyp):
    """Posterior and predictive parameters, and predictive likelihoods, after each of a sequence of updates"""
    gen = np.random.default_rng(0)
    d = hyp["mu0"].shape[0]
    py, cpp = py_cls(**hyp), cpp_model(py_cls, hyp)

    xs = 0.5 * gen.standard_normal((30, d)) + 1.0
    probes = gen.standard_normal((5, d))
    for t in range(len(xs) + 1):
        assert_dicts_close(cpp.post_params(), py.post_params())
        assert_dicts_close(cpp._pred_dist_params(), py._pred_dist_params())
        for x in probes:
            assert_close(cpp.pred_lh(to_cpp_value(x)), py.pred_lh(x))

        if t < len(xs):
            confidence = 0.5 if t % 3 == 0 else 1.0
            py.update(xs[t], confidence=confidence)
            cpp.update(to_cpp_value(xs[t]), confidence=confidence)


@pytest.mark.parametrize("py_cls, hyp", [(M.ConjugateBernoulli, HYP_BB), (M.ConjugateOptimalArm, HYP_CAT)])
def test_rew_model(py_cls, hyp):
    """Posterior and predictive parameters, and predictive likelihoods, after each of a sequence of updates"""
    gen = np.random.default_rng(1)
    py, cpp = py_cls(**hyp), cpp_model(py_cls, hyp)

    for t in range(40):
        assert_dicts_close(cpp.post_params(), py.post_params())
        assert_dicts_close(cpp._pred_dist_params(), py._pred_dist_params())
        for a in range(4):
            for r in (0.0, 1.0):
                assert_close(cpp.pred_lh(a, r), py.pred_lh(a, r))

        a = int(gen.integers(4))
        r = float(gen.random() < (0.8 if a == 2 else 0.2))
        py.update(a, r)
        cpp.update(a, r)


@pytest.mark.parametrize("py_cls, hyp", [(M.ConjugateBernoulli, HYP_BB), (M.ConjugateOptimalArm, HYP_CAT)])
@pytest.mark.parametrize("n_updates", [0, 10])
def test_rew_model_thompson_sampling(py_cls, hyp, n_updates):
    """Distribution of the action selected by Thompson sampling from the posterior, which is how the particle filter uses sample_post_dist"""
    utils_sample.seed(4)
    rng = pf_cpp.Rng(4)
    py, cpp = py_cls(**hyp), cpp_model(py_cls, hyp)
    for t in range(n_updates):
        a, r = t % 4, float(t % 4 == 2)
        py.update(a, r)
        cpp.update(a, r)

    n = 20000
    py_best = np.bincount([int(np.argmax(py.sample_post_dist())) for _ in range(n)], minlength=4)
    cpp_best = np.bincount([int(np.argmax(cpp.sample_post_dist(rng))) for _ in range(n)], minlength=4)
    assert_close(cpp_best / n, py_best / n, rtol=0, atol=0.02)


def test_obs_model_validation():
    with pytest.raises(ValueError):
        pf_cpp.ConjugateGaussianInvWish(mu0=np.zeros(2), kappa0=0.1, nu0=4.0, Lambda0=-np.eye(2))
    with pytest.raises(ValueError):
        pf_cpp.ConjugateGaussianInvGam(mu0=np.zeros(2), kappa0=0.1, alpha0=0.0, beta0=1.0)
    with pytest.raises(ValueError):
        pf_cpp.ConjugateOptimalArm(rho_c=0.75, rho_i=0.25, beta0=np.array([0.5, 0.6]))


# ----------------------------------------
#   models_context
# ----------------------------------------

def test_crp():
    gen = np.random.default_rng(5)
    py, cpp = MC.CRP(hyp_alpha=0.5), pf_cpp.CRP(hyp_alpha=0.5)
    for _ in range(50):
        assert cpp.n_active_contexts == py.n_active_contexts
        assert_close(cpp.counts, py.counts[:py.n_active_contexts])
        assert_close(cpp.probs, py.probs)
        c = int(gen.integers(py.n_active_contexts + 1))
        py.update(c)
        cpp.update(c)


@pytest.mark.parametrize("py_cls, hyp", [(MC.jCRP, HYP_JCRP), (MC.CjCRP, HYP_CJCRP)])
def test_context_model(py_cls, hyp):
    """Context hypotheses and their prior probabilities after each of a sequence of updates"""
    gen = np.random.default_rng(6)
    py, cpp = py_cls(**hyp), getattr(pf_cpp, py_cls.__name__)(**hyp)
    for _ in range(60):
        assert cpp.context_hypotheses == py.context_hypotheses
        assert_close(cpp.hypothesis_probs, py.hypothesis_probs)
        # Uniformly chosen hypotheses, so that jumps (and new contexts) are frequent
        c_t = int(gen.integers(py.n_hypotheses))
        py.update(c_t)
        cpp.update(c_t)


# ----------------------------------------
#   particle
# ----------------------------------------

OBS_LOCS = {(0, 0): [1.0, 0.0], (0, 1): [-1.0, 0.0], (1, 0): [0.0, 1.0], (1, 1): [0.0, -1.0]}


def observation(t: int, gen: np.random.Generator):
    """Observation and optimal action at trial t, with the observation context switching every 50 trials"""
    c = (t // 50) % 2
    s = int(gen.integers(2))
    o_t = np.array(OBS_LOCS[(c, s)]) + 0.14 * gen.standard_normal(2)
    return o_t, 2 * c + s


@pytest.mark.parametrize("agent", AGENTS.values(), ids=AGENTS.keys())
def test_particle_lockstep(agent):
    """
    Run a Python and a C++ particle in lockstep, with the C++ particle updated with the latent variables sampled by the Python particle.
    Beliefs, likelihoods, posteriors, and log weights must agree at every step.
    """
    # Seed chosen so that every agent's particle discovers a second context, see the end of the test
    utils_sample.seed(8)
    gen = np.random.default_rng(8)
    py, cpp = Particle(agent), pf_cpp.Particle(to_cpp_params(agent))

    for t in range(150):
        o_t, a_opt = observation(t, gen)
        o = to_cpp_value(o_t)

        assert_close(cpp.obs_likelihood(o), py.obs_likelihood(o_t))

        py.before_action(o_t)
        cpp.before_action(o)
        assert_close(cpp.temp_belief, py.temp_belief)
        assert_close(cpp.log_weight, float(py.log_weight))

        a_t = py.sample_action()
        r_t = float(a_t == a_opt)
        py_rew_lh = py.rew_likelihood(a_t, r_t)
        assert_close(cpp.rew_likelihood(a_t, r_t), py_rew_lh)
        expected_posterior = py_rew_lh * py.temp_belief / np.sum(py_rew_lh * py.temp_belief)

        hypotheses = [dict(h) for h in py.context_model.context_hypotheses]
        s_t, c_o_t, c_r_t, j_t = py.after_action(o_t, a_t, r_t)
        c_t = hypotheses.index({"jump": j_t, "obs_model": c_o_t, "rew_model": c_r_t})

        assert_close(cpp.latent_posterior(a_t, r_t), expected_posterior)
        assert cpp.update_params(o, a_t, r_t, s_t, c_t) == (s_t, c_o_t, c_r_t, j_t)
        assert_close(cpp.log_weight, float(py.log_weight))
        assert cpp.n_obs_models == py.n_obs_models
        assert cpp.n_rew_models == py.n_rew_models
        assert cpp.context_model.context_hypotheses == py.context_model.context_hypotheses

    # The run should have exercised more than one context (a single particle does not always discover the second one, e.g. with NIG observation models)
    assert py.n_obs_models >= 2


@pytest.mark.parametrize("agent", AGENTS.values(), ids=AGENTS.keys())
def test_particle_sample_action(agent):
    """Distribution of the actions sampled by a particle, after running Python and C++ particles in lockstep"""
    utils_sample.seed(8)
    gen = np.random.default_rng(8)
    py, cpp = Particle(agent), pf_cpp.Particle(to_cpp_params(agent))
    for t in range(60):
        o_t, a_opt = observation(t, gen)
        py.before_action(o_t)
        cpp.before_action(to_cpp_value(o_t))
        a_t = py.sample_action()
        r_t = float(a_t == a_opt)
        hypotheses = [dict(h) for h in py.context_model.context_hypotheses]
        s_t, c_o_t, c_r_t, j_t = py.after_action(o_t, a_t, r_t)
        cpp.latent_posterior(a_t, r_t)
        cpp.update_params(to_cpp_value(o_t), a_t, r_t, s_t, hypotheses.index({"jump": j_t, "obs_model": c_o_t, "rew_model": c_r_t}))

    o_t, _ = observation(60, gen)
    py.before_action(o_t)
    cpp.before_action(to_cpp_value(o_t))
    rng = pf_cpp.Rng(8)
    n = 5000
    py_actions = np.bincount([py.sample_action() for _ in range(n)], minlength=4)
    cpp_actions = np.bincount([cpp.sample_action(rng) for _ in range(n)], minlength=4)
    assert_close(cpp_actions / n, py_actions / n, rtol=0, atol=0.04)


def test_particle_copy_is_independent():
    agent = AGENTS["CjCRP_NIW_BB"]
    rng = pf_cpp.Rng(9)
    particle = pf_cpp.Particle(to_cpp_params(agent))
    o = np.array([1.0, 0.0])
    for _ in range(5):
        particle.before_action(o)
        particle.after_action(o, 0, 1.0, rng)
    copy = particle.copy()

    before = particle.obs_model(0, 0).post_params()["kappa_n"] + particle.obs_model(1, 0).post_params()["kappa_n"]
    for _ in range(5):
        copy.before_action(o)
        copy.after_action(o, 0, 1.0, rng)
    after = particle.obs_model(0, 0).post_params()["kappa_n"] + particle.obs_model(1, 0).post_params()["kappa_n"]
    assert after == before
    assert copy.context_model.n_hypotheses >= 1


# ----------------------------------------
#   idealobserver
# ----------------------------------------

def run_cpp_pf(agent, N=50, T=120, seed=10):
    utils_sample.seed(seed)
    gen = np.random.default_rng(seed)
    pf = pf_cpp.IdealObsPF(N, to_cpp_params(agent), seed)
    for t in range(T):
        o_t, a_opt = observation(t, gen)
        o = to_cpp_value(o_t)
        pf.before_action(o)
        a_t = pf.select_action()
        pf.after_action(o, a_t, float(a_t == a_opt))
    return pf


def test_ideal_observer_summaries():
    """p_action, p_state and p_jump agree with the tracked particle variables and weights, as computed in Python's IdealObsPF"""
    pf = run_cpp_pf(AGENTS["CjCRP_NIW_BB"])
    w = pf.weights
    assert_close(w.sum(), 1.0, atol=1e-12)
    assert_close(pf.p_action, np.round(np.bincount(pf.last_actions, weights=w, minlength=4), 2), atol=1e-12)
    assert_close(pf.p_state, np.round(np.bincount(pf.vec_s_t, weights=w, minlength=2), 2), atol=1e-12)
    assert_close(pf.p_jump, np.round(w[pf.vec_j_t == 1].sum(), 2), atol=1e-12)


def test_ideal_observer_action_log_probs():
    """action_log_probs estimates the weighted action probabilities from which select_action samples, as in Python's IdealObsPF"""
    pf = run_cpp_pf(AGENTS["CjCRP_NIW_BB"])
    o_t, _ = observation(120, np.random.default_rng(12))
    pf.before_action(to_cpp_value(o_t))

    n = 2000
    w = pf.weights
    expected = np.zeros(4)
    for _ in range(n):
        pf.select_action()
        expected += np.bincount(pf.last_actions, weights=w, minlength=4)
    expected /= n

    p_actions = np.exp(pf.action_log_probs(n))
    assert_close(p_actions.sum(), 1.0, atol=1e-9)
    assert_close(p_actions, expected, rtol=0, atol=0.01)


def test_ideal_observer_resample():
    """After resampling, weights are reset and every particle is a copy of a particle from before resampling"""
    pf = run_cpp_pf(AGENTS["CjCRP_NIW_BB"])

    def fingerprint(particle):
        return (str(particle.context_model.context_hypotheses), particle.n_obs_models,
                tuple(np.round(particle.obs_model(0, 0).post_params()["mu_n"], 12)))

    before = {fingerprint(pf.particle(i)) for i in range(pf.N)}
    pf.resample()
    assert np.all(pf.log_weights == 0.0)
    assert all(pf.particle(i).log_weight == 0.0 for i in range(pf.N))
    assert all(fingerprint(pf.particle(i)) in before for i in range(pf.N))


def test_ideal_observer_learns():
    """The ensemble learns the optimal action of the first context"""
    pf = run_cpp_pf(AGENTS["CjCRP_NIW_BB"], N=100, T=50)
    o_t, a_opt = observation(49, np.random.default_rng(11))
    pf.before_action(to_cpp_value(o_t))
    pf.select_action()
    assert pf.p_action[a_opt] > 0.8
