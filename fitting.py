"""
Fitting the hyperparameters of the ideal observer to a participant's choices, by pseudo-marginal Metropolis-Hastings.

The data are the observations o_{1:T} shown to the participant, the actions a_{1:T} they took, and the rewards r_{1:T} they received.
The participant is modelled as the ideal observer seeing o_t without sensory noise, which with a lapse rate ε chooses an action uniformly at random.

Likelihood
    The ideal observer is stochastic (particles sample latent states and contexts, and are resampled at random), so its internal state z_t is latent:

        p(a_{1:T} | o, r, θ) = 𝔼_z[ Π_t p(a_t | z_t, θ) ],     p(a_t | z_t, θ) = (1 - ε) P_IO(a_t | z_t) + ε / n_actions

    where z evolves given the participant's own o_t, a_t, r_t. One run of the ideal observer draws one trajectory z, so the product along a run is
    an unbiased estimate of the likelihood, as is its mean across independent runs. P_IO(a_t | z_t) is itself estimated without bias,
    from n_thompson Thompson samples per particle (action_log_probs).

Metropolis-Hastings
    As the likelihood is only estimated, the estimate for the current θ is kept and never recomputed (pseudo-marginal MH, Andrieu & Roberts 2009).
    The chain then targets the exact posterior over θ whatever the noise in the estimate, but noise slows mixing:
    the standard deviation of the log-likelihood estimate (log_likelihood_sd) should be below ~1.5.
    The noise has two sources: the Thompson samples from which P_IO(a_t | z_t) is estimated (reduced by n_thompson), which matter most on trials
    where the participant chose an action the ideal observer finds unlikely (e.g. exploration), and the ideal observer's own trajectory z
    (reduced by N). Both need to be large enough: increasing n_thompson does little while N is small.

    Parameters are sampled on an unconstrained scale (log or logit), with Normal priors on that scale.
    The random walk proposal is adapted during burn-in (shape to the covariance of the burn-in samples, scale to an acceptance rate of ~0.15),
    and fixed afterwards.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit, logsumexp

import utils_sample
from subject import IdealObsParams
from experiment import CombinedHistory, CPP_BACKEND_ERROR, make_subject


# Transforms of a hyperparameter to its unconstrained scale, and back
TRANSFORMS = {
    "log":   (np.log, np.exp),  # positive hyperparameters, e.g. CRP alpha
    "logit": (logit, expit),    # hyperparameters in (0, 1), e.g. jump probability gamma or the lapse rate
}

HYP_GROUPS = ("hyp_context", "hyp_obs", "hyp_rew")


@dataclass
class Dataset():
    """
    A participant's data, with one entry per trial
    """
    observations : np.ndarray # [T, d] observation shown to the participant
    actions      : np.ndarray # [T] action taken by the participant
    rewards      : np.ndarray # [T] reward received by the participant (0.0 or 1.0)

    @classmethod
    def from_history(cls, history: CombinedHistory) -> "Dataset":
        """
        Dataset of a simulated participant, e.g. for parameter recovery
        """
        exp_history = history.experiment_history
        return cls(observations=exp_history.observation, actions=exp_history.action.astype(int), rewards=exp_history.reward.astype(float))

    def __len__(self):
        return len(self.actions)


@dataclass(frozen=True)
class FreeParam():
    """
    A hyperparameter to fit, with a Normal prior on its unconstrained (transformed) scale.

    name:           "lapse", or "<group>.<key>" for the hyperparameter key of the group hyp_context, hyp_obs or hyp_rew (e.g. "hyp_context.gamma").
                    An array hyperparameter (e.g. "hyp_rew.alpha0" of ConjugateBernoulli) is set to the same value in all entries, which ties them.
    transform:      "log" for positive hyperparameters, "logit" for hyperparameters in (0, 1)
    prior_loc:      mean of the Normal prior, on the unconstrained scale
    prior_scale:    standard deviation of the Normal prior, on the unconstrained scale
    """
    name        : str
    transform   : str
    prior_loc   : float = 0.0
    prior_scale : float = 1.5


class FitModel():
    """
    The ideal observer with some of its hyperparameters free.
    Maps an unconstrained parameter vector θ (one entry per free hyperparameter, in order) to the hyperparameters of the ideal observer and the lapse rate.
    """
    def __init__(self, base_params: IdealObsParams, free_params: list[FreeParam], lapse: float = 0.01, n_actions: int = 4):
        """
        :param IdealObsParams base_params: parameters of the ideal observer, whose free hyperparameters are overridden by θ
        :param list[FreeParam] free_params: hyperparameters to fit, in the order of θ
        :param float lapse: lapse rate, if it is not a free parameter
        :param int n_actions: number of actions available to the participant
        """
        for param in free_params:
            assert param.transform in TRANSFORMS, f"transform of {param.name} must be one of {list(TRANSFORMS)}"
            if param.name != "lapse":
                group, _, key = param.name.partition(".")
                assert group in HYP_GROUPS, f"{param.name} must be 'lapse' or of the form '<group>.<key>', with group one of {HYP_GROUPS}"
                assert key in base_params[group], f"{key} is not a hyperparameter in {group} of base_params"
        assert len({param.name for param in free_params}) == len(free_params), "free parameters must be unique"
        assert 0.0 <= lapse < 1.0, "lapse must be in [0, 1)"

        self.base_params = base_params
        self.free_params = list(free_params)
        self.lapse = lapse
        self.n_actions = n_actions

        self.prior_loc   = np.array([param.prior_loc for param in self.free_params])
        self.prior_scale = np.array([param.prior_scale for param in self.free_params])

    @property
    def names(self) -> list[str]:
        return [param.name for param in self.free_params]

    @property
    def d(self) -> int:
        """
        Number of free parameters
        """
        return len(self.free_params)

    def to_natural(self, theta: np.ndarray) -> np.ndarray:
        """
        Free hyperparameters on their natural scale, from θ on the unconstrained scale
        """
        return np.array([TRANSFORMS[param.transform][1](x) for param, x in zip(self.free_params, theta)])

    def to_unconstrained(self, values: dict[str, float]) -> np.ndarray:
        """
        θ on the unconstrained scale, from the free hyperparameters on their natural scale, given as {name: value}
        """
        return np.array([TRANSFORMS[param.transform][0](values[param.name]) for param in self.free_params])

    def agent_params(self, theta: np.ndarray) -> tuple[IdealObsParams, float]:
        """
        Parameters of the ideal observer and lapse rate at θ
        """
        params = {**self.base_params, **{group: dict(self.base_params[group]) for group in HYP_GROUPS}} # copies, so base_params is not modified
        lapse = self.lapse

        for param, value in zip(self.free_params, self.to_natural(theta)):
            if param.name == "lapse":
                lapse = float(value)
                continue
            group, _, key = param.name.partition(".")
            base = self.base_params[group][key]
            # The models expect numpy arrays for array hyperparameters, and Python floats for the others
            params[group][key] = np.full_like(base, value, dtype=float) if isinstance(base, np.ndarray) else float(value)

        return params, lapse

    def log_prior(self, theta: np.ndarray) -> float:
        """
        Log density of the Normal prior at θ, on the unconstrained scale (up to a constant)
        """
        z = (np.asarray(theta) - self.prior_loc) / self.prior_scale
        return float(-0.5 * np.sum(z**2))

    def sample_prior(self, n: int, seed: int | None = None) -> np.ndarray:
        """
        n draws of θ from the prior, e.g. as dispersed starting points of chains [n, d]
        """
        rng = np.random.default_rng(seed)
        return self.prior_loc + self.prior_scale * rng.standard_normal((n, self.d))


# ----------------------------------------
#   Likelihood
# ----------------------------------------

def run_log_likelihood(
    agent_params: IdealObsParams,
    lapse: float,
    data: Dataset,
    N: int,
    n_thompson: int = 10,
    n_actions: int = 4,
    backend: str = "cpp"
    ) -> float:
    """
    Log of the product of the participant's choice probabilities along one run of the ideal observer,
    in which the ideal observer processes the participant's observations, actions and rewards (rather than acting itself).

    :param int n_thompson: number of Thompson samples per particle from which the ideal observer's action probabilities are estimated
    """
    subject = make_subject(agent_params, N, backend)

    with np.errstate(divide='ignore'): # with lapse 0, the lapse term has log probability -inf
        log_keep, log_lapse = np.log1p(-lapse), np.log(lapse / n_actions)

    ll = 0.0
    for o_t, a_t, r_t in zip(data.observations, data.actions, data.rewards):
        subject.before_action(o_t)

        # p(a_t) = (1 - ε) P_IO(a_t) + ε / n_actions
        ll += np.logaddexp(log_keep + subject.action_log_probs(n_thompson)[a_t], log_lapse)

        subject.after_action(o_t, int(a_t), float(r_t))

    return float(ll)


def log_likelihood(model: FitModel, theta: np.ndarray, data: Dataset, N: int, n_thompson: int = 10, n_runs: int = 1, backend: str = "cpp") -> float:
    """
    Estimate of log p(a_{1:T} | o_{1:T}, r_{1:T}, θ): the log of the mean likelihood across n_runs independent runs of the ideal observer.
    (The likelihood estimate is unbiased, its log is not, which is what pseudo-marginal MH needs.)
    """
    agent_params, lapse = model.agent_params(theta)
    run_lls = [run_log_likelihood(agent_params, lapse, data, N, n_thompson, model.n_actions, backend) for _ in range(n_runs)]
    return float(logsumexp(run_lls) - np.log(n_runs))


def log_likelihood_sd(
    model: FitModel,
    theta: np.ndarray,
    data: Dataset,
    N: int,
    n_thompson: int = 10,
    n_runs: int = 1,
    n_repeats: int = 20,
    backend: str = "cpp"
    ) -> float:
    """
    Standard deviation of the log-likelihood estimate at θ, across n_repeats estimates.
    Metropolis-Hastings mixes well if it is below ~1.5 near the posterior mode, otherwise increase N or n_thompson.
    The estimates have heavy tails at small N, so use enough repeats (30 or more) before trusting the standard deviation.
    """
    return float(np.std([log_likelihood(model, theta, data, N, n_thompson, n_runs, backend) for _ in range(n_repeats)]))


# ----------------------------------------
#   Metropolis-Hastings
# ----------------------------------------

@dataclass
class Chain():
    """
    Samples of a Metropolis-Hastings chain, after burn-in
    """
    names        : list[str]
    theta        : np.ndarray # [n_iter, d] samples on the unconstrained scale
    natural      : np.ndarray # [n_iter, d] samples on the natural scale
    log_lik      : np.ndarray # [n_iter] log-likelihood estimate of each sample
    accept_rate  : float      # acceptance rate after burn-in
    proposal_cov : np.ndarray # [d, d] covariance of the random walk proposal, as adapted during burn-in

    @property
    def samples(self) -> dict[str, np.ndarray]:
        """
        Samples of each free hyperparameter on its natural scale, as {name: [n_iter]}
        """
        return {name: self.natural[:, i] for i, name in enumerate(self.names)}


def run_chain(
    model: FitModel,
    data: Dataset,
    theta0: np.ndarray,
    n_iter: int,
    n_burnin: int,
    N: int,
    n_thompson: int = 10,
    n_runs: int = 1,
    backend: str = "cpp",
    seed: int | None = None,
    init_step: float = 0.1,
    target_accept: float = 0.15,
    adapt_every: int = 100,
    print_every: int = 0,
    ) -> Chain:
    """
    Run a pseudo-marginal Metropolis-Hastings chain over θ, see the module docstring.

    :param np.ndarray theta0: initial θ, on the unconstrained scale (see FitModel.to_unconstrained and FitModel.sample_prior)
    :param int n_iter: number of iterations after burn-in, which are returned
    :param int n_burnin: number of burn-in iterations, during which the proposal is adapted
    :param int N: number of particles of the ideal observer
    :param int n_thompson: number of Thompson samples per particle from which the ideal observer's action probabilities are estimated
    :param int n_runs: number of runs of the ideal observer averaged in each likelihood estimate
    :param int seed: seed for the proposals and for the ideal observer (utils_sample.rng). If None, they are seeded from fresh OS entropy.
    :param float init_step: standard deviation of the initial proposal, in each dimension of θ
    :param float target_accept: acceptance rate the proposal scale is adapted to (optimal values are ~0.1-0.2 for noisy likelihoods)
    :param int adapt_every: number of iterations between updates of the proposal shape during burn-in
    :param int print_every: if > 0, print progress every print_every iterations
    """
    # Independent random streams for the proposals and the ideal observer
    # (always reseeded, as copies of utils_sample.rng in worker processes would otherwise all draw the same random numbers)
    mh_seed, observer_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(mh_seed)
    utils_sample.seed(observer_seed)

    d = model.d
    theta = np.asarray(theta0, dtype=float)
    lp = model.log_prior(theta)
    ll = log_likelihood(model, theta, data, N, n_thompson, n_runs, backend)

    # Proposal covariance is scale^2 * shape
    shape = np.eye(d)
    log_scale = np.log(init_step)

    n_total = n_burnin + n_iter
    thetas = np.empty((n_total, d))
    log_liks = np.empty(n_total)
    accepted = np.zeros(n_total, dtype=bool)

    for it in range(n_total):
        prop = rng.multivariate_normal(theta, np.exp(2 * log_scale) * shape)
        lp_prop = model.log_prior(prop)
        ll_prop = log_likelihood(model, prop, data, N, n_thompson, n_runs, backend)

        # Pseudo-marginal acceptance: ll of the current θ is the stored estimate, never recomputed
        log_ratio = (ll_prop + lp_prop) - (ll + lp)
        if np.log(rng.random()) < log_ratio:
            theta, lp, ll = prop, lp_prop, ll_prop
            accepted[it] = True

        thetas[it] = theta
        log_liks[it] = ll

        # Adapt the proposal during burn-in only
        if it < n_burnin:
            # Scale: Robbins-Monro towards the target acceptance rate
            accept_prob = np.exp(min(0.0, log_ratio)) if np.isfinite(log_ratio) else 0.0
            log_scale += (accept_prob - target_accept) / (it + 1) ** 0.6

            # Shape: covariance of the second half of burn-in so far, once it has enough distinct samples,
            # with the scale reset to the optimal scaling 2.38 / sqrt(d) of a Gaussian target (Roberts & Rosenthal 2001)
            window = slice((it + 1) // 2, it + 1)
            if (it + 1) % adapt_every == 0 and accepted[window].sum() > 2 * d:
                shape = np.atleast_2d(np.cov(thetas[window].T)) + 1e-8 * np.eye(d)
                log_scale = np.log(2.38 / np.sqrt(d))

        if print_every > 0 and (it + 1) % print_every == 0:
            phase = "burn-in" if it < n_burnin else "sampling"
            recent = accepted[max(0, it + 1 - print_every): it + 1].mean()
            print(f"[{phase}] iteration {it + 1}/{n_total}, log-lik {ll:.2f}, acceptance (last {print_every}) {recent:.2f}, θ {np.round(model.to_natural(theta), 3)}")

    kept = slice(n_burnin, n_total)
    return Chain(
        names=model.names,
        theta=thetas[kept],
        natural=np.array([model.to_natural(t) for t in thetas[kept]]).reshape(n_iter, d),
        log_lik=log_liks[kept],
        accept_rate=float(accepted[kept].mean()) if n_iter > 0 else float("nan"),
        proposal_cov=np.exp(2 * log_scale) * shape,
    )


def run_chains(
    model: FitModel,
    data: Dataset,
    theta0s: np.ndarray,
    n_iter: int,
    n_burnin: int,
    N: int,
    n_workers: int,
    n_thompson: int = 10,
    n_runs: int = 1,
    backend: str = "cpp",
    seed: int | None = None,
    **chain_kwargs
    ) -> list[Chain]:
    """
    Run independent chains in parallel, with one chain per worker process at a time.

    :param np.ndarray theta0s: initial θ of each chain [n_chains, d], on the unconstrained scale
    :param int n_workers: number of worker processes
    :param int seed: if given, chain i is seeded with seed + i (reproducible), otherwise each chain is randomly seeded
    :param chain_kwargs: further arguments of run_chain (e.g. target_accept)
    :return list[Chain]: chains in the order of theta0s
    """
    assert 1 <= n_workers <= mp.cpu_count(), "Number of workers is limited to the number of hardware threads"
    if backend == "cpp" and CPP_BACKEND_ERROR is not None: # fail before starting the workers
        raise RuntimeError(CPP_BACKEND_ERROR)

    seeds = [None if seed is None else seed + i for i in range(len(theta0s))]

    # 'spawn' rather than 'fork', as forking a multithreaded process (e.g. a notebook kernel) can deadlock
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [
            pool.submit(run_chain, model, data, theta0, n_iter, n_burnin, N, n_thompson=n_thompson, n_runs=n_runs, backend=backend, seed=s, **chain_kwargs)
            for theta0, s in zip(theta0s, seeds)
            ]
        return [f.result() for f in futures]


def rhat(chains: list[Chain]) -> dict[str, float]:
    """
    Split R-hat (Gelman et al. 2013) of each free hyperparameter across chains, on the unconstrained scale.
    Values below ~1.01 indicate that the chains have converged to the same distribution.
    """
    n = min(len(chain.theta) for chain in chains) // 2
    halves = np.stack([half for chain in chains for half in (chain.theta[:n], chain.theta[n:2 * n])]) # [2 * n_chains, n, d]

    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n

    return {name: float(r) for name, r in zip(chains[0].names, np.sqrt(var_plus / within))}
