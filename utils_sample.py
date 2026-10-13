import numpy as np

"""
Random number generator used by the ideal observer (all samplers below draw from it).
"""
rng = np.random.default_rng()


def seed(s: int | np.random.SeedSequence | None = None):
    """
    Reseed the ideal observer's random number generator. With None, it is seeded from fresh OS entropy.
    The generator is reseeded in place, so modules that imported rng see the new state.

    :param int | np.random.SeedSequence | None s: seed
    """
    rng.bit_generator.state = np.random.default_rng(s).bit_generator.state


def normalise(probs: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """
    Normalise an array of probabilities to sum to 1, with numerical stability.

    :param np.ndarray probs: array of probabilities (not necessarily normalized)
    :param float eps: small constant to prevent division by zero
    :return np.ndarray: normalized probabilities
    """
    total = np.sum(probs) + eps
    return probs / total


def bern_sample(prob: float) -> int:
    """
    Sample from a Bernoulli distribution with given probability.

    :param float prob: probability of success (between 0 and 1)
    :return int: sampled value (0 or 1)
    """
    return int(rng.random() < prob)


def cat_sample(probs: np.ndarray) -> int:
    """
    Sample from a categorical distribution defined by the given probabilities.

    :param np.ndarray probs: array of probabilities (not necessarily normalized)
    :return int: sampled category index
    """
    # Inverse CDF: first index whose cumulative probability exceeds u (clipped in case of rounding in the cumulative sum)
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, rng.random() * cdf[-1], side='right'))
    return min(idx, len(cdf) - 1)


class Categorical2D():
    """
    Class for representing a 2D categorical distribution [e.g. a joint P(X,Y)].
    """
    def __init__(self, probs: np.ndarray):
        """
        :param np.ndarray probs: 2D array of probabilities (not necessarily normalized)
        """
        assert probs.ndim == 2, "probs must be a 2D array"
        self.dim1, self.dim2 = probs.shape
        self.probs2d = probs / np.sum(probs)
        self.probs = probs.flatten()

    def sample(self) -> tuple[int, int]:
        """
        Sample from the 2D categorical distribution and return the corresponding indices (i, j).
        """
        idx = cat_sample(self.probs)
        i = idx // self.dim2
        j = idx % self.dim2

        return int(i), int(j)


def cat2D_sample(probs: np.ndarray) -> tuple[int, int]:
    """
    Sample from a 2D categorical distribution defined by the given probabilities.

    :param np.ndarray probs: 2D array of probabilities
    :return tuple[int, int]: sampled indices (i, j)
    """
    return Categorical2D(probs).sample()


def wishart_sample(df: float, scale: np.ndarray) -> np.ndarray:
    """
    Sample from a Wishart distribution W(df, scale), using the Bartlett decomposition (as torch.distributions.Wishart does).

    :param float df: degrees of freedom > d - 1
    :param np.ndarray scale: scale matrix [d, d], positive definite
    :return np.ndarray: sampled matrix [d, d]
    """
    d = scale.shape[0]
    L = np.linalg.cholesky(scale)

    # Lower triangular A with A_ii = sqrt(chi2(df - i)) and A_ij ~ N(0, 1) below the diagonal
    A = np.tril(rng.standard_normal((d, d)), k=-1)
    A[np.diag_indices(d)] = np.sqrt(rng.chisquare(df - np.arange(d)))

    LA = L @ A
    return LA @ LA.T
