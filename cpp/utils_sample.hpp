// C++ port of utils_sample.py
//
// Differs from Python: normalise is not ported, as it is unused. beta_sample and cat_sample_n are added, for what Python does with torch.distributions.
#pragma once

#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace pf {

/**
 * Random number generator used throughout the port, passed explicitly to everything that samples.
 * Differs from Python: Python draws from torch's global generator, so the port reproduces the distributions of samples, not the samples themselves.
 */
using Rng = std::mt19937_64;

/// Python: bern_sample
int bern_sample(double prob, Rng& rng);

/// Python: cat_sample. probs need not be normalised.
/// Differs from Python: invalid probabilities (non-positive or non-finite sum) throw std::invalid_argument, as for all samplers here.
int cat_sample(const Eigen::VectorXd& probs, Rng& rng);

/// n independent samples from the categorical distribution defined by probs (not necessarily normalised).
/// Python: D.Categorical(probs).sample((n,)), as used in IdealObsPF.resample
std::vector<int> cat_sample_n(const Eigen::VectorXd& probs, int n, Rng& rng);

/// Python: cat2D_sample. Returns indices (i, j), probs need not be normalised.
std::pair<int, int> cat2D_sample(const Eigen::MatrixXd& probs, Rng& rng);

/// Elementwise Beta(alpha, beta) sample. Python: D.Beta(alpha, beta).sample()
/// Differs from Python: sampled as X / (X + Y) from Gamma variates X, Y computed in log space, so that small shapes
/// (e.g. the Beta(0.05, 0.5) reward prior) cannot underflow to 0 / 0. Torch samples Beta through a Dirichlet.
Eigen::VectorXd beta_sample(const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta, Rng& rng);

}  // namespace pf
