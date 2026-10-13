#include "utils_sample.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pf {

namespace {

/// Uniform sample in (0, 1)
double uniform_open(Rng& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double u;
    do {
        u = unif(rng);
    } while (u == 0.0);
    return u;
}

/**
 * Log of a Gamma(shape, 1) sample.
 * Computed in log space, since for small shapes (e.g. the Beta(0.05, 0.5) reward prior) samples can underflow to 0.
 */
double log_gamma_sample(double shape, Rng& rng) {
    if (shape < 1.0) {
        // Gamma(shape) = Gamma(shape + 1) * U^(1 / shape)
        return log_gamma_sample(shape + 1.0, rng) + std::log(uniform_open(rng)) / shape;
    }
    std::gamma_distribution<double> gamma(shape, 1.0);
    return std::log(gamma(rng));
}

double checked_total(double total) {
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::invalid_argument("probabilities must have a positive, finite sum");
    }
    return total;
}

}  // namespace

int bern_sample(double prob, Rng& rng) {
    std::bernoulli_distribution bern(prob);
    return bern(rng) ? 1 : 0;
}

int cat_sample(const Eigen::VectorXd& probs, Rng& rng) {
    const double total = checked_total(probs.sum());
    std::uniform_real_distribution<double> unif(0.0, total);
    const double u = unif(rng);

    double cum = 0.0;
    int last_positive = -1;
    for (Eigen::Index i = 0; i < probs.size(); ++i) {
        if (probs[i] > 0.0) {
            cum += probs[i];
            last_positive = static_cast<int>(i);
            if (u < cum) {
                return last_positive;
            }
        }
    }
    // Only reached through rounding in the cumulative sum
    return last_positive;
}

std::vector<int> cat_sample_n(const Eigen::VectorXd& probs, int n, Rng& rng) {
    const double total = checked_total(probs.sum());

    // Cumulative probabilities, so that each sample is a binary search
    std::vector<double> cum(probs.size());
    double running = 0.0;
    int last_positive = -1;
    for (Eigen::Index i = 0; i < probs.size(); ++i) {
        running += probs[i];
        cum[i] = running;
        if (probs[i] > 0.0) {
            last_positive = static_cast<int>(i);
        }
    }

    std::uniform_real_distribution<double> unif(0.0, total);
    std::vector<int> samples(n);
    for (int k = 0; k < n; ++k) {
        // First index whose cumulative probability exceeds u, which skips zero-probability entries
        const auto it = std::upper_bound(cum.begin(), cum.end(), unif(rng));
        samples[k] = (it == cum.end()) ? last_positive : static_cast<int>(it - cum.begin());
    }
    return samples;
}

std::pair<int, int> cat2D_sample(const Eigen::MatrixXd& probs, Rng& rng) {
    const double total = checked_total(probs.sum());
    std::uniform_real_distribution<double> unif(0.0, total);
    const double u = unif(rng);

    // Traverse in row-major order, as Python's Categorical2D samples from the row-major flattened probs
    double cum = 0.0;
    std::pair<int, int> last_positive{-1, -1};
    for (Eigen::Index i = 0; i < probs.rows(); ++i) {
        for (Eigen::Index j = 0; j < probs.cols(); ++j) {
            if (probs(i, j) > 0.0) {
                cum += probs(i, j);
                last_positive = {static_cast<int>(i), static_cast<int>(j)};
                if (u < cum) {
                    return last_positive;
                }
            }
        }
    }
    // Only reached through rounding in the cumulative sum
    return last_positive;
}

Eigen::VectorXd beta_sample(const Eigen::VectorXd& alpha, const Eigen::VectorXd& beta, Rng& rng) {
    // If X ~ Gamma(alpha), Y ~ Gamma(beta), then X / (X + Y) = 1 / (1 + Y / X) ~ Beta(alpha, beta)
    Eigen::VectorXd sample(alpha.size());
    for (Eigen::Index k = 0; k < alpha.size(); ++k) {
        const double log_x = log_gamma_sample(alpha[k], rng);
        const double log_y = log_gamma_sample(beta[k], rng);
        sample[k] = 1.0 / (1.0 + std::exp(log_y - log_x));
    }
    return sample;
}

}  // namespace pf
