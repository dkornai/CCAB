// C++ port of idealobserver.py
#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "particle.hpp"
#include "utils_sample.hpp"

namespace pf {

/// Python: IdealObsPF. Particle filtering implementation of the ideal observer.
class IdealObsPF {
public:
    /// The ensemble owns its random number generator, seeded with seed, which it passes to the particles.
    /// Differs from Python: Python uses torch's global generator, see utils_sample.hpp
    IdealObsPF(int N, const IdealObsParams& hyp_params, std::uint64_t seed);

    /// Weights of each particle, summing to 1
    VectorXd weights() const;

    /// Process the incoming observation o_t
    void before_action(const VectorXd& o_t);

    /// Select an action a_t
    int select_action();

    /// Process the resulting o_t, a_t, r_t emissions at time t
    void after_action(const VectorXd& o_t, int a_t, double r_t);

    /// Resample particles according to weights
    void resample();

    /// Weighted action probabilities at the current time step, inferred from the particles' last sampled actions (rounded to 2 decimals, as in Python)
    VectorXd p_action() const;

    /// Probabilities of latent states at the current time step, weighted by particle weights (rounded to 2 decimals, as in Python)
    VectorXd p_state() const;

    /// Probability of a jump at the current time step, weighted by particle weights (rounded to 2 decimals, as in Python)
    double p_jump() const;

    /// Probability of resampling after each trial
    static constexpr double resample_prob = 0.02;

    int N;
    IdealObsParams hyp_params;

    std::vector<Particle> particles;

    // Extra data structures that are tracked during particle filtering
    VectorXd log_weights;
    Eigen::VectorXi vec_s_t;
    Eigen::VectorXi vec_c_o_t;
    Eigen::VectorXi vec_c_r_t;
    Eigen::VectorXi vec_j_t;
    Eigen::VectorXi last_actions;

    Rng rng;
};

}  // namespace pf
