#include "idealobserver.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace pf {

namespace {

/// Python: torch.round(x, decimals=2)
/// Differs from Python: rounds a float64 value, where Python rounds the float32 conversion
double round2(double x) {
    return std::nearbyint(x * 100.0) / 100.0;
}

}  // namespace


IdealObsPF::IdealObsPF(int N, const IdealObsParams& hyp_params, std::uint64_t seed)
    : N(N),
      hyp_params(hyp_params),
      log_weights(VectorXd::Zero(N)),
      vec_s_t(Eigen::VectorXi::Zero(N)),
      vec_c_o_t(Eigen::VectorXi::Zero(N)),
      vec_c_r_t(Eigen::VectorXi::Zero(N)),
      vec_j_t(Eigen::VectorXi::Zero(N)),
      last_actions(Eigen::VectorXi::Zero(N)),
      rng(seed) {
    // Initialise the particles
    particles.reserve(N);
    for (int i = 0; i < N; ++i) {
        particles.emplace_back(hyp_params);
    }
}

VectorXd IdealObsPF::weights() const {
    // exp(log_weights - logsumexp(log_weights))
    const double max_lw = log_weights.maxCoeff();
    const double logsumexp = max_lw + std::log((log_weights.array() - max_lw).exp().sum());
    return (log_weights.array() - logsumexp).exp();
}

void IdealObsPF::before_action(const VectorXd& o_t) {
    for (Particle& particle : particles) {
        particle.before_action(o_t);
    }
}

int IdealObsPF::select_action() {
    // Each particle samples an action
    // Differs from Python: Python keeps the actions as _last_actions, here last_actions
    for (int i = 0; i < N; ++i) {
        last_actions[i] = particles[i].sample_action(rng);
    }

    // Pick a particle from which the action is emitted
    const int selected_particle = cat_sample(weights(), rng);
    return last_actions[selected_particle];
}

void IdealObsPF::after_action(const VectorXd& o_t, int a_t, double r_t) {
    // Each particle conducts inference on the latent state and contexts, and updates internal parameters
    for (int i = 0; i < N; ++i) {
        const auto [s_t, c_o_t, c_r_t, j_t] = particles[i].after_action(o_t, a_t, r_t, rng);
        vec_s_t[i] = s_t;
        vec_c_o_t[i] = c_o_t;
        vec_c_r_t[i] = c_r_t;
        vec_j_t[i] = j_t;
        log_weights[i] = particles[i].log_weight;
    }

    // Resample particles with a low probability to maintain diversity
    if (bern_sample(resample_prob, rng) == 1) {
        resample();
    }
}

void IdealObsPF::resample() {
    // sample N ancestor indices using current log weights (Python: Categorical(logits=log_weights))
    const VectorXd unnorm_weights = (log_weights.array() - log_weights.maxCoeff()).exp();
    const std::vector<int> ancestors = cat_sample_n(unnorm_weights, N, rng);

    // Number of descendants of each ancestor
    std::vector<int> remaining(N, 0);
    for (int anc : ancestors) {
        remaining[anc] += 1;
    }

    // Descendants get copies of their ancestor, except the last one, which takes over the ancestor itself
    // Differs from Python: in Python the first descendant takes over the ancestor. Here it is the last, so that all copies are made before the ancestor is moved.
    std::vector<Particle> new_particles;
    new_particles.reserve(N);
    for (int anc : ancestors) {
        remaining[anc] -= 1;
        if (remaining[anc] > 0) {
            new_particles.push_back(particles[anc]);
        } else {
            new_particles.push_back(std::move(particles[anc]));
        }
    }
    particles = std::move(new_particles);

    // Reset weights after resampling
    for (Particle& particle : particles) {
        particle.log_weight = 0.0;
    }
    log_weights.setZero();
}

VectorXd IdealObsPF::p_action() const {
    const VectorXd w = weights();  // [N], sums to 1

    // Python: bincount(last_actions, weights=w, minlength=4)
    // Differs from Python: returned as float64, where Python returns float32
    VectorXd p_actions = VectorXd::Zero(std::max(4, last_actions.maxCoeff() + 1));
    for (int i = 0; i < N; ++i) {
        p_actions[last_actions[i]] += w[i];
    }
    return p_actions.unaryExpr([](double x) { return round2(x); });
}

VectorXd IdealObsPF::p_state() const {
    const VectorXd w = weights();  // [N], sums to 1

    // inferred latent state distribution P(s_t). Python: bincount(vec_s_t, weights=w, minlength=2)
    VectorXd p_s = VectorXd::Zero(std::max(2, vec_s_t.maxCoeff() + 1));
    for (int i = 0; i < N; ++i) {
        p_s[vec_s_t[i]] += w[i];
    }
    return p_s.unaryExpr([](double x) { return round2(x); });
}

double IdealObsPF::p_jump() const {
    const VectorXd w = weights();  // [N], sums to 1

    // inferred jump probability P(j_t == 1)
    double p = 0.0;
    for (int i = 0; i < N; ++i) {
        if (vec_j_t[i] == 1) {
            p += w[i];
        }
    }
    return round2(p);
}

}  // namespace pf
