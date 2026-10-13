// C++ port of particle.py
#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "models_context.hpp"
#include "models_obs_rew.hpp"
#include "utils_sample.hpp"

namespace pf {

/**
 * Python: subject.IdealObsParams
 *
 * Differs from Python: Python gives the model types and their hyperparameters separately, and constructs new models as type(**hyperparameters).
 * Here each model is given as a prototype in its prior state, and new models are copies of the prototype.
 */
struct IdealObsParams {
    /// Stores independent copies of the prototypes, which must be in their prior state (i.e. not updated)
    IdealObsParams(const ContextModel& context, const ConjugateObsModel& obs, const ConjugateRewModel& rew);

    std::shared_ptr<const ContextModel>      context;
    std::shared_ptr<const ConjugateObsModel> obs;
    std::shared_ptr<const ConjugateRewModel> rew;
};


/// Python: Particle
class Particle {
public:
    explicit Particle(const IdealObsParams& particle_params);

    /// Independent copy of the particle, with copies of all its models (Python: deepcopy during resampling)
    Particle(const Particle& other);
    Particle& operator=(const Particle& other) = delete;
    Particle(Particle&& other) noexcept = default;
    Particle& operator=(Particle&& other) noexcept = default;

    /**
     * Compute the temporary belief P̄ given the newest observation o_t, and the previous sufficient statistics of the particle.
     *     P̄^(i)( S_t=s, C_t=c | o_t, {ξ, υ, ω}^(i)_{t-1} )
     */
    void before_action(const VectorXd& o_t);

    /// Thompson sampling using the temporary belief induced by o_t
    int sample_action(Rng& rng) const;

    /**
     * Infer the latent state and context, and update the sufficient statistics.
     * Returns (s_t, c_o_t, c_r_t, j_t).
     *
     * Differs from Python: Python does this in one method, here it is split into latent_posterior and update_params,
     * so that tests can drive the updates with the latent variables sampled by the Python particle.
     */
    std::tuple<int, int, int, int> after_action(const VectorXd& o_t, int a_t, double r_t, Rng& rng);

    /**
     * First half of after_action: posterior over the latent state and context hypotheses given the reward,
     *     P(s, c | {o, a, r}_t, {ξ, υ, ω}^(i)_{t-1} )
     * Also adjusts the log weight by the evidence for the reward.
     */
    MatrixXd latent_posterior(int a_t, double r_t);

    /// Second half of after_action: update parameters given the sampled state s_t and context hypothesis c_t. Returns (s_t, c_o_t, c_r_t, j_t).
    std::tuple<int, int, int, int> update_params(const VectorXd& o_t, int a_t, double r_t, int s_t, int c_t);

    /// L_O^(i)( o_t | S=s, C=c, ω^(i)_{t-1} ), shape (n_states, n_hypotheses)
    MatrixXd obs_likelihood(const VectorXd& o_t) const;

    /// L_R^(i)( r_t | S=s, C=c, a_t, υ^(i)_{t-1} ), shape (n_states, n_hypotheses)
    MatrixXd rew_likelihood(int a_t, double r_t) const;

    /// Number of observation models for each state (i.e. the number of observation contexts)
    int n_obs_models() const { return static_cast<int>(obs_models[0].size()); }

    /// Number of reward models for each state (i.e. the number of reward contexts)
    int n_rew_models() const { return static_cast<int>(rew_models[0].size()); }

    /// Create likelihood models for the new observation and reward contexts if needed
    void create_new_context_models(int c_o_t, int c_r_t);

    IdealObsParams params;

    std::unique_ptr<ContextModel> context_model;

    // Latent State Space
    std::vector<int> state_space{0, 1};
    VectorXd state_probs;
    int n_states;

    // Models for observations and rewards, organized by state and context (this starts empty and is populated as new contexts are discovered)
    std::vector<std::vector<std::unique_ptr<ConjugateObsModel>>> obs_models;
    std::vector<std::vector<std::unique_ptr<ConjugateRewModel>>> rew_models;

    // Novel context models (used for computing likelihood of new contexts)
    std::unique_ptr<ConjugateObsModel> novel_obs_model;
    std::unique_ptr<ConjugateRewModel> novel_rew_model;

    // Particle weight for resampling
    // Differs from Python: float64, where Python accumulates it as a float32 tensor
    double log_weight = 0.0;

    // Temporary belief computed in before_action, shape (n_states, n_hypotheses)
    MatrixXd temp_belief;

    // Differs from Python: Python also stores the likelihoods _obs_lh and _rew_lh as attributes, here they are local variables, as nothing reads them later
};

}  // namespace pf
