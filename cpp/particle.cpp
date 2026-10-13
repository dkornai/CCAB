#include "particle.hpp"

#include <cmath>

namespace pf {

IdealObsParams::IdealObsParams(const ContextModel& context, const ConjugateObsModel& obs, const ConjugateRewModel& rew)
    : context(context.clone()), obs(obs.clone()), rew(rew.clone()) {}


Particle::Particle(const IdealObsParams& particle_params)
    : params(particle_params),
      context_model(particle_params.context->clone()),
      state_probs(Eigen::Vector2d(0.5, 0.5)),
      n_states(2),
      obs_models(2),
      rew_models(2),
      // Differs from Python: the novel models are copies of the prior prototypes, see create_new_context_models
      novel_obs_model(particle_params.obs->clone()),
      novel_rew_model(particle_params.rew->clone()) {}

Particle::Particle(const Particle& other)
    : params(other.params),
      context_model(other.context_model->clone()),
      state_space(other.state_space),
      state_probs(other.state_probs),
      n_states(other.n_states),
      obs_models(other.obs_models.size()),
      rew_models(other.rew_models.size()),
      novel_obs_model(other.novel_obs_model->clone()),
      novel_rew_model(other.novel_rew_model->clone()),
      log_weight(other.log_weight),
      temp_belief(other.temp_belief) {
    for (int l = 0; l < n_states; ++l) {
        for (const auto& model : other.obs_models[l]) {
            obs_models[l].push_back(model->clone());
        }
        for (const auto& model : other.rew_models[l]) {
            rew_models[l].push_back(model->clone());
        }
    }
}


void Particle::before_action(const VectorXd& o_t) {
    // Calculate observation likelihoods for each state and context hypothesis
    const MatrixXd obs_lh = obs_likelihood(o_t);  // [n_states, n_hypotheses]

    // Compute the unnormalised belief
    //     P̄^(i)( s, c | o_t, {ξ, υ, ω}^(i)_{t-1} )
    //         ∝ P_S( s ) L_O^(i)( o_t | s, c, ω^(i)_{t-1} ) P_C^(i)( C_t=c | ξ^(i)_{t-1} )
    const MatrixXd unnorm_belief = state_probs.asDiagonal() * obs_lh * context_model->hypothesis_probs.asDiagonal();

    // Calculate the evidence for the observation
    const double Z_O = unnorm_belief.sum();

    // Normalize the belief to get the posterior probabilities
    temp_belief = unnorm_belief / (Z_O + 1e-30);

    // Adjust the particle's log weight based on the evidence
    log_weight += std::log(Z_O + 1e-30);
}

int Particle::sample_action(Rng& rng) const {
    // Sample state and context
    //     s_hat, c_hat ~ P̄^(i)( S_t=s, C_t=c | o_t, {ξ, υ, ω}^(i)_{t-1} )
    const auto [s_hat, c_hat] = cat2D_sample(temp_belief, rng);
    // select the reward model associated with the sampled context
    const int c_r_hat = context_model->hypothesis_r(c_hat);

    // Sample reward parameters for (s_hat, c_r_hat)
    //     υ_hat ~ P(ϒ | s_hat, c_r_hat)
    VectorXd predicted_rewards;
    if (c_r_hat == n_rew_models()) {  // equality corresponds to new model since models are indexed from 0 to n-1
        predicted_rewards = novel_rew_model->sample_post_dist(rng);  // special case for novel model
    } else {
        predicted_rewards = rew_models[s_hat][c_r_hat]->sample_post_dist(rng);
    }

    // Select best action
    //     a = argmax_a 𝔼[r_t | a, s_hat, c_r_hat, υ_hat]
    Eigen::Index a_t;
    predicted_rewards.maxCoeff(&a_t);

    return static_cast<int>(a_t);
}

std::tuple<int, int, int, int> Particle::after_action(const VectorXd& o_t, int a_t, double r_t, Rng& rng) {
    // ----------------------------------------------
    //   Calculate posterior over latent variables
    // ----------------------------------------------
    const MatrixXd posterior = latent_posterior(a_t, r_t);

    // Sample state and contexts from the posterior
    //     s_t, c_o_t ~ P(s, c | {o, a, r}_t, {ξ, υ, ω}^(i)_{t-1} )
    const auto [s_t, c_t] = cat2D_sample(posterior, rng);

    // ---------------------
    //   Update parameters
    // ---------------------
    return update_params(o_t, a_t, r_t, s_t, c_t);
}

MatrixXd Particle::latent_posterior(int a_t, double r_t) {
    // Calculate reward likelihoods for each state and context hypothesis
    const MatrixXd rew_lh = rew_likelihood(a_t, r_t);  // [n_states, n_hypotheses]

    // compute the unnormalised posterior
    //     P(s, c | {o, a, r}_t, {ξ, υ, ω}^(i)_{t-1} )
    //         ∝ L_R^(i)( r_t | s, c, a_t, υ^(i)_{t-1} ) P̄^(i)( s, c | o_t, {ξ, υ, ω}^(i)_{t-1} )
    const MatrixXd unnorm_post = rew_lh.cwiseProduct(temp_belief);  // [n_states, n_hypotheses]

    // Compute the evidence for the reward
    const double Z_R = unnorm_post.sum();

    // adjust the particle's log weight based on the evidence for the reward
    log_weight += std::log(Z_R + 1e-30);

    return unnorm_post / (Z_R + 1e-30);
}

std::tuple<int, int, int, int> Particle::update_params(const VectorXd& o_t, int a_t, double r_t, int s_t, int c_t) {
    const int j_t = context_model->hypothesis_jump(c_t);  // jump indicator for the sampled context
    const int c_o_t = context_model->hypothesis_o(c_t);
    const int c_r_t = context_model->hypothesis_r(c_t);

    // 1) Create new likelihood models if neeeded
    create_new_context_models(c_o_t, c_r_t);

    // 2) Update observation model sufficient statistics for the sampled context and state
    obs_models[s_t][c_o_t]->update(o_t);

    // 3) Update reward model sufficient statistics for the sampled context and state
    rew_models[s_t][c_r_t]->update(a_t, r_t);

    // 4) Update context model sufficient statistics given the sampled context
    context_model->update(c_t);

    return {s_t, c_o_t, c_r_t, j_t};
}

MatrixXd Particle::obs_likelihood(const VectorXd& o_t) const {
    // first calculate under each observation model for each state
    //     L_O^(i)( o_t | S=s, C=c^o , ω^(i)_{t-1} )
    MatrixXd per_model_likelihoods(n_states, n_obs_models() + 1);

    // Novel observation model, which is the same for all states
    // Differs from Python: evaluated once, where Python evaluates it for each state (same result)
    const double novel_lh = novel_obs_model->pred_lh(o_t);

    for (int l = 0; l < n_states; ++l) {
        // Existing observation models
        for (int m = 0; m < n_obs_models(); ++m) {
            per_model_likelihoods(l, m) = obs_models[l][m]->pred_lh(o_t);
        }
        // Novel observation model
        per_model_likelihoods(l, n_obs_models()) = novel_lh;
    }

    // then map the observation model likelihoods to the context hypotheses
    //     L_O^(i)( o_t | S=s, C=c, ω^(i)_{t-1} )
    // Differs from Python: columns are copied in a loop over hypotheses, where Python indexes with a tensor of model indices (same result)
    MatrixXd likelihoods(n_states, context_model->n_hypotheses());
    for (int c = 0; c < context_model->n_hypotheses(); ++c) {
        likelihoods.col(c) = per_model_likelihoods.col(context_model->hypothesis_o(c));
    }

    return likelihoods;  // [n_states, n_hypotheses]
}

MatrixXd Particle::rew_likelihood(int a_t, double r_t) const {
    // first calculate under each reward model for each state
    //     L_R^(i)( r_t | S=s, C=c^r, a_t, υ^(i)_{t-1} )
    MatrixXd per_model_likelihoods(n_states, n_rew_models() + 1);

    // Novel reward model, which is the same for all states
    // Differs from Python: evaluated once, where Python evaluates it for each state (same result)
    const double novel_lh = novel_rew_model->pred_lh(a_t, r_t);

    for (int l = 0; l < n_states; ++l) {
        // existing reward models
        for (int n = 0; n < n_rew_models(); ++n) {
            per_model_likelihoods(l, n) = rew_models[l][n]->pred_lh(a_t, r_t);
        }
        // novel reward model
        per_model_likelihoods(l, n_rew_models()) = novel_lh;
    }

    // then map the reward model likelihoods to the context hypotheses
    //     L_R^(i)( r_t | S=s, C=c, a_t, υ^(i)_{t-1} )
    // Differs from Python: columns are copied in a loop over hypotheses, where Python indexes with a tensor of model indices (same result)
    MatrixXd likelihoods(n_states, context_model->n_hypotheses());
    for (int c = 0; c < context_model->n_hypotheses(); ++c) {
        likelihoods.col(c) = per_model_likelihoods.col(context_model->hypothesis_r(c));
    }

    return likelihoods;  // [n_states, n_hypotheses]
}

void Particle::create_new_context_models(int c_o_t, int c_r_t) {
    // Differs from Python: new models are copies of the prior prototypes in params, where Python constructs them as type(**hyperparameters)

    // Create observation models for new context if needed
    if (c_o_t == n_obs_models()) {  // equality corresponds to new model since models are indexed from 0 to n-1
        for (int l = 0; l < n_states; ++l) {
            obs_models[l].push_back(params.obs->clone());
        }
    }

    // Create reward models for new context if needed
    if (c_r_t == n_rew_models()) {  // equality corresponds to new model since models are indexed from 0 to n-1
        for (int l = 0; l < n_states; ++l) {
            rew_models[l].push_back(params.rew->clone());
        }
    }
}

}  // namespace pf
