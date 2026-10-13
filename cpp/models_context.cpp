#include "models_context.hpp"

#include <stdexcept>
#include <string>

namespace pf {

// ----------------------------------------
//   CRP
// ----------------------------------------

CRP::CRP(double hyp_alpha) : hyp_alpha(hyp_alpha) {
    _update_probs();
}

void CRP::_update_probs() {
    // Probability of each context (including a new one).
    // These only change when the counts do, so are computed here rather than each time they are needed.
    probs.resize(n_active_contexts + 1);
    for (int c = 0; c < n_active_contexts; ++c) {
        probs[c] = counts[c];
    }
    probs[n_active_contexts] = hyp_alpha;
    probs /= probs.sum();
}

void CRP::update(int c) {
    if (c < 0 || c > n_active_contexts) {
        throw std::invalid_argument("context " + std::to_string(c) + " out of range for " + std::to_string(n_active_contexts) + " active contexts");
    }

    if (c == n_active_contexts) {
        counts.push_back(0.0);
        n_active_contexts += 1;
    }
    counts[c] += 1;

    _update_probs();
}

int CRP::sample(Rng& rng) const {
    return cat_sample(probs, rng);
}


// ----------------------------------------
//   jCRP
// ----------------------------------------

jCRP::jCRP(double gamma, double alpha) : hyp_gamma(gamma), hyp_alpha(alpha), crp(alpha) {
    // Fill the list of context hypotheses based on the current state of the CRP
    fill_context_hypotheses();
}

std::unique_ptr<ContextModel> jCRP::clone() const {
    return std::make_unique<jCRP>(*this);
}

void jCRP::fill_context_hypotheses() {
    // A single context determines both the observation model and reward model.
    context_hypotheses.clear();

    // If there are no active contexts, we can only have a jump to a new context
    if (crp.n_active_contexts == 0) {
        context_hypotheses.push_back({1, 0, 0});
    }
    // Otherwise
    else {
        // J = 0: exactly one possible hypothesis
        context_hypotheses.push_back({0, prev_c, prev_c});

        // J = 1: one hypothesis for each possible CRP context
        for (int c = 0; c < crp.n_active_contexts + 1; ++c) {
            context_hypotheses.push_back({1, c, c});
        }
    }

    // Prior probabilities of the context hypotheses
    // If there is only one hypothesis, it must be the jump-to-new-context hypothesis, so its probability is 1
    if (context_hypotheses.size() == 1) {
        hypothesis_probs = Eigen::VectorXd::Ones(1);
    }
    // Otherwise, compute probabilities based on the CRP prior and the jump probability
    else {
        hypothesis_probs.resize(n_hypotheses());

        // Hypothesis 0 is always the unique stay hypothesis
        hypothesis_probs[0] = 1.0 - hyp_gamma;

        // Remaining hypotheses are jumps, one for each CRP context c in order
        // The same context determines both observation and reward models
        hypothesis_probs.tail(n_hypotheses() - 1) = hyp_gamma * crp.probs;
    }
}

void jCRP::update(int c_t) {
    // Extract the jump and shared context from the current hypothesis
    const ContextHypothesis hyp = context_hypotheses[c_t];

    // Sufficient statistics are only updated when a jump occurs
    if (hyp.jump == 1) {
        // Update the single shared context CRP
        crp.update(hyp.obs_model);

        // Update previous context
        prev_c = hyp.obs_model;

        fill_context_hypotheses();  // Refill the context hypotheses after the update
    }
}


// ----------------------------------------
//   CjCRP
// ----------------------------------------

CjCRP::CjCRP(double gamma, double alpha_o, double alpha_r)
    : hyp_gamma(gamma), hyp_alpha_o(alpha_o), hyp_alpha_r(alpha_r), CRP_o(alpha_o), CRP_r(alpha_r) {
    // Fill the list of context hypotheses based on the current state of the CRPs
    fill_context_hypotheses();
}

std::unique_ptr<ContextModel> CjCRP::clone() const {
    return std::make_unique<CjCRP>(*this);
}

void CjCRP::fill_context_hypotheses() {
    // Each hypothesis is a combination of a jump (0 or 1), an observation model, and a reward model.
    context_hypotheses.clear();

    // If there are no active contexts, we can only have a jump to a new context
    if (CRP_o.n_active_contexts == 0 && CRP_r.n_active_contexts == 0) {
        context_hypotheses.push_back({1, 0, 0});
    }
    // Otherwise
    else {
        // J = 0: exactly one possible hypothesis
        context_hypotheses.push_back({0, prev_c_o, prev_c_r});

        // J = 1: Cartesian product of CRP possibilities
        for (int c_o = 0; c_o < CRP_o.n_active_contexts + 1; ++c_o) {
            for (int c_r = 0; c_r < CRP_r.n_active_contexts + 1; ++c_r) {
                context_hypotheses.push_back({1, c_o, c_r});
            }
        }
    }

    // Prior probabilities of the context hypotheses
    // If there is only one hypothesis, it must be the jump-to-new-context hypothesis, so its probability is 1
    if (context_hypotheses.size() == 1) {
        hypothesis_probs = Eigen::VectorXd::Ones(1);
    }
    // Otherwise, we compute the probabilities for each hypothesis based on the CRP priors and the jump probability
    else {
        hypothesis_probs.resize(n_hypotheses());

        // Hypothesis 0 is always the unique stay hypothesis
        hypothesis_probs[0] = 1.0 - hyp_gamma;

        // Remaining hypotheses 1, 2, 3 ... are jumps, ordered as in the Cartesian product above (c_o outer, c_r inner)
        // Differs from Python: a loop over the Cartesian product, where Python flattens the outer product of the CRP probabilities (same values and order)
        int i = 1;
        for (int c_o = 0; c_o < CRP_o.n_active_contexts + 1; ++c_o) {
            for (int c_r = 0; c_r < CRP_r.n_active_contexts + 1; ++c_r) {
                hypothesis_probs[i++] = (hyp_gamma * CRP_o.probs[c_o]) * CRP_r.probs[c_r];
            }
        }
    }
}

void CjCRP::update(int c_t) {
    // Extract the jump, obs_model, and rew_model from the current context hypothesis
    const ContextHypothesis hyp = context_hypotheses[c_t];

    // Sufficient statistics are only updated when a jump occurs
    if (hyp.jump == 1) {
        // Update counts
        CRP_o.update(hyp.obs_model);
        CRP_r.update(hyp.rew_model);

        // Update previous context
        prev_c_o = hyp.obs_model;
        prev_c_r = hyp.rew_model;

        fill_context_hypotheses();  // Refill the context hypotheses after the update
    }
}

}  // namespace pf
