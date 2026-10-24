// C++ port of models_context.py
#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "utils_sample.hpp"

namespace pf {

/// Python: ContextHypothesis
struct ContextHypothesis {
    int jump;       ///< 0 or 1
    int obs_model;
    int rew_model;
};


/**
 * Python: ContextModel
 *
 * Subclasses must set the following members whenever the context hypotheses change:
 *     context_hypotheses: all currently possible context hypotheses c
 *     hypothesis_probs:   P_C(C_t = c | xi_{t-1}) for all currently possible c, shape [n_hypotheses]
 */
class ContextModel {
public:
    virtual ~ContextModel() = default;

    /// Independent copy of the model, including its sufficient statistics
    virtual std::unique_ptr<ContextModel> clone() const = 0;

    virtual void update(int c_t) = 0;

    int n_hypotheses() const { return static_cast<int>(context_hypotheses.size()); }

    /// Index of the observation model associated with context hypothesis c
    int hypothesis_o(int c) const { return context_hypotheses[c].obs_model; }

    /// Index of the reward model associated with context hypothesis c
    int hypothesis_r(int c) const { return context_hypotheses[c].rew_model; }

    /// Jump indicator (0 or 1) associated with context hypothesis c
    int hypothesis_jump(int c) const { return context_hypotheses[c].jump; }

    std::vector<ContextHypothesis> context_hypotheses;
    Eigen::VectorXd hypothesis_probs;
};


/**
 * Python: CRP
 * Differs from Python: Python preallocates counts for 100 contexts, here counts grows with the number of active contexts.
 */
class CRP {
public:
    explicit CRP(double hyp_alpha);

    /// Update the counts with a single context assignment
    /// Differs from Python: c beyond the next new context (c > n_active_contexts) throws, where Python would count it without activating it
    void update(int c);

    /// Sample from the CRP prior
    int sample(Rng& rng) const;

    double hyp_alpha;
    std::vector<double> counts;
    int n_active_contexts = 0;

    /// Prior probabilities for each category, including a new one
    Eigen::VectorXd probs;

private:
    void _update_probs();
};


/// Python: jCRP
class jCRP : public ContextModel {
public:
    jCRP(double gamma, double alpha);

    std::unique_ptr<ContextModel> clone() const override;
    void update(int c_t) override;

    /// Fill the context hypotheses and their prior probabilities based on the current state of the CRP
    void fill_context_hypotheses();

    double hyp_gamma;
    double hyp_alpha;

    /// Python: CRP. Differs from Python: renamed, as a member cannot share the name of its type (the bindings expose it as CRP)
    CRP crp;

    /// Due to the jump mixture, we need to keep track of the previous context
    int prev_c = 0;
};


/// Python: CjCRP. Observation and reward contexts have separate CRPs, which share the concentration parameter alpha
class CjCRP : public ContextModel {
public:
    CjCRP(double gamma, double alpha);

    std::unique_ptr<ContextModel> clone() const override;
    void update(int c_t) override;

    /// Fill the context hypotheses and their prior probabilities based on the current state of the CRPs
    void fill_context_hypotheses();

    double hyp_gamma;
    double hyp_alpha;

    CRP CRP_o;
    CRP CRP_r;

    /// Due to the jump mixture, we need to keep track of the previous contexts
    int prev_c_o = 0;
    int prev_c_r = 0;
};

}  // namespace pf
