// C++ port of models_obs_rew.py
//
// Differences from Python that apply throughout the port are marked "Differs from Python:" here, as are those at specific places in the code.
//
// Differs from Python: invalid hyperparameters and arguments throw std::invalid_argument (ValueError in Python), where Python asserts.
//
// Differs from Python: the Python ConjugateModel base class is shared between observation and reward models.
// Here they get separate interfaces (ConjugateObsModel, ConjugateRewModel), as update and pred_lh take different arguments.
//
// Differs from Python: Python's post_params() and _pred_dist_params() return posterior and predictive parameters as dicts.
// Here they are public members, and the Python bindings provide post_params() and _pred_dist_params() returning the same dicts.
#pragma once

#include <memory>

#include <Eigen/Dense>

#include "utils_sample.hpp"

namespace pf {

using Eigen::MatrixXd;
using Eigen::VectorXd;


// ----------------------------------------
//   Observation models
// ----------------------------------------

/**
 * Python: ConjugateModel, for models of the observations o_t.
 * Differs from Python: sample_post_dist is not ported for observation models, as the particle filter never samples their posteriors.
 */
class ConjugateObsModel {
public:
    virtual ~ConjugateObsModel() = default;

    /// Independent copy of the model, including its sufficient statistics
    virtual std::unique_ptr<ConjugateObsModel> clone() const = 0;

    /// Add a new observation and update posterior parameters
    virtual void update(const VectorXd& x, double confidence = 1.0) = 0;

    /// Predictive likelihood of a new observation x
    virtual double pred_lh(const VectorXd& x) const = 0;
};


/// Python: SuffStatsGaussian
class SuffStatsGaussian {
public:
    explicit SuffStatsGaussian(int d);

    void update(const VectorXd& x, double confidence = 1.0);

    /// Mean of the observations.
    /// Differs from Python: Python returns None for n == 0, here it must only be called for n > 0.
    VectorXd mean() const;

    /// Scatter around the empirical mean.
    /// Differs from Python: Python returns None for n == 0, here it must only be called for n > 0.
    MatrixXd scatter() const;

    double   n = 0.0;
    VectorXd sum_x;
    MatrixXd sum_xx;
};


/// Python: ConjugateGaussianInvWish
class ConjugateGaussianInvWish : public ConjugateObsModel {
public:
    ConjugateGaussianInvWish(const VectorXd& mu0, double kappa0, double nu0, const MatrixXd& Lambda0);

    std::unique_ptr<ConjugateObsModel> clone() const override;
    void reset_posterior();
    void update(const VectorXd& x, double confidence = 1.0) override;
    double pred_lh(const VectorXd& x) const override;

    // Prior parameters
    VectorXd mu0;
    double   kappa0;
    double   nu0;
    MatrixXd Lambda0;
    int      d;

    SuffStatsGaussian suffstats;

    // Posterior parameters (Python: post_params)
    VectorXd mu_n;
    double   kappa_n;
    double   nu_n;
    MatrixXd Lambda_n;

    // Parameters of the multivariate Student-t predictive distribution (Python: _pred_dist_params, where loc is mu_n)
    double   df;
    MatrixXd scale;
    MatrixXd scale_tril;
    double   norm_const;

private:
    void _update_posterior();
    void _update_pred_dist_params();
};


/// Python: ConjugateGaussianInvGam
class ConjugateGaussianInvGam : public ConjugateObsModel {
public:
    ConjugateGaussianInvGam(const VectorXd& mu0, double kappa0, double alpha0, double beta0);

    std::unique_ptr<ConjugateObsModel> clone() const override;
    void reset_posterior();
    void update(const VectorXd& x, double confidence = 1.0) override;
    double pred_lh(const VectorXd& x) const override;

    // Prior parameters
    VectorXd mu0;
    double   kappa0;
    double   alpha0;
    double   beta0;
    int      d;

    SuffStatsGaussian suffstats;

    // Posterior parameters (Python: post_params)
    VectorXd mu_n;
    double   kappa_n;
    double   alpha_n;
    double   beta_n;

    // Parameters of the multivariate Student-t predictive distribution (Python: _pred_dist_params, where loc is mu_n)
    double   df;
    MatrixXd scale;
    MatrixXd scale_tril;
    double   norm_const;

private:
    void _update_posterior();
    void _update_pred_dist_params();
};


// ----------------------------------------
//   Reward models
// ----------------------------------------

/// Python: ConjugateModel, for models of the rewards r_t given actions a_t
class ConjugateRewModel {
public:
    virtual ~ConjugateRewModel() = default;

    /// Independent copy of the model, including its sufficient statistics
    virtual std::unique_ptr<ConjugateRewModel> clone() const = 0;

    /// Add a new observation (a, r) and update posterior parameters
    virtual void update(int a, double r, double confidence = 1.0) = 0;

    /// Predictive likelihood of observing reward r given action a
    virtual double pred_lh(int a, double r) const = 0;

    /// Sample of the (expected) reward of each action from the posterior
    virtual VectorXd sample_post_dist(Rng& rng) const = 0;
};


/// Python: SuffStatsRewards
class SuffStatsRewards {
public:
    explicit SuffStatsRewards(int K);

    void update(int a, double r, double confidence = 1.0);

    /// Total (soft) number of observations across all actions
    double n() const;

    int      K;
    VectorXd succ;
    VectorXd fail;
};


/// Python: ConjugateBernoulli
class ConjugateBernoulli : public ConjugateRewModel {
public:
    ConjugateBernoulli(const VectorXd& alpha0, const VectorXd& beta0);

    std::unique_ptr<ConjugateRewModel> clone() const override;
    void reset_posterior();
    void update(int a, double r, double confidence = 1.0) override;
    double pred_lh(int a, double r) const override;
    VectorXd sample_post_dist(Rng& rng) const override;

    /// Python: _pred_dist_params()["p_rew"]
    VectorXd _pred_dist_params() const;

    // Prior parameters
    VectorXd alpha0;
    VectorXd beta0;
    int      K;

    SuffStatsRewards suffstats;

    // Posterior parameters (Python: post_params)
    VectorXd alpha_n;
    VectorXd beta_n;

private:
    void _update_posterior();
};


/// Python: ConjugateOptimalArm
class ConjugateOptimalArm : public ConjugateRewModel {
public:
    ConjugateOptimalArm(double rho_c, double rho_i, const VectorXd& beta0);

    std::unique_ptr<ConjugateRewModel> clone() const override;
    void reset_posterior();
    void update(int a, double r, double confidence = 1.0) override;
    double pred_lh(int a, double r) const override;
    VectorXd sample_post_dist(Rng& rng) const override;

    /// Python: _pred_dist_params()["p_rew"]
    VectorXd _pred_dist_params() const;

    // Prior parameters
    double   rho_c;
    double   rho_i;
    VectorXd beta0;
    int      K;

    SuffStatsRewards suffstats;

    // Constant across all posterior updates
    double   log_rho_ratio;
    double   log_failure_ratio;
    VectorXd log_beta0;

    // Posterior parameters (Python: post_params)
    VectorXd beta_n;

private:
    void _update_posterior();
};


// ----------------------------------------
//   Utils
// ----------------------------------------

/// Python: mvt_log_prob
/// Differs from Python: for a single value and distribution, where Python broadcasts over batches.
/// Uses a triangular solve as pyro does, where Python uses the general np.linalg.solve (same solution).
double mvt_log_prob(const VectorXd& value, double df, const VectorXd& loc, const MatrixXd& scale_tril, double norm_const);

}  // namespace pf
