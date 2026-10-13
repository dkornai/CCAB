#include "models_obs_rew.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace pf {

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

/// Lower Cholesky factor of A. Python: torch.linalg.cholesky, which also raises if A is not positive definite
MatrixXd cholesky(const MatrixXd& A) {
    Eigen::LLT<MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("cholesky: matrix is not positive definite");
    }
    return llt.matrixL();
}

/// Probability of reward r under a Bernoulli with success probability p. Python: the body of the reward models' pred_lh
double bernoulli_lh(double p, double r) {
    if (r == 1.0) {
        return p;
    } else if (r == 0.0) {
        return 1.0 - p;
    }
    throw std::invalid_argument("reward r must be 0.0 or 1.0, got " + std::to_string(r));
}

}  // namespace


// ----------------------------------------
//   Observation models
// ----------------------------------------

SuffStatsGaussian::SuffStatsGaussian(int d)
    : sum_x(VectorXd::Zero(d)), sum_xx(MatrixXd::Zero(d, d)) {}

void SuffStatsGaussian::update(const VectorXd& x, double confidence) {
    n += confidence;
    sum_x += confidence * x;
    sum_xx += confidence * x * x.transpose();
}

VectorXd SuffStatsGaussian::mean() const {
    return sum_x / n;
}

MatrixXd SuffStatsGaussian::scatter() const {
    // S = Σ_i (x_i - x̄)(x_i - x̄)^T = Σ_i (x_i)(x_i)^T - n * (x̄)(x̄)^T
    const VectorXd mu = mean();
    return sum_xx - n * mu * mu.transpose();
}


ConjugateGaussianInvWish::ConjugateGaussianInvWish(const VectorXd& mu0, double kappa0, double nu0, const MatrixXd& Lambda0)
    : mu0(mu0), kappa0(kappa0), nu0(nu0), Lambda0(Lambda0), d(static_cast<int>(mu0.size())), suffstats(static_cast<int>(mu0.size())) {
    require(kappa0 > 0, "pseudocount of prior measurements kappa0 must be > 0");
    require(nu0 > d - 1, "degrees of freedom nu0 must be > " + std::to_string(d - 1));
    require(Lambda0.rows() == d && Lambda0.cols() == d, "scale matrix Lambda0 must be of shape (d,d)");
    require(Eigen::SelfAdjointEigenSolver<MatrixXd>(Lambda0).eigenvalues().minCoeff() > 0, "scale matrix Lambda0 must be positive definite");

    // Posterior params start at prior
    reset_posterior();
}

std::unique_ptr<ConjugateObsModel> ConjugateGaussianInvWish::clone() const {
    return std::make_unique<ConjugateGaussianInvWish>(*this);
}

void ConjugateGaussianInvWish::reset_posterior() {
    mu_n = mu0;
    kappa_n = kappa0;
    nu_n = nu0;
    Lambda_n = Lambda0;

    _update_pred_dist_params();
}

void ConjugateGaussianInvWish::update(const VectorXd& x, double confidence) {
    suffstats.update(x, confidence);
    _update_posterior();
}

void ConjugateGaussianInvWish::_update_posterior() {
    if (suffstats.n == 0) {  // do not update posterior if there are no datapoints
        return;
    }

    const VectorXd x_bar = suffstats.mean();
    const double n = suffstats.n;
    const MatrixXd S = suffstats.scatter();

    // κ_n = κ_0 + n
    kappa_n = kappa0 + n;

    // μ_n = ( (κ_0 / (κ_n)) * μ0 ) + ( (n / (κ_n)) * x̄ )
    mu_n = ((kappa0 / kappa_n) * mu0) + ((n / kappa_n) * x_bar);

    // ν_n = ν_0 + n
    nu_n = nu0 + n;

    // Λ_n = Λ_0 + S + ((κ_0 * n )/ (κ_n)) * (x̄ - μ0)(x̄ - μ0)^T
    const VectorXd diff = x_bar - mu0;  // difference between sample mean and prior mean
    Lambda_n = Lambda0 + S + ((kappa0 * n) / kappa_n) * (diff * diff.transpose());

    _update_pred_dist_params();
}

void ConjugateGaussianInvWish::_update_pred_dist_params() {
    df = nu_n - d + 1;
    scale = (Lambda_n * (kappa_n + 1)) / (kappa_n * df);
    scale_tril = cholesky(scale);

    // Log normalising constant of the multivariate Student-t, see mvt_log_prob
    norm_const = (
        scale_tril.diagonal().array().log().sum()
        + 0.5 * d * std::log(df)
        + 0.5 * d * std::log(std::numbers::pi)
        + std::lgamma(0.5 * df)
        - std::lgamma(0.5 * (df + d))
    );
}

double ConjugateGaussianInvWish::pred_lh(const VectorXd& x) const {
    return std::exp(mvt_log_prob(x, df, mu_n, scale_tril, norm_const));
}


ConjugateGaussianInvGam::ConjugateGaussianInvGam(const VectorXd& mu0, double kappa0, double alpha0, double beta0)
    : mu0(mu0), kappa0(kappa0), alpha0(alpha0), beta0(beta0), d(static_cast<int>(mu0.size())), suffstats(static_cast<int>(mu0.size())) {
    require(kappa0 > 0, "kappa0 must be > 0");
    require(alpha0 > 0, "alpha0 must be > 0");
    require(beta0 > 0, "beta0 must be > 0");

    // Initialize posterior to prior.
    reset_posterior();
}

std::unique_ptr<ConjugateObsModel> ConjugateGaussianInvGam::clone() const {
    return std::make_unique<ConjugateGaussianInvGam>(*this);
}

void ConjugateGaussianInvGam::reset_posterior() {
    mu_n = mu0;
    kappa_n = kappa0;
    alpha_n = alpha0;
    beta_n = beta0;

    _update_pred_dist_params();
}

void ConjugateGaussianInvGam::update(const VectorXd& x, double confidence) {
    suffstats.update(x, confidence);
    _update_posterior();
}

void ConjugateGaussianInvGam::_update_posterior() {
    if (suffstats.n == 0) {  // do not update posterior if there are no datapoints
        return;
    }

    const VectorXd x_bar = suffstats.mean();
    const double n = suffstats.n;
    const MatrixXd S = suffstats.scatter();
    const double sse = S.trace();  // sum of squared errors

    // κ_n = κ_0 + n
    kappa_n = kappa0 + n;

    // μ_n = (κ_0 * μ0 + n * x̄) / κ_n
    mu_n = ((kappa0 * mu0) + (n * x_bar)) / kappa_n;

    // α_n = α_0 + (n * d)/2
    alpha_n = alpha0 + (0.5 * n * d);

    // β_n = β_0 + 1/2 SSE + (κ_0 * n)/(2 κ_n) ||x̄ - μ0||^2
    const VectorXd diff = x_bar - mu0;  // difference between sample mean and prior mean
    beta_n = beta0 + (0.5 * sse) + ((kappa0 * n) / (2.0 * kappa_n) * diff.dot(diff));

    _update_pred_dist_params();
}

void ConjugateGaussianInvGam::_update_pred_dist_params() {
    df = 2.0 * alpha_n;
    scale = ((beta_n / alpha_n) * (kappa_n + 1.0) / kappa_n) * MatrixXd::Identity(d, d);
    scale_tril = cholesky(scale);

    // Log normalising constant of the multivariate Student-t, see mvt_log_prob
    norm_const = (
        scale_tril.diagonal().array().log().sum()
        + 0.5 * d * std::log(df)
        + 0.5 * d * std::log(std::numbers::pi)
        + std::lgamma(0.5 * df)
        - std::lgamma(0.5 * (df + d))
    );
}

double ConjugateGaussianInvGam::pred_lh(const VectorXd& x) const {
    return std::exp(mvt_log_prob(x, df, mu_n, scale_tril, norm_const));
}


// ----------------------------------------
//   Reward models
// ----------------------------------------

SuffStatsRewards::SuffStatsRewards(int K)
    : K(K), succ(VectorXd::Zero(K)), fail(VectorXd::Zero(K)) {}

void SuffStatsRewards::update(int a, double r, double confidence) {
    require(0 <= a && a < K, "action " + std::to_string(a) + " out of range for " + std::to_string(K) + " actions");
    require(r == 0.0 || r == 1.0, "reward r must be 0.0 or 1.0, got " + std::to_string(r));

    if (r == 1.0) {
        succ[a] += confidence;
    } else {
        fail[a] += confidence;
    }
}

double SuffStatsRewards::n() const {
    return (succ + fail).sum();
}


ConjugateBernoulli::ConjugateBernoulli(const VectorXd& alpha0, const VectorXd& beta0)
    : alpha0(alpha0), beta0(beta0), K(static_cast<int>(alpha0.size())), suffstats(static_cast<int>(alpha0.size())) {
    require(alpha0.size() == beta0.size(), "alpha0 and beta0 must have the same shape");
    require((alpha0.array() > 0).all(), "alpha0 must be > 0");
    require((beta0.array() > 0).all(), "beta0 must be > 0");

    reset_posterior();
}

std::unique_ptr<ConjugateRewModel> ConjugateBernoulli::clone() const {
    return std::make_unique<ConjugateBernoulli>(*this);
}

void ConjugateBernoulli::reset_posterior() {
    // Posterior parameters start at prior.
    alpha_n = alpha0;
    beta_n = beta0;
}

void ConjugateBernoulli::update(int a, double r, double confidence) {
    suffstats.update(a, r, confidence);
    _update_posterior();
}

void ConjugateBernoulli::_update_posterior() {
    // Beta posterior for each action:
    //     alpha_n[a] = alpha0[a] + succ[a]
    //     beta_n[a]  = beta0[a]  + fail[a]
    if (suffstats.n() == 0) {  // do not update posterior if there are no datapoints
        return;
    }
    alpha_n = alpha0 + suffstats.succ;
    beta_n = beta0 + suffstats.fail;
}

VectorXd ConjugateBernoulli::sample_post_dist(Rng& rng) const {
    // Sample of the reward probabilities for each action from the posterior Beta distribution
    // Differs from Python: sampled through log-Gamma variates, see beta_sample
    return beta_sample(alpha_n, beta_n, rng);
}

VectorXd ConjugateBernoulli::_pred_dist_params() const {
    return alpha_n.array() / (alpha_n + beta_n).array();
}

double ConjugateBernoulli::pred_lh(int a, double r) const {
    // P(r=1 | a) = alpha_n[a] / (alpha_n[a] + beta_n[a]), P(r=0 | a) = 1 - P(r=1 | a)
    // Differs from Python: computes p_rew for action a only, rather than for all actions via _pred_dist_params (same result)
    return bernoulli_lh(alpha_n[a] / (alpha_n[a] + beta_n[a]), r);
}


ConjugateOptimalArm::ConjugateOptimalArm(double rho_c, double rho_i, const VectorXd& beta0)
    : rho_c(rho_c), rho_i(rho_i), beta0(beta0), K(static_cast<int>(beta0.size())), suffstats(static_cast<int>(beta0.size())) {
    require(0.0 < rho_c && rho_c < 1.0, "rho_c must be in (0, 1), got " + std::to_string(rho_c));
    require(0.0 < rho_i && rho_i < 1.0, "rho_i must be in (0, 1), got " + std::to_string(rho_i));
    require(K >= 2, "beta0 must contain at least two arms");
    require((beta0.array() > 0).all(), "all beta0 entries must be > 0 to avoid degenerate posteriors");
    // Differs from Python: Python checks for exact equality, which float32 inputs that sum to 1 in float32 may not satisfy in float64
    require(std::abs(beta0.sum() - 1.0) < 1e-6, "beta0 must sum to 1.0");

    // These are constant across all posterior updates.
    log_rho_ratio = std::log(rho_c / rho_i);
    log_failure_ratio = std::log((1.0 - rho_c) / (1.0 - rho_i));
    log_beta0 = beta0.array().log();

    reset_posterior();
}

std::unique_ptr<ConjugateRewModel> ConjugateOptimalArm::clone() const {
    return std::make_unique<ConjugateOptimalArm>(*this);
}

void ConjugateOptimalArm::reset_posterior() {
    // Reset the posterior to the prior.
    beta_n = beta0;
}

void ConjugateOptimalArm::update(int a, double r, double confidence) {
    suffstats.update(a, r, confidence);
    _update_posterior();
}

void ConjugateOptimalArm::_update_posterior() {
    // log P(k | data) = log beta0[k] + succ[k] * log(rho_c / rho_i) + fail[k] * log((1-rho_c)/(1-rho_i)) + constant
    const VectorXd log_post = log_beta0 + (suffstats.succ * log_rho_ratio) + (suffstats.fail * log_failure_ratio);

    // softmax
    const VectorXd unnorm = (log_post.array() - log_post.maxCoeff()).exp();
    beta_n = unnorm / unnorm.sum();
}

VectorXd ConjugateOptimalArm::sample_post_dist(Rng& rng) const {
    // Posterior sample of the optimal arm identity, as a one-hot vector over the arms
    VectorXd arms = VectorXd::Zero(K);
    arms[cat_sample(beta_n, rng)] = 1.0;
    return arms;
}

VectorXd ConjugateOptimalArm::_pred_dist_params() const {
    return beta_n * rho_c + (VectorXd::Ones(K) - beta_n) * rho_i;
}

double ConjugateOptimalArm::pred_lh(int a, double r) const {
    // Differs from Python: computes p_rew for action a only, rather than for all actions via _pred_dist_params (same result)
    return bernoulli_lh(beta_n[a] * rho_c + (1.0 - beta_n[a]) * rho_i, r);
}


// ----------------------------------------
//   Utils
// ----------------------------------------

double mvt_log_prob(const VectorXd& value, double df, const VectorXd& loc, const MatrixXd& scale_tril, double norm_const) {
    const double n = static_cast<double>(loc.size());
    const VectorXd y = scale_tril.triangularView<Eigen::Lower>().solve(value - loc);
    return -0.5 * (df + n) * std::log1p(y.squaredNorm() / df) - norm_const;
}

}  // namespace pf
