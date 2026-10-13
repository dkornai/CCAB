// Python bindings of the C++ port, as the module pf_cpp.
// Classes and their constructor arguments have the same names as in the Python implementation,
// and post_params() / _pred_dist_params() return the same dicts, so that components can be compared directly.
//
// Differs from Python: arrays and sub-objects (e.g. suffstats, CRP_o) are returned as copies, so modifying them does not change the C++ object.
// Hyperparameters and observations are numpy float64 arrays, rather than torch tensors (idealobserver_cpp.to_cpp_value converts them).
#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "idealobserver.hpp"
#include "models_context.hpp"
#include "models_obs_rew.hpp"
#include "particle.hpp"
#include "utils_sample.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/// Getter returning a copy of a member, so that Python never holds a view into C++ memory
template <typename T, typename M>
auto copy_of(M T::*member) {
    return [member](const T& self) { return M(self.*member); };
}

nb::dict mvt_params(double df, const Eigen::VectorXd& loc, const Eigen::MatrixXd& scale, const Eigen::MatrixXd& scale_tril, double norm_const) {
    nb::dict d;
    d["df"] = df;
    d["loc"] = Eigen::VectorXd(loc);
    d["scale"] = Eigen::MatrixXd(scale);
    d["scale_tril"] = Eigen::MatrixXd(scale_tril);
    d["norm_const"] = norm_const;
    return d;
}

}  // namespace


NB_MODULE(pf_cpp, m) {
    m.doc() = "C++ port of the particle filtering ideal observer";

    // ----------------------------------------
    //   utils_sample
    // ----------------------------------------
    nb::class_<pf::Rng>(m, "Rng")
        .def(nb::init<std::uint64_t>(), "seed"_a);

    m.def("bern_sample", &pf::bern_sample, "prob"_a, "rng"_a);
    m.def("cat_sample", &pf::cat_sample, "probs"_a, "rng"_a);
    m.def("cat_sample_n", &pf::cat_sample_n, "probs"_a, "n"_a, "rng"_a);
    m.def("cat2D_sample", &pf::cat2D_sample, "probs"_a, "rng"_a);
    m.def("beta_sample", &pf::beta_sample, "alpha"_a, "beta"_a, "rng"_a);

    // ----------------------------------------
    //   models_obs_rew
    // ----------------------------------------
    m.def("mvt_log_prob", &pf::mvt_log_prob, "value"_a, "df"_a, "loc"_a, "scale_tril"_a, "norm_const"_a);

    nb::class_<pf::SuffStatsGaussian>(m, "SuffStatsGaussian")
        .def(nb::init<int>(), "d"_a)
        .def("update", &pf::SuffStatsGaussian::update, "x"_a, "confidence"_a = 1.0)
        .def_ro("n", &pf::SuffStatsGaussian::n)
        .def_prop_ro("sum_x", copy_of(&pf::SuffStatsGaussian::sum_x))
        .def_prop_ro("sum_xx", copy_of(&pf::SuffStatsGaussian::sum_xx));

    nb::class_<pf::ConjugateObsModel>(m, "ConjugateObsModel")
        .def("update", &pf::ConjugateObsModel::update, "x"_a, "confidence"_a = 1.0)
        .def("pred_lh", &pf::ConjugateObsModel::pred_lh, "x"_a);

    using NIW = pf::ConjugateGaussianInvWish;
    nb::class_<NIW, pf::ConjugateObsModel>(m, "ConjugateGaussianInvWish")
        .def(nb::init<const Eigen::VectorXd&, double, double, const Eigen::MatrixXd&>(), "mu0"_a, "kappa0"_a, "nu0"_a, "Lambda0"_a)
        .def("reset_posterior", &NIW::reset_posterior)
        .def_prop_ro("suffstats", copy_of(&NIW::suffstats))
        .def_ro("d", &NIW::d)
        .def("post_params", [](const NIW& s) {
            nb::dict d;
            d["mu_n"] = Eigen::VectorXd(s.mu_n);
            d["kappa_n"] = s.kappa_n;
            d["nu_n"] = s.nu_n;
            d["Lambda_n"] = Eigen::MatrixXd(s.Lambda_n);
            return d;
        })
        .def("_pred_dist_params", [](const NIW& s) { return mvt_params(s.df, s.mu_n, s.scale, s.scale_tril, s.norm_const); });

    using NIG = pf::ConjugateGaussianInvGam;
    nb::class_<NIG, pf::ConjugateObsModel>(m, "ConjugateGaussianInvGam")
        .def(nb::init<const Eigen::VectorXd&, double, double, double>(), "mu0"_a, "kappa0"_a, "alpha0"_a, "beta0"_a)
        .def("reset_posterior", &NIG::reset_posterior)
        .def_prop_ro("suffstats", copy_of(&NIG::suffstats))
        .def_ro("d", &NIG::d)
        .def("post_params", [](const NIG& s) {
            nb::dict d;
            d["mu_n"] = Eigen::VectorXd(s.mu_n);
            d["kappa_n"] = s.kappa_n;
            d["alpha_n"] = s.alpha_n;
            d["beta_n"] = s.beta_n;
            return d;
        })
        .def("_pred_dist_params", [](const NIG& s) { return mvt_params(s.df, s.mu_n, s.scale, s.scale_tril, s.norm_const); });

    nb::class_<pf::SuffStatsRewards>(m, "SuffStatsRewards")
        .def(nb::init<int>(), "K"_a)
        .def("update", &pf::SuffStatsRewards::update, "a"_a, "r"_a, "confidence"_a = 1.0)
        .def_prop_ro("n", &pf::SuffStatsRewards::n)
        .def_ro("K", &pf::SuffStatsRewards::K)
        .def_prop_ro("succ", copy_of(&pf::SuffStatsRewards::succ))
        .def_prop_ro("fail", copy_of(&pf::SuffStatsRewards::fail));

    nb::class_<pf::ConjugateRewModel>(m, "ConjugateRewModel")
        .def("update", &pf::ConjugateRewModel::update, "a"_a, "r"_a, "confidence"_a = 1.0)
        .def("pred_lh", &pf::ConjugateRewModel::pred_lh, "a"_a, "r"_a)
        .def("sample_post_dist", &pf::ConjugateRewModel::sample_post_dist, "rng"_a);

    using BB = pf::ConjugateBernoulli;
    nb::class_<BB, pf::ConjugateRewModel>(m, "ConjugateBernoulli")
        .def(nb::init<const Eigen::VectorXd&, const Eigen::VectorXd&>(), "alpha0"_a, "beta0"_a)
        .def("reset_posterior", &BB::reset_posterior)
        .def_prop_ro("suffstats", copy_of(&BB::suffstats))
        .def_ro("K", &BB::K)
        .def("post_params", [](const BB& s) {
            nb::dict d;
            d["alpha_n"] = Eigen::VectorXd(s.alpha_n);
            d["beta_n"] = Eigen::VectorXd(s.beta_n);
            return d;
        })
        .def("_pred_dist_params", [](const BB& s) {
            nb::dict d;
            d["p_rew"] = s._pred_dist_params();
            return d;
        });

    using OA = pf::ConjugateOptimalArm;
    nb::class_<OA, pf::ConjugateRewModel>(m, "ConjugateOptimalArm")
        .def(nb::init<double, double, const Eigen::VectorXd&>(), "rho_c"_a, "rho_i"_a, "beta0"_a)
        .def("reset_posterior", &OA::reset_posterior)
        .def_prop_ro("suffstats", copy_of(&OA::suffstats))
        .def_ro("K", &OA::K)
        .def("post_params", [](const OA& s) {
            nb::dict d;
            d["beta_n"] = Eigen::VectorXd(s.beta_n);
            return d;
        })
        .def("_pred_dist_params", [](const OA& s) {
            nb::dict d;
            d["p_rew"] = s._pred_dist_params();
            return d;
        });

    // ----------------------------------------
    //   models_context
    // ----------------------------------------
    nb::class_<pf::CRP>(m, "CRP")
        .def(nb::init<double>(), "hyp_alpha"_a)
        .def("update", &pf::CRP::update, "c"_a)
        .def("sample", &pf::CRP::sample, "rng"_a)
        .def_ro("hyp_alpha", &pf::CRP::hyp_alpha)
        .def_ro("n_active_contexts", &pf::CRP::n_active_contexts)
        .def_prop_ro("counts", copy_of(&pf::CRP::counts))
        .def_prop_ro("probs", copy_of(&pf::CRP::probs));

    nb::class_<pf::ContextModel>(m, "ContextModel")
        .def("update", &pf::ContextModel::update, "c_t"_a)
        .def_prop_ro("n_hypotheses", &pf::ContextModel::n_hypotheses)
        .def("hypothesis_o", &pf::ContextModel::hypothesis_o, "c"_a)
        .def("hypothesis_r", &pf::ContextModel::hypothesis_r, "c"_a)
        .def("hypothesis_jump", &pf::ContextModel::hypothesis_jump, "c"_a)
        .def_prop_ro("hypothesis_probs", copy_of(&pf::ContextModel::hypothesis_probs))
        .def_prop_ro("context_hypotheses", [](const pf::ContextModel& s) {
            nb::list hypotheses;
            for (const pf::ContextHypothesis& h : s.context_hypotheses) {
                nb::dict d;
                d["jump"] = h.jump;
                d["obs_model"] = h.obs_model;
                d["rew_model"] = h.rew_model;
                hypotheses.append(d);
            }
            return hypotheses;
        });

    nb::class_<pf::jCRP, pf::ContextModel>(m, "jCRP")
        .def(nb::init<double, double>(), "gamma"_a, "alpha"_a)
        .def_ro("hyp_gamma", &pf::jCRP::hyp_gamma)
        .def_ro("hyp_alpha", &pf::jCRP::hyp_alpha)
        .def_prop_ro("CRP", copy_of(&pf::jCRP::crp))
        .def_ro("prev_c", &pf::jCRP::prev_c);

    nb::class_<pf::CjCRP, pf::ContextModel>(m, "CjCRP")
        .def(nb::init<double, double, double>(), "gamma"_a, "alpha_o"_a, "alpha_r"_a)
        .def_ro("hyp_gamma", &pf::CjCRP::hyp_gamma)
        .def_ro("hyp_alpha_o", &pf::CjCRP::hyp_alpha_o)
        .def_ro("hyp_alpha_r", &pf::CjCRP::hyp_alpha_r)
        .def_prop_ro("CRP_o", copy_of(&pf::CjCRP::CRP_o))
        .def_prop_ro("CRP_r", copy_of(&pf::CjCRP::CRP_r))
        .def_ro("prev_c_o", &pf::CjCRP::prev_c_o)
        .def_ro("prev_c_r", &pf::CjCRP::prev_c_r);

    // ----------------------------------------
    //   particle
    // ----------------------------------------
    nb::class_<pf::IdealObsParams>(m, "IdealObsParams")
        .def(nb::init<const pf::ContextModel&, const pf::ConjugateObsModel&, const pf::ConjugateRewModel&>(), "context"_a, "obs"_a, "rew"_a);

    nb::class_<pf::Particle>(m, "Particle")
        .def(nb::init<const pf::IdealObsParams&>(), "particle_params"_a)
        .def("copy", [](const pf::Particle& p) { return pf::Particle(p); })
        .def("before_action", &pf::Particle::before_action, "o_t"_a)
        .def("sample_action", &pf::Particle::sample_action, "rng"_a)
        .def("after_action", &pf::Particle::after_action, "o_t"_a, "a_t"_a, "r_t"_a, "rng"_a)
        .def("latent_posterior", &pf::Particle::latent_posterior, "a_t"_a, "r_t"_a)
        .def("update_params", &pf::Particle::update_params, "o_t"_a, "a_t"_a, "r_t"_a, "s_t"_a, "c_t"_a)
        .def("obs_likelihood", &pf::Particle::obs_likelihood, "o_t"_a)
        .def("rew_likelihood", &pf::Particle::rew_likelihood, "a_t"_a, "r_t"_a)
        .def_prop_ro("n_obs_models", &pf::Particle::n_obs_models)
        .def_prop_ro("n_rew_models", &pf::Particle::n_rew_models)
        .def_rw("log_weight", &pf::Particle::log_weight)
        .def_prop_ro("temp_belief", copy_of(&pf::Particle::temp_belief))
        .def_prop_ro("context_model", [](pf::Particle& p) -> pf::ContextModel& { return *p.context_model; }, nb::rv_policy::reference_internal)
        .def("obs_model", [](pf::Particle& p, int s, int c) -> pf::ConjugateObsModel& { return *p.obs_models.at(s).at(c); },
             "s"_a, "c"_a, nb::rv_policy::reference_internal)
        .def("rew_model", [](pf::Particle& p, int s, int c) -> pf::ConjugateRewModel& { return *p.rew_models.at(s).at(c); },
             "s"_a, "c"_a, nb::rv_policy::reference_internal);

    // ----------------------------------------
    //   idealobserver
    // ----------------------------------------
    // The methods that do the particle work release the GIL, as they do not touch Python objects
    nb::class_<pf::IdealObsPF>(m, "IdealObsPF")
        .def(nb::init<int, const pf::IdealObsParams&, std::uint64_t>(), "N"_a, "hyp_params"_a, "seed"_a)
        .def("before_action", &pf::IdealObsPF::before_action, "o_t"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("select_action", &pf::IdealObsPF::select_action, nb::call_guard<nb::gil_scoped_release>())
        .def("after_action", &pf::IdealObsPF::after_action, "o_t"_a, "a_t"_a, "r_t"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("resample", &pf::IdealObsPF::resample, nb::call_guard<nb::gil_scoped_release>())
        .def_ro("N", &pf::IdealObsPF::N)
        .def_prop_ro("weights", &pf::IdealObsPF::weights)
        .def_prop_ro("p_action", &pf::IdealObsPF::p_action)
        .def_prop_ro("p_state", &pf::IdealObsPF::p_state)
        .def_prop_ro("p_jump", &pf::IdealObsPF::p_jump)
        .def_prop_ro("log_weights", copy_of(&pf::IdealObsPF::log_weights))
        .def_prop_ro("vec_s_t", copy_of(&pf::IdealObsPF::vec_s_t))
        .def_prop_ro("vec_c_o_t", copy_of(&pf::IdealObsPF::vec_c_o_t))
        .def_prop_ro("vec_c_r_t", copy_of(&pf::IdealObsPF::vec_c_r_t))
        .def_prop_ro("vec_j_t", copy_of(&pf::IdealObsPF::vec_j_t))
        .def_prop_ro("last_actions", copy_of(&pf::IdealObsPF::last_actions))
        .def("particle", [](pf::IdealObsPF& s, int i) -> pf::Particle& { return s.particles.at(i); }, "i"_a, nb::rv_policy::reference_internal);
}
