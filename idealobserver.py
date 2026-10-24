from copy import deepcopy

import numpy as np

from particle import Particle
from subject import Subject, IdealObsParams
from utils_sample import rng, bern_sample, cat_sample


class IdealObsPF(Subject):
    """
    Particle filtering implementation of the ideal observer.
    """
    def __init__(self, N: int, hyp_params: IdealObsParams):
        self.N = N
        self.hyp_params = hyp_params

        # Initialise the particles
        self.particles = [Particle(hyp_params) for _ in range(N)]

        # Initalise extra data structures that are tracked during particle filtering
        self.log_weights    = np.zeros(N)
        self.vec_s_t        = np.zeros(N, dtype=int)
        self.vec_c_o_t      = np.zeros(N, dtype=int)
        self.vec_c_r_t      = np.zeros(N, dtype=int)
        self.vec_j_t        = np.zeros(N, dtype=int)

    @property
    def weights(self):
        """
        Weights of each particle, summing to 1
        """
        # exp(log_weights - logsumexp(log_weights))
        max_log_weight = np.max(self.log_weights)
        logsumexp = max_log_weight + np.log(np.sum(np.exp(self.log_weights - max_log_weight)))
        return np.exp(self.log_weights - logsumexp)

    def before_action(self, o_t: np.ndarray):
        """
        Process the incoming observation o_t
        """
        for particle in self.particles:
            particle.before_action(o_t)

    def select_action(self):
        """
        Select an action a_t
        """
        # Each particle samples an action
        actions = np.array([particle.sample_action() for particle in self.particles], dtype=int)

        self._last_actions = actions # store for eval

        # Pick a particle from which the action is emitted
        selected_particle = cat_sample(self.weights)
        selected_action = int(actions[selected_particle])

        return selected_action

    def after_action(self, o_t: np.ndarray, a_t: int, r_t: float):
        """
        Process the resulting o_t, a_t, r_t emissions at time t
        """
        # Each particle conducts inference on the latent state and contexts, and updates internal parameters
        for i, particle in enumerate(self.particles):
            s_t, c_o_t, c_r_t, j_t = particle.after_action(o_t, int(a_t), float(r_t))
            self.vec_s_t[i]     = s_t
            self.vec_c_o_t[i]   = c_o_t
            self.vec_c_r_t[i]   = c_r_t
            self.vec_j_t[i]     = j_t
            self.log_weights[i] = float(particle.log_weight)

        # Resample particles with a low probability to maintain diversity
        if bern_sample(0.02) == 1.0:
            self.resample()

    def resample(self):
        """
        Resample particles according to weights
        """
        # sample N ancestor indices using current log weights
        ancestors = rng.choice(self.N, size=self.N, p=self.weights).tolist()

        # The first descendant of an ancestor reuses the ancestor object, further descendants get independent copies
        used = set()
        new_particles = []
        for anc in ancestors:
            if anc in used:
                new_particles.append(deepcopy(self.particles[anc]))
            else:
                used.add(anc)
                new_particles.append(self.particles[anc])
        self.particles = new_particles

        # Reset weights after resampling
        for particle in self.particles:
            particle.log_weight = 0.0
        self.log_weights[:] = 0.0

    def action_log_probs(self, n_samples: int = 1) -> np.ndarray:
        """
        Log probabilities of each action at the current time step, estimated from n_samples Thompson samples per particle, weighted by particle weights. \\
        Unlike p_action these are not rounded, so they can be used as the likelihood of a participant's choice.
        More samples reduce the noise of the estimate, mostly for unlikely actions.
        """
        w = self.weights  # [N], sums to 1

        p_actions = np.zeros(4)
        for _ in range(n_samples):
            actions = np.array([particle.sample_action() for particle in self.particles], dtype=int)
            p_actions += np.bincount(actions, weights=w, minlength=4)
        p_actions /= n_samples

        with np.errstate(divide='ignore'): # actions sampled by no particle have log probability -inf
            return np.log(p_actions)

    @property
    def p_action(self) -> np.ndarray:
        """
        Weighted action probabilities at the current time step, inferred from the particles' last sampled actions.
        """
        w = self.weights  # [N], sums to 1

        p_actions = np.bincount(self._last_actions, weights=w, minlength=4)
        p_actions = np.round(p_actions, decimals=2)

        return p_actions

    @property
    def p_state(self) -> np.ndarray:
        """
        Probabilities of latent states at the current time step, inferred from the particles' latent state variable vec_s_t and weighted by particle weights.
        """
        w = self.weights  # [N], sums to 1

        # 1) inferred latent state distribution P(s_t)
        p_s = np.bincount(self.vec_s_t, weights=w, minlength=2) # vec_s_t is [N] with entries 0/1
        p_s = np.round(p_s, decimals=2)

        return p_s

    @property
    def p_jump(self) -> float:
        """
        Probability of a jump at the current time step, inferred from the particles' jump variable vec_j_t and weighted by particle weights.
        """
        w = self.weights  # [N], sums to 1

        # 3) inferred jump probability P(j_t == 1)
        p_jump = w[self.vec_j_t == 1].sum() # vec_j_t is [N] with entries 0/1
        p_jump = float(np.round(p_jump, decimals=2))

        return p_jump
