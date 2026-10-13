from copy import deepcopy

import numpy as np
import torch
import torch.distributions as D

from particle import Particle
from subject import Subject, IdealObsParams
from utils_sample import bern_sample


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
        self.log_weights    = torch.zeros(N, dtype=torch.float64)
        self.vec_s_t        = torch.zeros(N, dtype=torch.long)
        self.vec_c_o_t      = torch.zeros(N, dtype=torch.long)
        self.vec_c_r_t      = torch.zeros(N, dtype=torch.long)
        self.vec_j_t        = torch.zeros(N, dtype=torch.long)

    @property
    def weights(self):
        """
        Weights of each particle, summing to 1
        """
        return torch.exp(self.log_weights - torch.logsumexp(self.log_weights, dim=0))

    def before_action(self, o_t: np.ndarray | torch.Tensor):
        """
        Process the incoming observation o_t
        """
        o_t = torch.as_tensor(o_t, dtype=torch.float32) # the environment provides numpy observations, the particles compute in float32 torch

        for particle in self.particles:
            particle.before_action(o_t)

    def select_action(self):
        """
        Select an action a_t
        """
        # Each particle samples an action
        actions = torch.tensor([particle.sample_action() for particle in self.particles], dtype=torch.long)

        self._last_actions = actions # store for eval

        # Pick a particle from which the action is emitted
        selected_particle = D.Categorical(probs=self.weights).sample().item()
        selected_action = int(actions[selected_particle].item())

        return selected_action

    def after_action(self, o_t: np.ndarray | torch.Tensor, a_t: int, r_t: float):
        """
        Process the resulting o_t, a_t, r_t emissions at time t
        """
        o_t = torch.as_tensor(o_t, dtype=torch.float32) # the environment provides numpy observations, the particles compute in float32 torch

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
        ancestors = D.Categorical(logits=self.log_weights).sample((self.N,)).tolist()

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
        self.log_weights.zero_()

    @property
    def p_action(self) -> np.ndarray:
        """
        Weighted action probabilities at the current time step, inferred from the particles' last sampled actions.
        """
        w = self.weights.to(torch.float64)  # [N], sums to 1

        acts = self._last_actions.to(torch.long)  # [N]
        p_actions = torch.bincount(acts, weights=w, minlength=4).to(torch.float)
        p_actions = torch.round(p_actions, decimals=2).detach().numpy()

        return p_actions

    @property
    def p_state(self) -> np.ndarray:
        """
        Probabilities of latent states at the current time step, inferred from the particles' latent state variable vec_s_t and weighted by particle weights.
        """
        w = self.weights.to(torch.float64)  # [N], sums to 1

        # 1) inferred latent state distribution P(s_t)
        p_s = torch.bincount(self.vec_s_t.to(torch.long), weights=w, minlength=2).to(torch.float) # vec_s_t is [N] with entries 0/1
        p_s = torch.round(p_s, decimals=2).detach().numpy()

        return p_s

    @property
    def p_jump(self) -> float:
        """
        Probability of a jump at the current time step, inferred from the particles' jump variable vec_j_t and weighted by particle weights.
        """
        w = self.weights.to(torch.float64)  # [N], sums to 1

        # 3) inferred jump probability P(j_t == 1)
        p_jump = w[self.vec_j_t.to(torch.long) == 1].sum().to(torch.float) # vec_j_t is [N] with entries 0/1
        p_jump = float(torch.round(p_jump, decimals=2).detach().numpy())

        return p_jump
