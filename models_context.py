from typing import TypedDict, Literal

import numpy as np

from utils_sample import cat_sample


class ContextHypothesis(TypedDict):
    """
    A context hypothesis represents a possible combination of a jump, an observation model, and a reward model.
    """
    jump : Literal[0, 1]
    obs_model : int
    rew_model : int

class ContextModel():
    """
    General interface for a context model. This is an abstract class that should be subclassed to implement specific context models.
    The methods here provide type hints and documentation for the expected behavior of context models.

    Subclasses must set the following attributes whenever the context hypotheses change:
        context_hypotheses: list of all currently possible context hypotheses c
        hypothesis_probs:   P_C(C_t = c | xi_{t-1}) for all currently possible c, shape [n_hypotheses]
    """
    context_hypotheses : list[ContextHypothesis]
    hypothesis_probs   : np.ndarray

    @property
    def n_hypotheses(self) -> int:
        return len(self.context_hypotheses)

    def hypothesis_o(self, c: int) -> int:
        """
        Return the index of the observation model associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["obs_model"]

    def hypothesis_r(self, c: int) -> int:
        """
        Return the index of the reward model associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["rew_model"]

    def hypothesis_jump(self, c: int) -> int:
        """
        Return the jump indicator (0 or 1) associated with context hypothesis c.
        """
        return self.context_hypotheses[c]["jump"]

    def update(self, c_t: int):
        raise NotImplementedError




class CRP():
    """
    Class for a Chinese Restaurant Process Prior
    """
    def __init__(self, hyp_alpha):
        self.hyp_alpha = hyp_alpha
        
        self.counts = np.zeros(100) # this is an artificial upper limit on the number of contexts that is not expected to be reached

        self.n_active_contexts = 0

        self._update_probs()

    def _update_probs(self):
        """
        Prior probabilities for each category, including a new one. \\
        These only change when the counts do, so are computed here rather than each time they are needed.
        """
        # probability of each context (including a new one)
        prob = np.zeros(self.n_active_contexts+1)
        prob[:-1] = self.counts[:self.n_active_contexts]
        prob[-1] = self.hyp_alpha
        self.probs = prob / prob.sum()

    def update(self, c:int):
        """
        Update the counts with a single context assignment
        """
        self.counts[c] += 1

        if c == self.n_active_contexts:
            self.n_active_contexts += 1

        self._update_probs()

    def sample(self) -> int:
        """
        Sample from the CRP prior
        """
        return cat_sample(self.probs)


class jCRP(ContextModel):
    """
    Class for a jump CRP
    """
    def __init__(self, gamma: float, alpha: float):
        self.hyp_gamma = gamma
        self.hyp_alpha = alpha

        self.CRP = CRP(hyp_alpha=alpha)

        # Due to the jump mixture, we need to keep track of the previous context
        self.prev_c = 0

        # Fill the list of context hypotheses based on the current state of the CRP
        self.fill_context_hypotheses()

    def fill_context_hypotheses(self):
        """
        Fill the list of context hypotheses based on the current state of the CRP.
        A single context determines both the observation model and reward model.
        """
        self.context_hypotheses = []

        # If there are no active contexts, we can only have a jump to a new context
        if self.CRP.n_active_contexts == 0:
            self.context_hypotheses.append({
                "jump":      1, 
                "obs_model": 0, 
                "rew_model": 0
                })

        # Otherwise
        else:
            # J = 0: exactly one possible hypothesis
            self.context_hypotheses.append(
                {
                    "jump":      0,
                    "obs_model": self.prev_c,
                    "rew_model": self.prev_c,
                })

            # J = 1: one hypothesis for each possible CRP context
            for c in range(self.CRP.n_active_contexts + 1):
                self.context_hypotheses.append(
                    {
                        "jump":      1,
                        "obs_model": c,
                        "rew_model": c,
                    })

        # Prior probabilities of the context hypotheses
        # If there is only one hypothesis, it must be the jump-to-new-context hypothesis, so its probability is 1
        if len(self.context_hypotheses) == 1:
            self.hypothesis_probs = np.ones(1)

        # Otherwise, compute probabilities based on the CRP prior and the jump probability
        else:
            self.hypothesis_probs = np.zeros(self.n_hypotheses)

            # Hypothesis 0 is always the unique stay hypothesis
            self.hypothesis_probs[0] = 1.0 - self.hyp_gamma

            # Remaining hypotheses are jumps, one for each CRP context c in order
            # The same context determines both observation and reward models
            self.hypothesis_probs[1:] = self.hyp_gamma * self.CRP.probs

    def update(self, c_t: int):
        """
        Wrapper function for updating the underlying CRP.
        """
        # Extract the jump and shared context from the current hypothesis
        hyp = self.context_hypotheses[c_t]
        j_t = hyp["jump"]
        c_t = hyp["obs_model"]

        # Sufficient statistics are only updated when a jump occurs
        if j_t == 1:
            # Update the single shared context CRP
            self.CRP.update(c_t)

            # Update previous context
            self.prev_c = c_t

            self.fill_context_hypotheses()  # Refill the context hypotheses after the update


class CjCRP(ContextModel):
    """
    Class for a coupled jump CRP.
    Observation and reward contexts have separate CRPs, which share the concentration parameter alpha, so that the model has the same hyperparameters as the jCRP.
    """
    def __init__(self, gamma : float, alpha : float):
        self.hyp_gamma = gamma
        self.hyp_alpha = alpha

        # Set up individual CRPs for observation and reward contexts
        self.CRP_o = CRP(hyp_alpha=alpha)
        self.CRP_r = CRP(hyp_alpha=alpha)

        # Due to the jump mixture, we need to keep track of the previous contexts
        self.prev_c_o = 0
        self.prev_c_r = 0

        # Fill the list of context hypotheses based on the current state of the CRPs
        self.fill_context_hypotheses()

    def fill_context_hypotheses(self):
        """
        Fill the list of context hypotheses based on the current state of the CRPs.
        Each hypothesis is a combination of a jump (0 or 1), an observation model, and a reward model.
        """
        self.context_hypotheses = []

        # If there are no active contexts, we can only have a jump to a new context
        if self.CRP_o.n_active_contexts == 0 and self.CRP_r.n_active_contexts == 0:
            self.context_hypotheses.append({
                "jump":      1,
                "obs_model": 0,
                "rew_model": 0
            })

        # Otherwise
        else:
            # J = 0: exactly one possible hypothesis
            self.context_hypotheses.append({
                "jump":      0,
                "obs_model": self.prev_c_o,
                "rew_model": self.prev_c_r,
            })

            # J = 1: Cartesian product of CRP possibilities
            for c_o in range(self.CRP_o.n_active_contexts + 1):
                for c_r in range(self.CRP_r.n_active_contexts + 1):
                    self.context_hypotheses.append({
                        "jump":    1,
                        "obs_model": c_o,
                        "rew_model": c_r,
                    })

        # Prior probabilities of the context hypotheses
        # If there is only one hypothesis, it must be the jump-to-new-context hypothesis, so its probability is 1
        if len(self.context_hypotheses) == 1:
            self.hypothesis_probs = np.ones(1)

        # Otherwise, we compute the probabilities for each hypothesis based on the CRP priors and the jump probability
        else:
            self.hypothesis_probs = np.zeros(self.n_hypotheses)

            # Hypothesis 0 is always the unique stay hypothesis
            self.hypothesis_probs[0] = 1.0 - self.hyp_gamma

            # Remaining hypotheses 1, 2, 3 ... are jumps, ordered as in the Cartesian product above (c_o outer, c_r inner),
            # which is the order of the flattened outer product of the CRP probabilities
            self.hypothesis_probs[1:] = ((self.hyp_gamma * self.CRP_o.probs)[:, None] * self.CRP_r.probs[None, :]).flatten()

    def update(self, c_t: int):
        """
        Wrapper function that combines updates to the underlying CRPs.
        """
        # Extract the jump, obs_model, and rew_model from the current context hypothesis
        hyp = self.context_hypotheses[c_t]
        j_t = hyp["jump"]
        c_o_t = hyp["obs_model"]
        c_r_t = hyp["rew_model"]

        # Sufficient statistics are only updated when a jump occurs
        if j_t == 1:
            # Update counts
            self.CRP_o.update(c_o_t)
            self.CRP_r.update(c_r_t)
            
            # Update previous context
            self.prev_c_o = c_o_t
            self.prev_c_r = c_r_t

            self.fill_context_hypotheses()  # Refill the context hypotheses after the update