import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import utils_sample
from utils import niceprint
from subject import Subject
from idealobserver import IdealObsPF

# The C++ ideal observer is only available once the pf_cpp module has been built for this machine.
# If it is not, CPP_BACKEND_ERROR explains how to build it (and is None otherwise).
try:
    from idealobserver_cpp import IdealObsPFCpp
    PF_SUBJECTS = (IdealObsPF, IdealObsPFCpp)
    CPP_BACKEND_ERROR = None
except ImportError as error:
    if error.name != "pf_cpp": # an import error in the Python code itself
        raise
    IdealObsPFCpp = None
    PF_SUBJECTS = (IdealObsPF,)
    CPP_BACKEND_ERROR = f"The C++ backend is not built for this machine ({error}). Build it with `python cpp/build_cpp.py`, then restart the kernel."

class History():
    """
    Base class for histories, which track variables as lists with one entry per time step
    """
    def to_numpy(self):
        """
        Copy of the history in which each tracked variable is a numpy array, with time steps along the first axis
        """
        out = self.__class__.__new__(self.__class__)
        for name, values in vars(self).items():
            setattr(out, name, np.array(values))
        return out


class ExperimentHistory(History):
    """
    Class to store the history of an experiment, tracking variables available to the experimenter
    """
    def __init__(self):
        # Initialise the variables to be tracked
        self.state          : list[int]             = [] # true latent state (not observable to participant)
        self.context_o      : list[list[int]]       = [] # true context for observation (not observable to participant)
        self.context_r      : list[list[int]]       = [] # true context for reward (not observable to participant)
        self.observation    : list[np.ndarray]      = [] # observation passed to participant
        self.optimal_action : list[int]             = [] # optimal action given the true state and reward context (not observable to participant)
        self.action         : list[int]             = [] # action taken by participant
        self.reward         : list[float]           = [] # reward given to participant

   
    def append(self, state, context_o, context_r, observation, optimal_action, action, reward):
        """
        Append a new timestep to the history, with the given variables
        """
        self.state.append(state)
        self.context_o.append(context_o)
        self.context_r.append(context_r)
        self.observation.append(observation)
        self.optimal_action.append(optimal_action)
        self.action.append(action)
        self.reward.append(reward)
       
    def __len__(self):
        """
        Returns the number of timesteps in the history
        """
        return len(self.state)


class SubjectHistory(History):
    """
    Class to store the history of a participant's internal variables, which are not directly observable to the experimenter
    """
    def __init__(self):
        self.p_action   : list[np.ndarray] = [] # probability of each action being chosen

    def append(self, p_action):
        """
        Append a new timestep to the history, with the given variables
        """        
        self.p_action.append(p_action)
        
    def __len__(self):
        return len(self.p_action)


class IOPFSubjectHistory(SubjectHistory):
    """
    Class to store the history of a participant's internal variables, which are not directly observable to the experimenter
    """
    def __init__(self):
        super().__init__()                      # probability of each action being chosen is tracked in the base class
        self.p_state    : list[np.ndarray] = [] # probability of each latent state
        self.p_jump     : list[float]      = [] # probability of a jump
        self.context_o  : list[np.ndarray] = [] # sampled context_o for each particle
        self.context_r  : list[np.ndarray] = [] # sampled context_r for each particle
        self.weights    : list[np.ndarray] = [] # weights of each particle

    def append(self,  p_action, p_state, p_jump, context_o, context_r, weights):
        """
        Append a new timestep to the history, with the given variables
        """        
        super().append(p_action) # action probabilities are tracked in the base class
        self.p_state.append(p_state)
        self.p_jump.append(p_jump)
        self.context_o.append(context_o)
        self.context_r.append(context_r)
        self.weights.append(weights)
        


class ExperimentalEnv():
    """
    Class to represent the experimental environment
    """
    def __init__(self, context_o_params : dict = None, context_r_params : dict = None, seed : int | None = None):
        """
        Observation and reward contexts are defined by their parameters, which are provided at initialisation.

        :param dict[int, dict] context_o_params: parameters for observation contexts, of the form {context_id: {state_id: {param_name: param_value}}}
        :param dict[int, dict] context_r_params: parameters for reward contexts, of the form {context_id: {state_id: {param_name: param_value}}}
        :param int seed: seed for the environment's random number generator, see seed()
        """
        self.state_probs = np.array([0.5, 0.5])
        self.state_cdf = np.cumsum(self.state_probs)

        self.seed(seed)

        
        assert type(context_o_params) == dict, "context_o_params must be a dictionary of the form {context_id: context_parameters}"
        assert all(isinstance(k, int) for k in context_o_params.keys()), "context_o_params keys must be integers representing context IDs"
        assert all(isinstance(v, dict) for v in context_o_params.values()), "context_o_params values must be dictionaries representing context parameters"
        self.contexts_o = list(context_o_params.keys())

        for context_id, context_params in context_o_params.items():
            assert all(isinstance(k, int) for k in context_params.keys()), f"State IDs in context_o_params for context {context_id} must be integers"
            assert all(isinstance(v, dict) for v in context_params.values()), f"Parameters for each state in context_o_params for context {context_id} must be dictionaries"

            for state_id, state_params in context_params.items():
                assert "loc" in state_params and "cov" in state_params, f"Parameters for state {state_id} in context_o_params for context {context_id} must include 'loc' and 'cov'"
                assert isinstance(state_params["loc"], np.ndarray), f"'loc' parameter for state {state_id} in context_o_params for context {context_id} must be a numpy array"
                assert isinstance(state_params["cov"], np.ndarray), f"'cov' parameter for state {state_id} in context_o_params for context {context_id} must be a numpy array"

        self.context_o_params : dict[int, dict] = context_o_params

        # Cholesky factors of the observation covariances, so that sampling an observation is loc + L @ z with z ~ N(0, I)
        self.obs_cov_tril = {
            context_id: {state_id: np.linalg.cholesky(state_params["cov"]) for state_id, state_params in context_params.items()}
            for context_id, context_params in context_o_params.items()
        }


        assert type(context_r_params) == dict, "context_r_params must be a dictionary of the form {context_id: context_parameters}"
        assert all(isinstance(k, int) for k in context_r_params.keys()), "context_r_params keys must be integers representing context IDs"
        assert all(isinstance(v, dict) for v in context_r_params.values()), "context_r_params values must be dictionaries representing context parameters"
        self.contexts_r = list(context_r_params.keys())
        
        for context_id, context_params in context_r_params.items():
            assert all(isinstance(k, int) for k in context_params.keys()), f"State IDs in context_r_params for context {context_id} must be integers"
            assert all(isinstance(v, dict) for v in context_params.values()), f"Parameters for each state in context_r_params for context {context_id} must be dictionaries"

            for state_id, state_params in context_params.items():
                assert "p_rew" in state_params, f"Parameters for state {state_id} in context_r_params for context {context_id} must include 'p_rew'"
                assert all(isinstance(v, np.ndarray) for v in state_params.values()), f"Reward probabilities in context_r_params for state {state_id} in context {context_id} must be numpy arrays"

        self.context_r_params : dict[int, dict] = context_r_params


    def seed(self, seed : int | None = None):
        """
        (Re)initialise the environment's random number generator. With seed None, it is seeded from fresh OS entropy.
        """
        self.rng = np.random.default_rng(seed)

    def before_action(self, c_o):

        # Sample states i.i.d. for each trial (inverse CDF, clipped in case the CDF sums to slightly less than 1)
        s_t = min(int(np.searchsorted(self.state_cdf, self.rng.random(), side='right')), len(self.state_probs) - 1)

        # Sample observation given the state and observation context
        obs_params = self.context_o_params[c_o][s_t]
        o_t = obs_params["loc"] + self.obs_cov_tril[c_o][s_t] @ self.rng.standard_normal(obs_params["loc"].shape[0])

        return s_t, o_t

    def after_action(self, c_r, s_t, a_t):

        rew_params = self.context_r_params[c_r][s_t]

        r_t = float(self.rng.random() < rew_params["p_rew"][a_t])

        opt_a_t = int(np.argmax(rew_params["p_rew"]))

        return r_t, opt_a_t
    


    def trial_step(
        self, 
        subject: Subject,
        c_o : int,
        c_r : int,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        print_level : int = 0
        ):
        # -------------------------
        #   Checks and assertions
        # -------------------------
        assert isinstance(subject, Subject), "Subject must be an instance of the Subject class or its subclasses"
        assert c_o in self.contexts_o, "Observation context must be one of the contexts defined in the environment"
        assert c_r in self.contexts_r, "Reward context must be one of the contexts defined in the environment"
        assert isinstance(experiment_history, ExperimentHistory), "experiment_history must be an instance of the ExperimentHistory class"
        assert isinstance(subject_history, SubjectHistory), "subject_history must be an instance of the SubjectHistory class or its subclasses"
        assert print_level in [0, 1, 2, 3], "print_level must be an integer between 0 and 3, inclusive"


        # -----------------
        #   Before action
        # -----------------
        # Environment genreates state, observation and optimal action based on the true contexts
        s_t, o_t = self.before_action(c_o)
        
        if print_level > 1:
            print(f"\tTrue state: {s_t}, Obs.: {niceprint(o_t, 2)}", end='')
        
        # Subject processes observation
        subject.before_action(o_t)    
        

        # ------------------------
        #   Action selection
        # ------------------------
        a_t = subject.select_action()
        

        # ----------------
        #   After action
        # ----------------
        # Environment generates reward based on the true state, true reward context and selected action
        r_t, opt_a_t = self.after_action(c_r, s_t, a_t)

        if print_level > 1:
            print(f", Action: {a_t}, Reward: {r_t}", end='')

        # Subject updates internal representations based on the selected action and received reward
        subject.after_action(o_t, a_t, r_t)

        # Record keeping from the experimenter's perspective
        experiment_history.append(s_t, c_o, c_r, o_t, opt_a_t, a_t, r_t)

        # Record keeping from the subject's perspective, depends on the type of subject
        # (np.array copies, as the subject may update its variables in place)
        if isinstance(subject, PF_SUBJECTS) and isinstance(subject_history, IOPFSubjectHistory):
            subject_history.append(
                p_action=np.array(subject.p_action),
                p_state=np.array(subject.p_state),
                p_jump=float(subject.p_jump),
                context_o=np.array(subject.vec_c_o_t),
                context_r=np.array(subject.vec_c_r_t),
                weights=np.array(subject.weights)
                )
        else:
            subject_history.append(p_action=np.array(subject.p_action))
        
        
        if print_level > 2:          
            if isinstance(subject, PF_SUBJECTS):
                print(f", Action probs.: {niceprint(subject.p_action, 2)}", end='')

        if print_level > 1:
            print(f"\n", end='')


    def repeat_trials(
        self,
        subject: Subject,
        experiment_history: ExperimentHistory,
        subject_history: SubjectHistory,
        N : int, 
        c_o : int, 
        c_r : int,
        print_level : int = 0         
        ):
        """
        Run a batch of trials with the given context combination.
        """

        if print_level > 0:
            print(f"\n << Running {N} trials with true obs. context = {c_o} and true rew. context = {c_r} >> \n")

        for i in range(N):
            if print_level > 1:
                print(f"Step {i+1}:   ", end='')
            # Run a single trial step
            self.trial_step(subject, c_o, c_r, experiment_history, subject_history, print_level)
        
        if print_level > 0:
            print(f"\n >> Batch completed \n")


class CombinedHistory():
    """
    Class to perform analysis on the experimental and subject history. \\
    Stores copies of the histories in which each tracked variable is a numpy array, with time steps along the first axis.
    """
    def __init__(self, experiment_history: ExperimentHistory, subject_history: SubjectHistory):
        self.experiment_history = experiment_history.to_numpy()
        self.subject_history = subject_history.to_numpy()

    @property
    def p_optimal_action(self):
        """
        Calculate the probability of optimal actions being taken by the subject at each time step
        """
        n_steps = len(self.experiment_history.optimal_action)
        return self.subject_history.p_action[np.arange(n_steps), self.experiment_history.optimal_action]


def run_replicate(
    env: ExperimentalEnv,
    agent_params: dict,
    N: int,
    schedule: list[tuple[int, int, int]],
    seed: int | None = None,
    print_level: int = 0,
    backend: str = "python"
    ) -> CombinedHistory:
    """
    Run a single replicate of the experiment with a freshly initialised ideal observer.

    :param ExperimentalEnv env: environment to run the trials in
    :param dict agent_params: parameters of the ideal observer (IdealObsParams)
    :param int N: number of particles
    :param list schedule: blocks of trials, run in order, each of the form (n_trials, c_o, c_r)
    :param int seed: seed for the random number generators of the environment and of the ideal observer (utils_sample.rng, which also seeds the C++ ideal observer).
        If None, they are seeded from fresh OS entropy.
    :param str backend: implementation of the ideal observer, "python" (IdealObsPF) or "cpp" (IdealObsPFCpp)
    """
    # Independent random streams for the environment and the ideal observer (seeding both with seed would make them draw the same numbers).
    # They are always reseeded, as copies of them in worker processes would otherwise all draw the same random numbers.
    env_seed, observer_seed = np.random.SeedSequence(seed).spawn(2)
    env.seed(env_seed)
    utils_sample.seed(observer_seed)

    exp_history = ExperimentHistory()
    sub_history = IOPFSubjectHistory()
    if backend == "python":
        subject = IdealObsPF(N=N, hyp_params=agent_params)
    elif backend == "cpp":
        if CPP_BACKEND_ERROR is not None:
            raise RuntimeError(CPP_BACKEND_ERROR)
        subject = IdealObsPFCpp(N=N, hyp_params=agent_params)
    else:
        raise ValueError(f"Unknown backend {backend}, must be 'python' or 'cpp'")

    for n_trials, c_o, c_r in schedule:
        env.repeat_trials(subject, experiment_history=exp_history, subject_history=sub_history, N=n_trials, c_o=c_o, c_r=c_r, print_level=print_level)

    return CombinedHistory(experiment_history=exp_history, subject_history=sub_history)


def run_replicates(
    env: ExperimentalEnv,
    agent_params: dict,
    N: int,
    schedule: list[tuple[int, int, int]],
    n_replicates: int,
    n_workers: int,
    seed: int | None = None,
    backend: str = "python"
    ) -> list[CombinedHistory]:
    """
    Run independent replicates of the experiment in parallel, with one replicate per worker process at a time.

    :param int n_replicates: number of replicates to run
    :param int n_workers: number of worker processes
    :param int seed: if given, replicate i is seeded with seed + i (reproducible), otherwise each worker process is randomly seeded
    :param str backend: implementation of the ideal observer, "python" or "cpp", see run_replicate
    :return list[CombinedHistory]: histories in replicate order
    """
    assert 1 <= n_workers <= mp.cpu_count(), "Number of workers is limited to the number of hardware threads"
    if backend == "cpp" and CPP_BACKEND_ERROR is not None: # fail before starting the workers
        raise RuntimeError(CPP_BACKEND_ERROR)

    seeds =[None if seed is None else seed + i for i in range(n_replicates)]

    # 'spawn' rather than 'fork', as forking a multithreaded process (e.g. a notebook kernel) can deadlock
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(run_replicate, env, agent_params, N, schedule, s, backend=backend) for s in seeds]
        return [f.result() for f in futures]
    
