import numpy as np
import math

import job_distribution


class Parameters:
    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000        # [x] number of training epochs
        self.simu_len = 50             # number of jobs in a jobset (includes jobs of size 0)
        self.num_ex = 1                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 510  # enforces an maximum horizon

        self.num_q = 7                 # [x] number of queues in the system
        self.num_nw = 1                # [x] maximum allowed number of work in the queue

        self.time_horizon = 80         # number of time steps in the graph
        self.max_job_len = 15          # maximum duration of new jobs

        self.max_job_size = 10         # maximum resource request of new work

        self.backlog_size = 60         # backlog queue size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.q_num_cap = 40            # maximum number of distinct colors in current work graph

        self.new_job_rate = 1.0        # lambda in new job arrival Poisson Process

        self.job_aware = 0             # 1 => loader aware of pending job size
                                       # 0 => loader oblivious to pending job size

        self.discount = 1              # discount factor

        # distribution for new job arrival
        self.dist = job_distribution.Dist(self.num_q, self.max_job_len)

        # neural network matrix input
        self.network_input_width = self.num_q + self.job_aware
        self.network_input_height = self.time_horizon
        # neural network vectorized input
        self.network_vec_input_dim = self.num_q + self.job_aware
        # neural network outputs probability distribution over queues
        self.network_output_dim = self.num_q

        self.delay_penalty = -1       # penalty for delaying things in the current work screen
        self.hold_penalty = -1        # penalty for holding things in the new work screen
        self.dismiss_penalty = -1     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.unseen = False  # change random seed to generate unseen example

        # supervised learning mimic policy
        self.batch_size = 10
        self.evaluate_policy_name = "LLQ"

    def compute_dependent_parameters(self):
        # neural network matrix input
        self.network_input_width = self.num_q + self.job_aware
        self.network_input_height = self.time_horizon
        # neural network vectorized input
        self.network_vec_input_dim = self.num_q + self.job_aware
        # neural network outputs probability distribution over queues
        self.network_output_dim = self.num_q
