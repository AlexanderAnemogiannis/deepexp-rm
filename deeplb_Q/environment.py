import numpy as np
import math
import matplotlib.pyplot as plt
import theano

import parameters


class Env:
       
    def __init__(self, pa, nw_len_seqs=None, seed=42, render=False, 
                 repre='image', end='no_new_job'):
        '''
        initialize environment parameters and objects
        input:  pa - clsas containing environment parameters
                nw_len_seqs - matrix where the i'th row is job durations
                              for the i'th jobset
                seed - used to randomly generate job sequences
                render - indicates if env state should be plotted after
                         each timestep
                repre - how state is represented (image, compact)
                end - specifies termination to be when there (1) are no new
                      new jobs or (2) when all jobs have finishd processing
                nw_dist - distribution of jobs (normal, bi_model_dist)
        objects: machine - hosts queues and processing jobs
                 job_slot - job queue that holds pending job(s)
                 job_backlog - job backlog that holds jobs not in job_slot
                 job_record - keeps track of jobs that have entered system
                 extra_info - additional information, like the time since the
                              last job arrived

        '''

        self.pa = pa
        self.render = render
        self.repre = repre  # image or compact representation
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.bi_model_dist

        self.curr_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        if nw_len_seqs is None:
            # generate jobs (for all job sequences and num_ex iterations of them)
            self.nw_len_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)

            # reshape vectorized job durations into a matrix, where each row is
            # a job sequence (and the number of job sequences is num_ex)
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
            
        else:
            self.nw_len_seqs = nw_len_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

    def generate_sequence_work(self, simu_len):
        """ 
        Returns sequences of jobs, characterized by their durations,
        for simu_len = (timesteps/episode) * num_episodes 
                     = L * N (from HotNets paper)
        --> note that for some timesteps, there may be no jobs
        """

        nw_len_seq = np.zeros(simu_len, dtype=int)

        for i in range(simu_len):
            # new_job_rate % of the timesteps have a job
            if np.random.rand() < self.pa.new_job_rate:
                nw_len_seq[i] = self.nw_dist()

        return nw_len_seq


    def get_new_job_from_seq(self, seq_no, seq_idx):
        """ 
        instantializes a Job object
        input:  seq_no - jobsets within an iteration
                seq_idx - indexes jobs within a jobset
        output: new_job - job object characterized 
        """

        new_job = Job(job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job


    def observe(self):
        """ 
        Returns representation of system state.
            image: matrix where the i'th column expresses the jobs stored in
                   the corresponding queue
            vec: vector of queue lengths and pending job size 

        """
        # image representation of queue occupancies and pending job
        if self.repre == 'image':

            image_repr = np.zeros((self.pa.network_input_height, \
                                   self.pa.network_input_width))

            # iterate over number of queues in the system
            for i in xrange(self.pa.num_q):
                
                # store image representation of allocated jobs in machine
                image_repr[:, 0 : self.pa.num_q] = self.machine.canvas

                # store image representation of pending job
                if self.pa.job_aware and self.job_slot.slot[0] is not None:
                    image_repr[: self.job_slot.slot[0].len, self.pa.num_q] = 1

            return image_repr

        # vector containing size of queues and pending job
        elif self.repre == 'vec':

            vec_repr = np.zeros(network_vec_input_dim, \
                                    dtype=theano.config.floatX)

            # store queue sizes
            vec_repr[:self.pa.num_q] = self.machine.avbl_slot # np.sum(self.machine.canvas != 0, axis=0)

            # store size of pending job
            if self.pa.job_aware and self.job_slot.slot[0] is not None:
                vec_repr[self.pa.num_q+1] = self.job_slot.slot[0].len

            return compact_repr

    # def plot_state(self):
    #     plt.figure("screen", figsize=(20, 5))

    #     skip_row = 0

    #     for i in xrange(self.pa.num_q):

    #         plt.subplot(self.pa.num_q,
    #                     1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                     i * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

    #         plt.imshow(self.machine.canvas[i, :, :], interpolation='nearest', vmax=1)

    #         for j in xrange(self.pa.num_nw):

    #             job_slot = np.zeros((self.pa.time_horizon, self.pa.max_job_size))
    #             if self.job_slot.slot[j] is not None:  # fill in a block of work
    #                 job_slot[: self.job_slot.slot[j].len, :self.job_slot.slot[j].res_vec[i]] = 1

    #             plt.subplot(self.pa.num_q,
    #                         1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                         1 + i * (self.pa.num_nw + 1) + j + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

    #             plt.imshow(job_slot, interpolation='nearest', vmax=1)

    #             if j == self.pa.num_nw - 1:
    #                 skip_row += 1

    #     skip_row -= 1
    #     backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))
    #     backlog = np.zeros((self.pa.time_horizon, backlog_width))

    #     backlog[: self.job_backlog.curr_size / backlog_width, : backlog_width] = 1
    #     backlog[self.job_backlog.curr_size / backlog_width, : self.job_backlog.curr_size % backlog_width] = 1

    #     plt.subplot(self.pa.num_q,
    #                 1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                 self.pa.num_nw + 1 + 1)

    #     plt.imshow(backlog, interpolation='nearest', vmax=1)

    #     plt.subplot(self.pa.num_q,
    #                 1 + self.pa.num_nw + 1,  # first +1 for current work, last +1 for backlog queue
    #                 self.pa.num_q * (self.pa.num_nw + 1) + skip_row + 1)  # plot the backlog at the end, +1 to avoid 0

    #     extra_info = np.ones((self.pa.time_horizon, 1)) * \
    #                  self.extra_info.time_since_last_new_job / \
    #                  float(self.extra_info.max_tracking_time_since_last_job)

    #     plt.imshow(extra_info, interpolation='nearest', vmax=1)

    #     plt.show()     # manual
    #     # plt.pause(0.01)  # automatic

    def get_reward(self):
        """
        Return rewards (penalties) of pending jobs, weighted by their status
        (if they are being processed or waiting in the job queue)
        """
        
        reward = 0
        
        # use reward that incetivizes minimizing avg slowdown
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)
        for j in self.job_backlog.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.len)

        return reward

    def step(self, a, repeat=False):
        """
        Assign pending job to queue designated by action a. Then either advance
        to the next timestep or terminate (if all jobs have been processed). If
        repeat is False, increment the job sequence counter, otherwise the next
        trajectory will be over the same sequence.
        """
        status = None

        done = False
        reward = 0
        info = None

        #######################################################################
        # Attempt to assign pending job to a'th queue in the machine
        #######################################################################

        # skip assignment if there is no pending job
        if self.job_slot.slot[0] is None:
            status = 'MoveOn'

        else:
            # attempt to make allocation to a'th queue
            allocated = self.machine.allocate_job(self.job_slot.slot[0], a, 
                                                  self.curr_time)
            
            # update system parameters if previous allocation succeeded
            if allocated:
                status = 'Allocate'
            else:
                status = 'MoveOn'  

        #######################################################################
        # Advance a timestep and adjust environmental parameters accordingly
        #######################################################################
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            self.extra_info.time_proceed()

            # add new jobs
            self.seq_idx += 1

            # ----- given end criteria, determine if job ------
            # ----- sequence has finished processing     ------

            # end of new job sequence
            if self.end == "no_new_job": 
                if self.seq_idx >= self.pa.simu_len:
                    done = True

            # everything has to be finished
            elif self.end == "all_done":  
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot) and \
                   all(s is None for s in self.job_backlog.backlog):
                    done = True

                # run too long, force termination
                elif self.curr_time > self.pa.episode_max_length:
                    done = True

            # if job sequence hasn't terminated, continue simulation
            if not done:
                # check that the job sequence has remaining entries
                if self.seq_idx < self.pa.simu_len:
                    new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx)

                    # check that job popped from job sequence isn't empty
                    if new_job.len > 0:
                        # if job queue/slot is open, use it to store new job
                        to_backlog = True
                        for i in xrange(self.pa.num_nw):
                            if self.job_slot.slot[i] is None:
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job
                                to_backlog = False
                                break
                        # otherwise attempt to store the job in the backlog
                        if to_backlog:
                            if self.job_backlog.curr_size < self.pa.backlog_size:
                                self.job_backlog.backlog[self.job_backlog.curr_size] = new_job
                                self.job_backlog.curr_size += 1
                                self.job_record.record[new_job.id] = new_job
                            else:  # abort, backlog full
                                print("Backlog is full.")
                                # exit(1)

                        self.extra_info.new_job_comes()

            reward = self.get_reward()

        if status == 'Allocate':
            self.job_record.record[self.job_slot.slot[0].id] = self.job_slot.slot[0]
            self.job_slot.slot[0] = None

            # dequeue backlog
            if self.job_backlog.curr_size > 0:
                # if backlog empty, it will be 0
                self.job_slot.slot[0] = self.job_backlog.backlog[0]  
                self.job_backlog.backlog[: -1] = self.job_backlog.backlog[1:]
                self.job_backlog.backlog[-1] = None
                self.job_backlog.curr_size -= 1
                 
        # allocation and time advancement complete - update remaining
        # system parameters
        ob = self.observe()
        info = self.job_record

        # if terminating criteria have been met, prepare to process another
        # job sequence
        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()
        
        if self.render:
            self.plot_state()

        return ob, reward, done, info

    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_backlog = JobBacklog(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)


class Job:
    def __init__(self, job_len, job_id, enter_time):
        self.id = job_id
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw


class JobBacklog:
    def __init__(self, pa):
        self.backlog = [None] * pa.backlog_size
        self.curr_size = 0


class JobRecord:
    def __init__(self):
        self.record = {}


class Machine:
    """
    Contains the resources used in job allocation. Responsible for 
    assigning resources to any jobs passed to it and advancing the
    state of its resources across timesteps.
    """
    def __init__(self, pa):
        # number of available queues in the machine
        self.num_q = pa.num_q
        # the length of each queue (measured in time)
        self.time_horizon = pa.time_horizon
        # vector whose i'th entry represents the free slots in the i'th queue
        self.avbl_slot = np.ones(self.num_q) * self.time_horizon
        # list of allocated jobs stil running
        self.running_job = []

        # colormap for graphical representation
        self.colormap = np.arange(1 / float(pa.q_num_cap), 1,
                                  1 / float(pa.q_num_cap))
        np.random.shuffle(self.colormap)

        # graphical representation
        self.canvas = np.zeros((pa.time_horizon, pa.num_q))

    def allocate_job(self, job, q_idx, curr_time):
        """ 
        Allocates job in the queue indicated by q_idx
        """

        allocated = False

        # allocate the job if there's room for it
        if self.avbl_slot[q_idx] >=  job.len:
            # update the job's start and stop times
            job_delay = self.time_horizon - self.avbl_slot[q_idx]
            job.start_time = curr_time + job_delay
            job.finish_time = job.start_time + job.len

            # update the available slots in queue q_idx
            self.avbl_slot[q_idx] -= job.len

            # add to the machine's list of running jobs
            self.running_job.append(job)

            # update graphical representation
            used_colors = np.unique(self.canvas[:])
            for color in self.colormap:
                if color not in used_colors:
                    new_color = color
                    break

            canvas_start_time = int(job.start_time - curr_time)
            canvas_end_time = int(job.finish_time - curr_time)
            for t in range(canvas_start_time, canvas_end_time):
                self.canvas[t, q_idx] = new_color

            # indicate that job was successfully allocated
            allocated = True

        return allocated


    def time_proceed(self, curr_time):
        """
        Advances machine's state forward in time by shifting all resource
        slots up a row and resetting the bottom resource slot
        """
        # free a timeslot for each queue
        for i in range(len(self.avbl_slot)):
            self.avbl_slot[i] += (self.avbl_slot[i] < self.time_horizon)

        # remove running jobs that have terminated
        for job in self.running_job:
            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation
        self.canvas[:-1, :] = self.canvas[1:, :]
        self.canvas[-1, :] = 0


class ExtraInfo:
    """ 
    Miscillaneous environment properties, including the time since 
    the last new job arrived.
    """
    def __init__(self, pa):
        self.time_since_last_new_job = 0
        self.max_tracking_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_tracking_time_since_last_job:
            self.time_since_last_new_job += 1


# ==========================================================================
# ------------------------------- Unit Tests -------------------------------
# ==========================================================================


def test_backlog():
    pa = parameters.Parameters()
    pa.num_nw = 5
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 1
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    env.step(5)
    assert env.job_backlog.backlog[0] is not None
    assert env.job_backlog.backlog[1] is None
    print "New job is backlogged."

    env.step(5)
    env.step(5)
    env.step(5)
    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(0)
    assert env.job_slot.slot[0] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    job = env.job_backlog.backlog[0]
    env.step(1)
    assert env.job_slot.slot[1] == job

    env.step(5)

    job = env.job_backlog.backlog[0]
    env.step(3)
    assert env.job_slot.slot[3] == job

    print "- Backlog test passed -"


def test_compact_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='compact')

    import other_agents
    import time

    start_time = time.time()
    for i in xrange(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print "- Elapsed time: ", end_time - start_time, "sec -"


def test_image_speed():

    pa = parameters.Parameters()
    pa.simu_len = 50
    pa.num_ex = 10
    pa.new_job_rate = 0.3
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image')

    import other_agents
    import time

    start_time = time.time()
    for i in xrange(100000):
        a = other_agents.get_sjf_action(env.machine, env.job_slot)
        env.step(a)
    end_time = time.time()
    print "- Elapsed time: ", end_time - start_time, "sec -"


if __name__ == '__main__':
    test_backlog()
    test_compact_speed()
    test_image_speed()
