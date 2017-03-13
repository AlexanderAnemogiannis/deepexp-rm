import time
import threading
import numpy as np
import theano
import cPickle
import matplotlib.pyplot as plt

from multiprocessing import Process
from multiprocessing import Manager

import environment
import job_distribution
import pg_network
import slow_down_cdf

from collections import deque
from adversarial_q_learner import AdversarialQLearner, build_q_learner
import tensorflow as tf

### get discounted reward for sequence of rewards x, discount factor gamma
def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(xrange(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

#### TODO: rewrite by inserting DQN

### simulate episode_max_length episodes with the provided policy gradient 
### learner (agent) and simulation environment (env). Return the resulting
### states (obs), actions (acts), rewards(rews) for all trajectories.
### Note: states are characterized by an image representation
def get_traj(agent, env, episode_max_length, train=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """
    env.reset()
    rews = []
    info = []

    ob = env.observe()
    
    for _ in xrange(episode_max_length):

        # note: we flatten the image
        state = np.array(list(np.array(ob).flat))
        a = agent.e_greedy_policy(state[np.newaxis, :])

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if train:
            # note: we flatten the image
            next_state = np.array(list(np.array(ob).flat))

            # store the experience
            agent.store_experience(state, a, rew, next_state, done)

            # train the model
            agent.updateModel()

        if done: break

    return {'reward': np.array(rews),
            'info': info
            }

def process_all_info(trajs):
    enter_time = []
    finish_time = []
    job_len = []

    for traj in trajs:
        enter_time.append(np.array([traj['info'].record[i].enter_time for i in xrange(len(traj['info'].record))]))
        finish_time.append(np.array([traj['info'].record[i].finish_time for i in xrange(len(traj['info'].record))]))
        job_len.append(np.array([traj['info'].record[i].len for i in xrange(len(traj['info'].record))]))

    enter_time = np.concatenate(enter_time)
    finish_time = np.concatenate(finish_time)
    job_len = np.concatenate(job_len)

    return enter_time, finish_time, job_len


def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                  ref_discount_rews, ref_slow_down):
    num_colors = len(ref_discount_rews) + 2
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(122)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    ax.plot(slow_down_lr_curve, linewidth=2, label='PG mean')
    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(np.concatenate(ref_slow_down[k])), len(slow_down_lr_curve)), linewidth=2, label=k)

    plt.legend(loc=1)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Slowdown", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")


###  generate/evaluate trajectories for  policy given by pg_learner
def get_traj_worker(q_learner, env, pa, result):

    trajs = []

    ### generate instances of trajectories for policy given by NN (pg_learner)
    ### traj characterized by obs (images), rewards, actions
    for i in xrange(pa.num_seq_per_batch):
        traj = get_traj(q_learner, env, pa.episode_max_length)
        trajs.append(traj)

    ### begin processing trajectory results

    # Compute discounted sums of rewards
    rets = [discount(traj["reward"], pa.discount) for traj in trajs]

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])  # episode total rewards
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])  # episode lengths

    # All Job Stat
    enter_time, finish_time, job_len = process_all_info(trajs)
    finished_idx = (finish_time >= 0)
    all_slowdown = (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]

    result.append({"all_eprews": all_eprews,
                   "all_eplens": all_eplens,
                   "all_slowdown": all_slowdown})


def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    # dimension of state space
    # NOTE: we have to flatten the images before sending them into the network...
    state_dim   = pa.network_input_height * pa.network_input_width

    # number of actions
    num_actions = pa.network_output_dim

    # initialize the q networks
    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    q_learner = AdversarialQLearner(session=sess, optimizer=optimizer, q_network=build_q_learner, 
                                    state_dim=state_dim, num_actions=num_actions)

    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    ### create sequence of environments for each of the num_ex job sets/sequences
    for ex in xrange(pa.num_ex):

        print "-prepare for env-", ex

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    # ### generate sequence of NNs for each batch, each of which is a a policy gradient agent
    # for ex in xrange(pa.batch_size + 1):  # last worker for updating the parameters

    #     print "-prepare for worker-", ex

    #     pg_learner = pg_network.PGLearner(pa)

    #     if pg_resume is not None:
    #         net_handle = open(pg_resume, 'rb')
    #         net_params = cPickle.load(net_handle)
    #         pg_learner.set_net_params(net_params)

    #     pg_learners.append(pg_learner)

    # accums = init_accums(pg_learners[pa.batch_size])

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews, ref_slow_down = slow_down_cdf.launch(pa, pg_resume=None, render=False, plot=False, repre=repre, end=end)
    mean_rew_lr_curve = []
    max_rew_lr_curve = []
    slow_down_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in xrange(1, pa.num_epochs):

        ### use a thread for each use manager to share results across threads
        # ps = []  # threads
        # manager = Manager()  # managing return results
        # manager_result = manager.list([])

        ex_indices = range(pa.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        loss_all = []
        eprews = []
        eplens = []
        all_slowdown = []

        ex_counter = 0

        ### for each jobset
        for ex in xrange(pa.num_ex):

            ex_idx = ex_indices[ex]

            current_env = envs[ex_idx]

            man_result = []
            get_traj_worker(q_learner, current_env, pa, man_result)

            ### evaluate several instances of trajectories for set of PG agents
            # p = Process(target=get_traj_worker,
            #             args=(pg_learners[ex_counter], envs[ex_idx], pa, manager_result, ))
            # ps.append(p)

            ex_counter += 1

            all_eprews.extend([r["all_eprews"] for r in man_result])
            eprews.extend(np.concatenate([r["all_eprews"] for r in man_result]))  # episode total rewards
            eplens.extend(np.concatenate([r["all_eplens"] for r in man_result]))  # episode lengths

            all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in man_result]))
            
            ##

            # if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

                # print ex, "out of", pa.num_ex

                # ex_counter = 0

                # for p in ps:
                #     p.start()

                # for p in ps:
                #     p.join()

                # result = []  # convert list from shared memory
                # for r in manager_result:
                #     result.append(r)

                # ps = []
                # manager_result = manager.list([])

                # all_ob = concatenate_all_ob_across_examples([r["all_ob"] for r in result], pa)
                # all_action = np.concatenate([r["all_action"] for r in result])
                # all_adv = np.concatenate([r["all_adv"] for r in result])

                # all_eprews.extend([r["all_eprews"] for r in result])

                # eprews.extend(np.concatenate([r["all_eprews"] for r in result]))  # episode total rewards
                # eplens.extend(np.concatenate([r["all_eplens"] for r in result]))  # episode lengths

                # all_slowdown.extend(np.concatenate([r["all_slowdown"] for r in result]))

        # # assemble gradients
        # grads = grads_all[0]
        # for i in xrange(1, len(grads_all)):
        #     for j in xrange(len(grads)):
        #         grads[j] += grads_all[i][j]

        # # propagate network parameters to others
        # params = pg_learners[pa.batch_size].get_params()

        # rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

        # for i in xrange(pa.batch_size + 1):
        #     pg_learners[i].set_net_params(params)

        timer_end = time.time()

        print "-----------------"
        print "Iteration: \t %i" % iteration
        print "NumTrajs: \t %i" % len(eprews)
        print "NumTimesteps: \t %i" % np.sum(eplens)
        # print "Loss:     \t %s" % np.mean(loss_all)
        print "MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews])
        print "MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews))
        print "MeanSlowdown: \t %s" % np.mean(all_slowdown)
        print "MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens))
        print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        print "-----------------"

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            # param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            # cPickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
            # param_file.close()

            pa.unseen = True
            slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end, q_resume=q_learner)
            pa.unseen = False
            # test on unseen examples

            plot_lr_curve(pa.output_filename,
                          max_rew_lr_curve, mean_rew_lr_curve, slow_down_lr_curve,
                          ref_discount_rews, ref_slow_down)


def main():

    import parameters

    pa = parameters.Parameters()

    pa.simu_len = 50  # 1000
    pa.num_ex = 50  # 100
    pa.num_nw = 10
    pa.num_seq_per_batch = 20
    pa.output_freq = 50
    pa.batch_size = 10

    # pa.max_nw_size = 5
    # pa.job_len = 5
    pa.new_job_rate = 0.3

    pa.episode_max_length = 2000  # 2000

    pa.compute_dependent_parameters()

    pg_resume = None
    # pg_resume = 'data/tmp_450.pkl'

    render = False

    launch(pa, pg_resume, render, repre='image', end='all_done')


if __name__ == '__main__':
    main()
