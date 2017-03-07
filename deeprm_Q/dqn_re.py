# q_re.py

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

def run_simulation(env, session, deep_qnet, max_episodes, max_steps, render=True, debug=True, adversarial=True, train=True):
    """
    
    Assumes that deep_qnet.reset() has been called.

    If training, returns number of episodes it took to train.
    If testing, returns the episode history for the last 100 episodes.

    """

    if train:
        # keep track of total rewards from last 100 episodes
        episode_history = deque(maxlen=100)
    else:
        episode_history = deque(maxlen=max_episodes)

    for ep in range(max_episodes):

        # reset the environment
        env.reset()

        state = env.observe()

        # keep track of total rewards during this episode
        total_rewards = 0 

        for t in range(max_steps):

            if train:
                # take next action according to eps-greedy policy
                action = deep_qnet.e_greedy_policy(state[np.newaxis, :])
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward

                # store the experience
                deep_qnet.store_experience(state, action, reward, next_state, done)

                # train the model
                deep_qnet.updateModel()

            else:
                # take next action greedily with respect to Q
                action = deep_qnet.greedy_policy(state[np.newaxis, :])
                next_state, reward, done, _ = env.step(action)
                total_rewards += reward


            # get adverserial state
            #if (adversarial) and (ep == 0) and (t == 0):
            if (adversarial) and (t == 0):
                adv_state = deep_qnet.get_adversarial_state(EPS, state, action, reward, next_state, done)
                env._set_state(adv_state)
                next_state = adv_state

            # advance to next state
            state = next_state

            if done:
                break

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        if debug:
            print("Episode {}".format(ep))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {}".format(mean_rewards))

        # TODO: when is environment solved? or just run for max num of iters..

        # if train and mean_rewards >= 195.0:
        #     #print("Environment {} solved after {} episodes".format(env_name, ep+1))
        #     print("Solved after {} episodes".format(ep + 1))
        #     return (ep + 1)

    if train:
        return max_episodes
    else:
        return episode_history

def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    print("Using DQN...")

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
    deep_qnet = AdversarialQLearner(session=sess, optimizer=optimizer, q_network=build_q_learner, 
                                    state_dim=state_dim, num_actions=num_actions)



    #Q_learners = []
    envs = []

    nw_len_seqs, nw_size_seqs = job_distribution.generate_sequence_work(pa, seed=42)

    ### create sequence of environments for each of the num_ex job sets/sequences
    for ex in xrange(pa.num_ex):

        print "-prepare for env-", ex

        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_size_seqs=nw_size_seqs,
                              render=False, repre=repre, end=end)
        env.seq_no = ex
        envs.append(env)

    ### TODO: when parallelizing, should use pa.batch_size?

    ### generate sequence of NNs for each batch, each of which is a a policy gradient agent
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

    ### TODO: put in training stuff here

    ### iterate over epochs
    for iteration in xrange(1, pa.num_epochs):

        ex_indices = range(pa.num_ex)
        np.random.shuffle(ex_indices)

        ex_counter = 0

        ### for each jobset
        for ex in xrange(pa.num_ex):

            ex_idx = ex_indices[ex]
            current_env = envs[ex_ids]

            # TODO: am I supposed to just give the agent different environments like this???
            episode_history = deque(maxlen=100)

            # TODO: note that we aren't using batch size...
            # TODO: how to integrate batch size...

            # iterate over episodes
            for ep in range(pa.num_seq_per_batch):

                # reset the environment
                env.reset()

                state = np.array(env.observe())

                # keep track of total rewards during this episode
                total_rewards = 0 

                for t in range(max_steps):

                    # take next action according to eps-greedy policy
                    action = deep_qnet.e_greedy_policy(state[np.newaxis, :])
                    next_state, reward, done, _ = env.step(action)
                    total_rewards += reward

                    # store the experience
                    deep_qnet.store_experience(state, action, reward, next_state, done)

                    # train the model
                    deep_qnet.updateModel()

                    # get adverserial state
                    #if (adversarial) and (ep == 0) and (t == 0):
                    if (adversarial) and (t == 0):
                        adv_state = deep_qnet.get_adversarial_state(EPS, state, action, reward, next_state, done)
                        env._set_state(adv_state)
                        next_state = adv_state

                    # advance to next state
                    state = next_state

                    if done:
                        break

                episode_history.append(total_rewards)
                mean_rewards = np.mean(episode_history)

                if debug:
                    print("Episode {}".format(ep))
                    print("Finished after {} timesteps".format(t+1))
                    print("Reward for this episode: {}".format(total_rewards))
                    print("Average reward for last 100 episodes: {}".format(mean_rewards))

                # TODO: when is environment solved? or just run for max num of iters..

                # if train and mean_rewards >= 195.0:
                #     #print("Environment {} solved after {} episodes".format(env_name, ep+1))
                #     print("Solved after {} episodes".format(ep + 1))
                #     return (ep + 1)

            if train:
                return max_episodes
            else:
                return episode_history


        # assemble gradients
        grads = grads_all[0]
        for i in xrange(1, len(grads_all)):
            for j in xrange(len(grads)):
                grads[j] += grads_all[i][j]

        # propagate network parameters to others
        params = pg_learners[pa.batch_size].get_params()

        rmsprop_updates_outside(grads, params, accums, pa.lr_rate, pa.rms_rho, pa.rms_eps)

        for i in xrange(pa.batch_size + 1):
            pg_learners[i].set_net_params(params)

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
        print "MeanEntropy \t %s" % (np.mean(all_entropy))
        print "Elapsed time\t %s" % (timer_end - timer_start), "seconds"
        print "-----------------"

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))
        slow_down_lr_curve.append(np.mean(all_slowdown))

        if iteration % pa.output_freq == 0:
            param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            cPickle.dump(pg_learners[pa.batch_size].get_params(), param_file, -1)
            param_file.close()

            pa.unseen = True
            slow_down_cdf.launch(pa, pa.output_filename + '_' + str(iteration) + '.pkl',
                                 render=False, plot=True, repre=repre, end=end)
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