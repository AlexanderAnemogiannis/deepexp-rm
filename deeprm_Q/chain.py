import numpy as np
import tensorflow as tf
import gym
from collections import deque
from deep_q_learner import DeepQLearner
from deep_booty_learner import DeepBootyLearner

# define the q network
# def get_q_network(input_states, state_dim, num_actions):
#     W1 = tf.get_variable("W1", [state_dim, 20],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     b1 = tf.get_variable("b1", [20],
#                          initializer=tf.constant_initializer(0))
#     h1 = tf.nn.relu(tf.matmul(input_states, W1) + b1)
#     W2 = tf.get_variable("W2", [20, num_actions],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     b2 = tf.get_variable("b2", [num_actions],
#                          initializer=tf.constant_initializer(0))
#     q = tf.matmul(h1, W2) + b2
#     return q

def get_q_network(input_states, state_dim, num_actions):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(input_states, W1) + b1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h1, W2) + b2
    return q

def run_simulation(env, session, deep_qnet, max_episodes, max_steps, render=True, debug=True, train=True, state_is_num=False, use_booty=False):
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
        state = env.reset()

        # keep track of total rewards during this episode
        total_rewards = 0 

        if use_booty:
            learner_ind = np.random.randint(0, deep_qnet.K)

        for t in range(max_steps):
            if render:
                env.render()

            if train:
                # take next action according to eps-greedy policy
                if state_is_num:
                    state = np.array([state])

                if use_booty:
                    action = deep_qnet.greedy_policy(state[np.newaxis, :], learner_ind)
                else:
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

            # advance to next state
            state = next_state

            if done:
                break

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)
        std_rewards = np.std(episode_history)

        if debug:
            print("Episode {}".format(ep))
            print("Finished after {} timesteps".format(t+1))
            print("Reward for this episode: {}".format(total_rewards))
            print("Average reward for last 100 episodes: {}".format(mean_rewards))
            print("Std dev reward for last 100 episodes: {}".format(std_rewards))

        # if train and mean_rewards >= 195.0:
        #     #print("Environment {} solved after {} episodes".format(env_name, ep+1))
        #     print("Solved after {} episodes".format(ep + 1))
        #     return (ep + 1)

    if train:
        return max_episodes
    else:
        return episode_history


if __name__ == "__main__":

    NUM_TRIALS = 1
    MAX_EPISODES = 10000
    NUM_TEST_EPISODES = 1000
    MAX_STEPS = 1000
    EPS = 0.1

    # the simulator
    env_name = 'NChain-v0'
    env = gym.make(env_name)

    # dimension of state space
    state_dim   = 1 #env.observation_space.n

    # number of actions
    num_actions = env.action_space.n

    # initialize the q networks
    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    # deep_qnet = DeepQLearner(session=sess, optimizer=optimizer, q_network=get_q_network, state_dim=state_dim, 
    #                                 num_actions=num_actions)

    # run_simulation(env=env, session=sess, deep_qnet=deep_qnet, max_episodes=MAX_EPISODES, 
    #                max_steps=MAX_STEPS, render=False, debug=True, train=True, state_is_num=True, use_booty=False)

    deep_qnet = DeepBootyLearner(session=sess, optimizer=optimizer, q_network=get_q_network, state_dim=state_dim, 
                                 num_actions=num_actions, K=2, var=2.0)

    run_simulation(env=env, session=sess, deep_qnet=deep_qnet, max_episodes=MAX_EPISODES, 
                   max_steps=MAX_STEPS, render=False, debug=True, train=True, state_is_num=True, use_booty=True)


