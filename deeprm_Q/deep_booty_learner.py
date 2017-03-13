import random
import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
from collections import deque
from deep_q_learner import DeepQLearner

class DeepBootyLearner(object):
    def __init__(self, session,
                       optimizer,
                       q_network,
                       state_dim,
                       num_actions,
                       batch_size=32,
                       replay_buffer_size=10000,
                       store_replay_every=5, # how frequent to store experience
                       discount_factor=0.9, # discount future rewards
                       target_update_rate=0.01,
                       name="DeepBootyLearner",
                       K=2,
                       var=1.0
                       ):
        """ Initializes the Bootstrapped Deep Q Learner.

            Args:
                session: A TensorFlow session.
                optimizer: A TensorFlow optimizer.
                q_network: A TensorFlow network that takes in a state and output the Q-values over
                           all actions. 
                state_dim: Dimension of states.
                num_actions: Number of actions.
                batch_size: Batch size for training with experience replay.
                replay_buffer_size: Size of replay buffer.
                store_replay_every: Frequency with which to store replay.
                discount_factor: For discounting future rewards.
                target_update_rate: For the slow update of the target network.
                name: Used to create a variable scope. Useful for creating multiple
                      networks.
                K: number of Q learners
                var: noise variance on rewards added to noisy replay buffers
        """
        self.session = session
        self.optimizer = optimizer
        self.q_network = q_network # tensorflow constructor for Q network
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.batch_size = batch_size

        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate

        self.name = name
        self.var = var # noise variance

        # Construct all of the q-learners.
        self.K = K
        self.q_learners = []
        with tf.variable_scope(self.name):
            for i in range(self.K):
                dqn = DeepQLearner(session=session,
                                   optimizer=optimizer,
                                   q_network=q_network,
                                   state_dim=state_dim,
                                   num_actions=num_actions,
                                   batch_size=batch_size,
                                   replay_buffer_size=replay_buffer_size,
                                   store_replay_every=store_replay_every,
                                   discount_factor=0.9,
                                   target_update_rate=0.01,
                                   name="DeepQLearner{}".format(i))

                self.q_learners.append(dqn)

        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()



    def store_experience(self, state, action, reward, next_state, done):
        """ 
        Adds an experience to the replay buffer.
        """
        for i in range(self.K):
            # add noise to the reward
            corrupt_reward = reward + np.random.normal(scale=self.var)
            self.q_learners[i].store_experience(state, action, corrupt_reward, next_state, done)

    def greedy_policy(self, states, learner_ind):
        """ 
        Executes the greedy policy. Useful for executing a learned agent.
        """
        return self.q_learners[learner_ind].greedy_policy(states)

    def updateModel(self):
        """ 
        Update the model by sampling a batch from the replay buffer and
        performing Q-learning updates on the network parameters.
        """
        for i in range(self.K):
            self.q_learners[i].updateModel()

    # saves the trained model
    def saveModel(self, name):
        self.saver.save(self.session, name)

    def restoreModel(self, name):
        self.saver.restore(self.session, './' + name)

    def reset(self):
        for i in range(self.K):
            self.q_learners[i].reset()




