import random
import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
from collections import deque


def build_q_learner(input_states, state_dim, num_actions):
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

class AdversarialQLearner(object):
    def __init__(self, session,
                       optimizer,
                       q_network,
                       state_dim,
                       num_actions,
                       batch_size=32,
                       init_exp=0.5,       # initial exploration prob
                       final_exp=0.1,      # final exploration prob
                       anneal_steps=10000, # N steps for annealing exploration 
                       replay_buffer_size=10000,
                       store_replay_every=5, # how frequent to store experience
                       discount_factor=0.9, # discount future rewards
                       target_update_rate=0.01,
                       adversarial_type=0
                       ):
        """ Initializes the Deep Q Network.

            Args:
                session: A TensorFlow session.
                optimizer: A TensorFlow optimizer.
                q_network: A TensorFlow network that takes in a state and output the Q-values over
                           all actions. 
                state_dim: Dimension of states.
                num_actions: Number of actions.
                batch_size: Batch size for training with experience replay.
                init_exp: Initial exploration probability for eps-greedy policy.
                final_exp: Final exploration probability for eps-greedy policy.
                anneal_steps: Number of steps to anneal from init_exp to final_exp.
                replay_buffer_size: Size of replay buffer.
                store_replay_every: Frequency with which to store replay.
                discount_factor: For discounting future rewards.
                target_update_rate: For the slow update of the target network.
                adversarial_type: 0 means adversarial with respect to CE loss, 1 is TD loss, 
                                  2 is random perturbation
        """
        self.session = session
        self.optimizer = optimizer
        self.q_network = q_network # tensorflow constructor for Q network
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.batch_size = batch_size

        # initialize exploration
        self.exploration = init_exp
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps

        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate

        # Initialize the replay buffer.
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.store_replay_every = store_replay_every
        self.experience_cnt = 0

        self.adversarial_type = adversarial_type

        self.train_iteration = 0
        self.constructModel()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def constructModel(self):
        """ Constructs the model to do Q-learning.
        """

        # this part of the model is for predicting actions using the learned Q_network.
        with tf.name_scope("predict_actions"):

            # input: vectors of states (in a batch)
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")

            # use new scope to differentiate this q_network from one used for target evaluation
            # note that this will differentiate the weights, for example "learn_q_network/W1"
            with tf.variable_scope("learn_q_network"):
                # the current q_network that we train
                self.action_scores = self.q_network(self.states, self.state_dim, self.num_actions)
            self.predicted_actions = tf.argmax(self.action_scores, axis=1, name="predicted_actions")

        # this part of the model is for estimating future rewards, to be used for the Q-learning
        # update for estimating the target Q-value.
        with tf.name_scope("estimate_future_rewards"):

            # input: vectors of next states (in a batch)
            self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")

            # input: binary inputs that indicate whether states are unfinished or terminal
            # this is important to compute the target and do the Bellman update correctly, since
            # it tells us whether to include the optimal Q value for the next state or not.
            self.unfinished_states_flags = tf.placeholder(tf.float32, (None,), name="unfinished_states_flags")

            # input: rewards from last state and action
            self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")

            # use new scope to differentiate this q_network from one we are training
            # note that this will differentiate the weights, for example "target_q_network/W1"
            with tf.variable_scope("target_q_network"):
                # the q_network used for evaluation
                self.eval_q_vals = self.q_network(self.next_states, self.state_dim, self.num_actions)

            # note that this term is only non-zero for a state if it is non-terminal
            # also note the use of stop_gradient to make sure we don't train this q_network
            self.best_future_q_vals = tf.reduce_max(tf.stop_gradient(self.eval_q_vals), axis=1) * self.unfinished_states_flags

            # future rewards given by Bellman equation
            self.future_rewards = self.rewards + self.discount_factor * self.best_future_q_vals

        # this part of the model is for computing the loss and gradients
        with tf.name_scope("loss"):
            # input: one-hot vectors that give the current actions to evaluate the loss for
            self.action_selects = tf.placeholder(tf.float32, (None, self.num_actions), name="action_select")

            # get Q-values for the actions that we took
            self.selected_action_scores = tf.reduce_sum(self.action_scores * self.action_selects, axis=1)

            # temporal difference loss
            self.td_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.future_rewards - self.selected_action_scores)))

            # cross-entropy loss for adversarial example generation
            self.cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.action_scores, self.action_selects))

            # TODO: regularization loss

            # TODO: gradient clipping

            self.train_op = self.optimizer.minimize(self.td_loss)

            # TODO: check if this is correct
            if self.adversarial_type == 0:
                self.input_gradients = tf.gradients(self.cross_entropy_loss, self.states)
            elif self.adversarial_type == 1:
                self.input_gradients = tf.gradients(self.td_loss, self.states)

        # this part of the model is for updating the target Q network
        with tf.name_scope("eval_q_network_update"):
            target_network_update = []
            # slowly update target network parameters with Q network parameters
            # we do this by grabbing all the parameters in both networks and manually defining
            # update operations
            self.q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="learn_q_network")
            self.target_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_q_network")
            for v_source, v_target in zip(self.q_network_variables, self.target_network_variables):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                target_network_update.append(update_op)
            # this groups all operations to run together
            # this operation will update all of the target Q network variables
            self.target_network_update = tf.group(*target_network_update)

    def store_experience(self, state, action, reward, next_state, done):
        """ 
        Adds an experience to the replay buffer.
        """
        if self.experience_cnt % self.store_replay_every == 0 or done:
            self.replay_buffer.add(state, action, reward, next_state, done)
        self.experience_cnt += 1

    def greedy_policy(self, states):
        """ 
        Executes the greedy policy. Useful for executing a learned agent.
        """
        return self.session.run(self.predicted_actions, {self.states: states})[0]


    def e_greedy_policy(self, states):
        """ 
        Executes the epsilon greedy policy. 
        """
        # with probability exploration, choose random action
        if random.random() < self.exploration:
            return random.randint(0, self.num_actions-1)
        # choose greedy action given by current Q network
        else:
            return self.greedy_policy(states)


    def annealExploration(self):
        """ 
        Anneals the exploration probability linearly with training iteration.
        """
        ratio = max((self.anneal_steps - self.train_iteration) / float(self.anneal_steps), 0)
        self.exploration = (self.init_exp- self.final_exp) * ratio + self.final_exp

    def updateModel(self):
        """ 
        Update the model by sampling a batch from the replay buffer and
        performing Q-learning updates on the network parameters.
        """

        # not enough experiences yet
        if self.replay_buffer.count() < self.batch_size:
            return

        # sample a random batch from the replay buffer
        batch = self.replay_buffer.getBatch(self.batch_size)

        # keep track of these inputs to the Q networks for the batch
        states                     = np.zeros((self.batch_size, self.state_dim))
        rewards                    = np.zeros((self.batch_size,))
        action_selects             = np.zeros((self.batch_size, self.num_actions))
        next_states                = np.zeros((self.batch_size, self.state_dim))
        unfinished_states_flags    = np.zeros((self.batch_size,))

        # train on the experiences in this batch
        for k, (s0, a, r, s1, done) in enumerate(batch):
            states[k] = s0
            rewards[k] = r
            action_selects[k][a] = 1
            # check terminal state
            if not done:
                next_states[k] = s1
                unfinished_states_flags[k] = 1

        # perform one update of training
        cost, _ = self.session.run([self.td_loss, self.train_op], {
          self.states : states,
          self.next_states : next_states,
          self.unfinished_states_flags : unfinished_states_flags,
          self.action_selects : action_selects,
          self.rewards : rewards
        })

        # update target network using learned Q-network
        self.session.run(self.target_network_update)

        self.annealExploration()
        self.train_iteration += 1

    def get_adversarial_state(self, eps, state, action, reward, next_state, done):
        """ 
        Return an adversarial state corresponding to a certain experience.
        The adversarial state is generated using the fast sign method.
        """

        states                     = np.zeros((1, self.state_dim))
        rewards                    = np.zeros((1,))
        action_selects             = np.zeros((1, self.num_actions))
        next_states                = np.zeros((1, self.state_dim))
        unfinished_states_flags    = np.zeros((1,))

        states[0] = state
        rewards[0] = reward
        action_selects[0][action] = 1
        # check terminal state
        if not done:
            next_states[0] = next_state
            unfinished_states_flags[0] = 1

        if self.adversarial_type < 2:
            # get gradients with respect to input
            input_grads = self.session.run(self.input_gradients, 
                                        feed_dict={ self.states : states,
                                                    self.next_states : next_states,
                                                    self.unfinished_states_flags : unfinished_states_flags,
                                                    self.action_selects : action_selects,
                                                    self.rewards : rewards })

            adv_state = state + eps * np.sign(input_grads[0][0])
        else:
            # a random, epsilon max-norm perturbation (we draw a random sign vector)
            adv_state = state + eps * (2.0 * np.random.binomial(1, 0.5, self.state_dim) - 1)

        # project into allowed state
        if adv_state[0] > 4.8:
            adv_state[0] = 4.8
            print('clipped adv_state[0] to 4.8')
        elif adv_state[0] < -4.8:
            adv_state[0] = -4.8
            print('clipped adv_state[0] to -4.8')
        if adv_state[2] > 0.41888:
            adv_state[2] = 0.41888
            print('clipped adv_state[2] to 0.41888')
        elif adv_state[2] < -0.41888:
            adv_state[2] = -0.41888
            print('clipped adv_state[2] to -0.41888')

        return adv_state

    # saves the trained model
    def saveModel(self, name):
        self.saver.save(self.session, name)

    def restoreModel(self, name):
        self.saver.restore(self.session, './' + name)

    def setAdversarialType(self, type):
        self.adversarial_type = type

    def reset(self):
        # initialize exploration
        self.exploration = self.init_exp

        # Initialize the replay buffer.
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.experience_cnt = 0

        self.train_iteration = 0
        self.session.run(tf.global_variables_initializer())




