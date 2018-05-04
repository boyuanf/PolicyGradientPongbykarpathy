from datetime import datetime
import os.path
import re
import time

import numpy as np
import gym
import tensorflow as tf
from tensorflow.python.framework import ops

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './',
                           """Directory where to write event logs and checkpoint. """)
tf.app.flags.DEFINE_string('checkpoint_file', './',
                           """Name of the checkpoint file """)
tf.app.flags.DEFINE_integer('num_episode', 10000,
                            """number of epochs of the optimization loop.""")
tf.app.flags.DEFINE_integer('layer1_unit_num', 200,
                            """Number of the hidden unit in layer1.""")
tf.app.flags.DEFINE_float('regularizer_scale', 0.01,
                            """L1 regularizer scale.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """every how many episodes to do a param update.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('gamma', 0.99,
                            """discount factor for reward.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume from previous checkpoint.""")
tf.app.flags.DEFINE_boolean('render', False,
                            """Whether to display the game.""")

input_size = 80 * 80  # input dimensionality: 80x80 grid
observation_size = 210 * 160 * 3  # observation dimensionality: 210x160x3

# TO start the tensorboard: tensorboard: 1.5.1, tensorflow: 1.5.0
# python -m tensorboard.main --logdir=C:\Boyuan\MyPython\MNIST_Dataset

def create_placeholders(input_size):
    """
        Creates the placeholders for the tensorflow session.

        Arguments:
        input_size -- scalar, size of an image vector (num_px * num_px = 80 * 80 = 6400)
    """
    X = tf.placeholder(tf.float32, shape=(None, input_size), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
    R = tf.placeholder(tf.float32, shape=(None, 1), name="Z2")
    return X, Y, R

def forward_propagation(X):
    """
        Implements the forward propagation for the model
        Arguments:
        X -- input dataset placeholder, of shape (number of examples, input size)
        Returns:
        Z2 -- the output of the last LINEAR unit
    """
    he_init = tf.contrib.layers.variance_scaling_initializer()
    l1_regularizer = tf.contrib.layers.l1_regularizer(FLAGS.regularizer_scale)
    Z1 = tf.layers.dense(inputs=X, units=FLAGS.layer1_unit_num, kernel_initializer=he_init,
                        kernel_regularizer=l1_regularizer, name="layer_1")
    A1 = tf.nn.relu(Z1)
    Z2 = tf.layers.dense(inputs=A1, units=1, kernel_initializer=he_init, name="layer_2")
    A2 = tf.nn.sigmoid(Z2)
    return Z2, A2


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    """ Explained in 'More general advantage functions' section """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * FLAGS.gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def normalize_rewards(R):
    mean, var = tf.nn.moments(R, axes=[0])
    R = tf.subtract(R, mean)
    R = tf.divide(R, var)
    return R


def compute_cost(Z2, Y, Rewards):
    """
    Computes the cost

    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z2

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.sigmoid_cross_entropy_with_logits(...,...)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y)
    cost = tf.reduce_mean(tf.multiply(Rewards, cross_entropy, name="rewards"))
    return cost

# get an input image
def prepro(I, new_ep):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2, only take the first color channel
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    tf.get_variable_scope().reuse_variables()
    prev_x = tf.get_variable("prev_x")  # shared
    cur_x = tf.cast(tf.reshape(I, [1, input_size]), tf.float32)
    if new_ep:
        x = tf.zeros([1, input_size], tf.float32)
        new_ep = False
    else:
        x = tf.subtract(cur_x, prev_x)
    prev_x_ops = tf.assign(prev_x, cur_x)
    return x, new_ep, prev_x_ops


def train():
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    X, Y, Reward = create_placeholders(input_size)
    env = gym.make("Pong-v0")
    observation = env.reset()
    running_reward = None
    reward_sum = 0
    episode_number = 0
    R, X_list, Y_list = [], [], []
    # Forward propagation
    Z2, A2 = forward_propagation(X)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z2, Y, Reward)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
    prev_x = tf.get_variable("prev_x", [1, input_size], dtype=tf.float32,
                             initializer=tf.zeros_initializer)  # used in computing the difference frame

    # Initialize all the variables
    init = tf.global_variables_initializer()
    new_ep = True

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for ep in range(FLAGS.num_episode):

            if FLAGS.render:
                env.render()
                time.sleep(0.01)

            # preprocess the observation, set input to network to be difference image
            x, new_ep, prev_x_ops = prepro(observation, new_ep)
            x_eval, _ = sess.run([x, prev_x_ops])

            A2_eval = sess.run([A2], feed_dict={X: x_eval})

            if np.random.uniform() < A2_eval[0][0][0]:
                action = 2
                y = 1
            else:
                action = 3
                y = 0

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            R.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
            Y_list.append(y)
            X_list.append(x_eval)

            if done:  # an episode finished
                episode_number += 1
                # stack input and intermediate result
                ep_R = np.vstack(R)
                ep_Y = np.vstack(Y_list)
                ep_X = np.vstack(X_list)

                # compute the discounted reward backwards through time
                discounted_ep_R = discount_rewards(ep_R)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                normal_ep_R = normalize_rewards(tf.cast(discounted_ep_R, tf.float32))
                reward_eval = sess.run(normal_ep_R)

                _, cost_eval = sess.run([optimizer, cost], feed_dict={X: ep_X, Y: ep_Y, Reward: reward_eval})

                R, X_list, Y_list = [], [], []

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                print('cost is: ', cost_eval)

                reward_sum = 0
                observation = env.reset()  # reset env
                new_ep = True

            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))



def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()