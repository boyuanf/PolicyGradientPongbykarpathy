from datetime import datetime
import os.path
import re
import time

import numpy as np
import gym
import tensorflow as tf
from tensorflow.python.framework import ops

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tf_train_pong',
                           """Directory where to write event logs and checkpoint. """)
tf.app.flags.DEFINE_string('restore_file_path', '/home/ubuntu/PolicyGradientPongbykarpathy/tf_train_pong/run-20180507182051-checkpoint/pg_pong_model.ckpt',
                           """Path of the restore file """)
tf.app.flags.DEFINE_integer('num_episode', 10000,
                            """number of epochs of the optimization loop.""")
tf.app.flags.DEFINE_integer('layer1_unit_num', 200,
                            """Number of the hidden unit in layer1.""")
tf.app.flags.DEFINE_float('regularizer_scale', 0.01,
                            """L1 regularizer scale.""")
tf.app.flags.DEFINE_integer('batch_size', 8,
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
    with tf.name_scope("forward_propagation"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        l1_regularizer = tf.contrib.layers.l1_regularizer(FLAGS.regularizer_scale)
        Z1 = tf.layers.dense(inputs=X, units=FLAGS.layer1_unit_num, kernel_initializer=he_init,
                            kernel_regularizer=l1_regularizer, name="Z1")
        A1 = tf.nn.relu(Z1, name='A1')
        Z2 = tf.layers.dense(inputs=A1, units=1, kernel_initializer=he_init, name="Z2")
        A2 = tf.nn.sigmoid(Z2, name='A2')
        return Z2, A2


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    """ Explained in 'More general advantage functions' section """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * FLAGS.gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def normalize_rewards(R):
    with tf.name_scope("normalize_rewards"):
        mean, var = tf.nn.moments(R, axes=[0], name='MeanVar')
        R = tf.subtract(R, mean)
        R = tf.divide(R, var)
        return R


def compute_cost(Z2, Y, Rewards):
    """
    Computes the cost

    Arguments:
    Z2 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z2
    Rewards -- the rewards coressponding to each of the output, same shape as Z2

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.sigmoid_cross_entropy_with_logits(...,...)
    with tf.name_scope("cost"):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2, labels=Y, name='cross_entropy')
        cost = tf.reduce_mean(tf.multiply(Rewards, cross_entropy, name="cross_reward"), name='cross_cost')
        loss_summary = tf.summary.scalar('log_loss', cost)
        return cost, loss_summary


# This function is a bad example, by calling this function, new code will be created, which should not in the while loop
# get an input image
def pre_processing(I, new_ep):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    with tf.name_scope("pre_processing"):
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2, only take the first color channel
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        tf.get_variable_scope().reuse_variables()
        prev_x = tf.get_variable("prev_x")  # shared
        cur_x = tf.cast(tf.reshape(I, [1, input_size]), tf.float32)
        if new_ep:
            x = tf.zeros([1, input_size], tf.float32, name='set_x_0')
            new_ep = False
        else:
            x = tf.subtract(cur_x, prev_x, name='set_x_diff')
        prev_x_ops = tf.assign(prev_x, cur_x, name='update_prev')
        return x, new_ep, prev_x_ops


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2, only take the first color channel
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()  # flatten to 1D array

def train():
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    X, Y, Reward = create_placeholders(input_size)
    env = gym.make("Pong-v0")
    observation = env.reset()
    running_reward = None
    reward_sum = 0
    episode_number = 0
    R_list, X_list, Y_list, R_batch = [], [], [], []
    #prev_x = tf.get_variable("prev_x", [1, input_size], dtype=tf.float32,
    #                         initializer=tf.zeros_initializer)  # used in computing the difference frame

    prev_x = None

    with tf.name_scope("train"):
        # Forward propagation
        Z2, A2 = forward_propagation(X)
        # Cost function: Add cost function to tensorflow graph
        cost, loss_summary = compute_cost(Z2, Y, Reward)
        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, name='AdamOptimizer').minimize(cost)

    # Initialize all the variables
    with tf.name_scope("init"):
        init = tf.global_variables_initializer()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = FLAGS.train_dir
    log_dir = "{}/run-{}-log".format(root_logdir, now)
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    saver = tf.train.Saver()  # will only keep the latest 5 models
    merged_summary = tf.summary.merge_all()  # get all summary in the graph, and put them in collection

    # finalize the graph to avoid accidentally change
    tf.get_default_graph().finalize()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        if FLAGS.resume:
            saver.restore(sess, FLAGS.restore_file_path)
        else:
            # Run the initialization
            sess.run(init)

        # Do the training loop
        while episode_number < FLAGS.num_episode:

            if FLAGS.render:
                env.render()
                time.sleep(0.01)

            # preprocess the observation, set input to network to be difference image
            # x, new_ep, prev_x_ops = pre_processing(observation, new_ep)
            # x_eval, _ = sess.run([x, prev_x_ops])
            # A2_eval = sess.run([A2], feed_dict={X: x_eval})

            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(input_size)
            prev_x = cur_x
            x = x.reshape((1, x.shape[0]))
            A2_eval = sess.run([A2], feed_dict={X: x})

            if np.random.uniform() < A2_eval[0][0][0]:
                action = 2
                y = 1
            else:
                action = 3
                y = 0

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            R_list.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
            Y_list.append(y)
            # X_list.append(x_eval)
            X_list.append(x)

            if done:  # an episode finished
            #if True:  # debug
                episode_number += 1
                # compute the discounted reward backwards through time
                discounted_ep_R = discount_rewards(R_list)
                # better not create graph in while loop, otherwise the model is keep growing!
                # normal_ep_R = normalize_rewards(tf.cast(discounted_ep_R, tf.float32))
                # reward_eval = sess.run(normal_ep_R)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_ep_R -= np.mean(discounted_ep_R)
                discounted_ep_R /= np.std(discounted_ep_R)
                R_batch = np.concatenate((R_batch, discounted_ep_R), axis=0)
                R_list = []

                if episode_number % FLAGS.batch_size == 0:
                    # stack input and intermediate result
                    R_batch = R_batch.reshape(R_batch.shape[0], 1)
                    ep_Y = np.vstack(Y_list)
                    ep_X = np.vstack(X_list)

                    # add only loss_summary to summary
                    # _, cost_eval, summary_str = sess.run([optimizer, cost, loss_summary], feed_dict={X: ep_X, Y: ep_Y, Reward: ep_R})
                    _, cost_eval, summary_str = sess.run([optimizer, cost, merged_summary],
                                                         feed_dict={X: ep_X, Y: ep_Y, Reward: R_batch})
                    file_writer.add_summary(summary_str, global_step=episode_number)
                    print('cost is: ', cost_eval)
                    R_batch, X_list, Y_list = [], [], []

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                # Add user data to TensorBoard
                reward_mean_summary = tf.Summary(value=[tf.Summary.Value(tag="reward_mean", simple_value=running_reward)])
                file_writer.add_summary(reward_mean_summary, global_step=episode_number)
                # Save the model checkpoint periodically.
                if episode_number % 100 == 0 or (episode_number + 1) == FLAGS.num_episode:
                # if episode_number % 1 == 0 or (episode_number + 1) == FLAGS.num_episode:  # debug
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    check_point_dir = "{}/run-{}-checkpoint".format(root_logdir, now)
                    checkpoint_path = os.path.join(check_point_dir, 'pg_pong_model.ckpt')
                    saver.save(sess, checkpoint_path)

                reward_sum = 0
                observation = env.reset()  # reset env
                prev_x = None

            if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

    file_writer.close()


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
