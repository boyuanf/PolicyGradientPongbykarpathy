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
tf.app.flags.DEFINE_integer('layer1_hidden_num', 200,
                            """Number of the hidden unit in layer1.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """every how many episodes to do a param update.""")
tf.app.flags.DEFINE_integer('learning_rate', 1e-3,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('gamma', 0.99,
                            """discount factor for reward.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume from previous checkpoint.""")
tf.app.flags.DEFINE_boolean('render', False,
                            """Whether to display the game.""")

input_size = 80 * 80  # input dimensionality: 80x80 grid

# TO start the tensorboard: tensorboard: 1.5.1, tensorflow: 1.5.0
# python -m tensorboard.main --logdir=C:\Boyuan\MyPython\MNIST_Dataset

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
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    running_reward = None
    reward_sum = 0
    episode_number = 0
    # Initialize all the variables
    init = tf.global_variables_initializer()

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
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(input_size)
            prev_x = cur_x
            x = x.reshape((x.shape[0], 1))
            image = tf.cast(x, tf.float32)








def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()