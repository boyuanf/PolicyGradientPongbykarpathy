from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './',
                           """Directory where to write event logs and checkpoint. """)
tf.app.flags.DEFINE_string('checkpoint_file', './',
                           """Name of the checkpoint file """)
tf.app.flags.DEFINE_integer('num_epochs ', 10000,
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

# TO start the tensorboard: tensorboard: 1.5.1, tensorflow: 1.5.0
# python -m tensorboard.main --logdir=C:\Boyuan\MyPython\MNIST_Dataset





def train():
    pass

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()