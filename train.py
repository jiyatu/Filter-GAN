import numpy as np
import tensorflow as tf
from Filter import VI_Block

# Data
tf.app.flags.DEFINE_string('input',           'data/***', 'Path to data')
tf.app.flags.DEFINE_integer('disize',         512,                      'dimention ')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.01,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         50,                    'Maximum epochs to pre-training ')
tf.app.flags.DEFINE_integer('epochs',         80,                  'Maximum epochs to adversarial-training ')
tf.app.flags.DEFINE_integer('N',              50,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
tf.app.flags.DEFINE_integer('batchsize',      128,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,                   'Number of state inputs for each sample')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            True,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          'your_path/tmp/filter_tf/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# symbolic input reward tensor where typically first channel is reward, second is the Value prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.disize, config.disize, config.ch_i])
# symbolic input batches of reward
S1 = tf.placeholder(tf.int32,   name="S1", shape=[None, config.statebatchsize])
# symbolic input batches of value
S2 = tf.placeholder(tf.int32,   name="S2", shape=[None, config.statebatchsize])
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

# Construct model (Value Iteration Network)
if (config.untied_weights):
    logits, nn = VI_Untied_Block(X, S1, S2, config)
else:
    logits, nn = VI_Block(X, S1, S2, config)

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

    # Test model
    correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))
    acc = accuracy.eval({X: Xtest, S1: S1test, S2: S2test, y: ytest})
    print(f'Accuracy: {100 * (1 - acc)}%')
