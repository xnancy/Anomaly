import tensorflow as tf
import numpy as np
import csv
from csv_utils import encode_csv

def decode_bad_csv(filename):
    names = [str(i) for i in range(ncols)]
    data = pd.read_sv(filename, header=None, names=names)
    del data['0']
    val = data.values
    del data
    return val

INPUT_SZ = 98304

def decode_bad_csv_lines(filename, lines):
    ret = np.zeros((0, INPUT_SZ + 1))
    with open(filename, 'r') as csvfile:
        i = 0
        reader = csv.reader(csvfile)
        for row in reader:
            if i % 50 == 0:
                print(i)
            if i == lines:
                break
            i += 1
            ret = np.vstack((ret, row))
    return ret[:, 1:]

SHRINK_SZ = 10000
ENC_SZ = 1000

pie_vec = decode_bad_csv_lines('./pie_prepool.csv', 10)
sushi_vec = decode_bad_csv_lines('./sushi_prepool.csv', 10)
all_vec = np.vstack((pie_vec, sushi_vec))
training_data = tf.constant(all_vec, dtype=tf.float32, name="Input")

relu = tf.nn.relu

#prepool_vectors = tf.placeholder(dtype=tf.float32, shape = (None, INPUT_SZ))
prepool_vectors = training_data
fan_in = tf.layers.dense(inputs = prepool_vectors, units = SHRINK_SZ, activation=relu)
encoder = tf.layers.dense(inputs = fan_in, units = ENC_SZ, activation=relu)
fan_out = tf.layers.dense(inputs = encoder, units = SHRINK_SZ, activation= relu)
output = tf.layers.dense(inputs = fan_out, units = INPUT_SZ, activation=relu)

least_squares_cost = tf.reduce_mean(tf.pow(prepool_vectors - output, 2))

train_step = tf.train.AdamOptimizer(1e-4).minimize(least_squares_cost)
init_step = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_step)
    for i in range(1000):
        _, lsc = sess.run([train_step, least_squares_cost])
        print(lsc)
    vects = sess.run(encoder)
    encode_csv('./autoencoder_pie_sushi.csv', outputs)

