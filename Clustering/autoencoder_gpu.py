import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from csv_utils import encode_csv

def decode_bad_csv(filename):
    names = [str(i) for i in range(98305)]
    data = pd.read_csv(filename, header=None, names=names)
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

SHRINK_SZ = 1000
ENC_SZ = 100

pie_vec = decode_bad_csv('./pie_prepool.csv')
sushi_vec = decode_bad_csv('./sushi_prepool.csv')
all_vec = np.vstack((pie_vec, sushi_vec))

with tf.device('/device:GPU:1'):
    training_data = tf.constant(all_vec, dtype=tf.float32, name="Input")

    relu = tf.nn.relu

    n, m = all_vec.shape

    minibatches = []
    '''
    a_v = np.random.shuffle(all_vec)
    for i in range(0, n - 200, 200):
        print(i)
        minibatches.append(a_v[i:(i + 200), :])
    '''
    # prepool_vectors = tf.placeholder(dtype=tf.float32, shape = (None, INPUT_SZ))
    # prepool_vectors = tf.placeholder_with_default(
    #     input = training_data,
    #     shape = (None, INPUT_SZ)
    # )
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
    saver = tf.train.Saver()
    for i in range(1000):
        #batch = minibatches[i % len(minibatches)]
        _, lsc = sess.run([train_step, least_squares_cost])
        print(lsc)
        #if i % 100 == 0:
        #    saver.save(sess, './autoencoder') 
    vects = sess.run(encoder)
    encode_csv('./autoencoder_pie_sushi.csv', vects)
    saver.save(sess, './autoencoder')
