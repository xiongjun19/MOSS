# coding=utf8


import os
import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True



'''
self.conn = redis.Redis(host='localhost', port=6379, db=0)
'''

class LoRaModel(object):
    def __init__(self, r, hidden):
        self.B = tf.Variable(tf.truncated_normal([hidden, r], stddev=0.1,
            seed=7))
        self.A = tf.Variable(tf.zeros([r, hidden]))

    def run_forward(self, embed):
        hid = tf.matmul(embed, self.B)
        hid = tf.matmul(hid, self.A)
        return hid


def save_model(model_path):
    r = 8
    hid_dim = 6144 
    model = LoRaModel(r, hid_dim)
    input_rnd = tf.placeholder(tf.float32, shape=[None, None, hid_dim])
    hid_state =model.run_forward(input_rnd)
    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('./graphs', graph=sess.graph)
        tf.saved_model.simple_save(sess,
                                   model_path,
                                   inputs = {'embed': input_rnd},
                                   outputs = {'hid_state': hid_state})
        writer.flush()
        writer.close()


if __name__ == '__main__':
    model_path = 'moss_lora'
    save_model(model_path)
