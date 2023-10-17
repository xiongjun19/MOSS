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
        # self.c1 = tf.Variable(tf.truncated_normal([64, 64], stddev=0.1,
        #     seed=7))
        # self.c2 = tf.Variable(tf.zeros([64, 64]))

    def run_forward(self, embed):
        hid = tf.matmul(embed, self.B)
        hid = tf.matmul(hid, self.A)
        return hid

    # def run_forward(self, embed2):
    #     hid2 = tf.matmul(embed2, self.c1)
    #     hid2 = tf.matmul(hid2, self.c2)
    #     return hid2


def save_model(model_path):
    r = 64 
    hid_dim = 6144 
    model = LoRaModel(r, hid_dim)
    input_rnd = tf.placeholder(tf.float32, shape=[None, hid_dim])
    # input_rnd2 = tf.placeholder(tf.float32, shape=[None, 512])
    # input_rnd2 = tf.Variable(tf.ones([64, 64]))
    # hid_state, out_2 =model.run_forward(input_rnd, input_rnd2)
    hid_state = model.run_forward(input_rnd)
    init_op = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter('./graphs', graph=sess.graph)
        tf.saved_model.simple_save(sess,
                                   model_path,
                                   inputs = {'embed': input_rnd, },
                                   outputs = {'hid_state': hid_state,
                                              #'dag_out': out_2,
                                             }
                                  )
        writer.flush()
        writer.close()


if __name__ == '__main__':
    model_path = 'moss_lora_v2'
    save_model(model_path)
