# coding=utf8

import redis
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


def get_tensor_data(conn, s_key, d_key, retry_time=5):
    t_data = None
    shape_val = None
    for _ in range(retry_time):
        if t_data is None:
            t_data = conn.get(d_key)
            conn.delete(d_key)
        if shape_val is None:
            shape_val = conn.get(s_key)
            conn.delete(s_key)
        if t_data is not None and shape_val is not None:
            shape = tuple(np.frombuffer(shape_val, dtype=np.int64))
            res_tensor = np.frombuffer(t_data, dtype=np.float16).copy()
            res_tensor = res_tensor.reshape(shape)
            return res_tensor
        time.sleep(0.002)
    return None


def set_tensor_data(conn, s_key, shape, d_key, _data):
    _shape = np.array(shape, dtype=np.int64).tobytes()
    _arr = _data
    serialized_array = _arr.tobytes()
    conn.set(s_key, _shape)
    conn.set(d_key, serialized_array)


def infer(model_path):
    conn = redis.Redis(host='localhost', port=6379, db=0)
    with tf.compat.v1.Session(config=config) as sess:
        model = tf.saved_model.loader.load(sess, ["serve"], model_path)
        x_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['embed'].name
        x = tf.get_default_graph().get_tensor_by_name(x_name)
        y_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['hid_state'].name
        y = tf.get_default_graph().get_tensor_by_name(y_name)
        while True:
            embed = get_tensor_data(conn, 'emb_shape', 'emb_data', retry_time=1)
            if embed is not None:
                result = sess.run(y, feed_dict={x: embed})
                print("result type: ", type(result))
                print('result shape', result.shape)
                set_tensor_data(conn, 'lora_shape',  result.shape,
                        'lora_data', result)
            time.sleep(0.003)


if __name__ == '__main__':
    import sys
    model_path = sys.argv[1]
    infer(model_path)


