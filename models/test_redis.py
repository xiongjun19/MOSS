# coding=utf8


import redis
import torch
import time
import numpy as np


def get_tensor_data(conn, s_key, d_key, retry_time=5):
    t_data = None
    shape_val = None
    for i in range(retry_time):
        if t_data is None:
            t_data = conn.getdel(d_key)
        if shape_val is None:
            shape_val = conn.getdel(s_key)
        if t_data is not None and shape_val is not None:
            shape = tuple(np.frombuffer(shape_val, dtype=np.int64))
            res_tensor = torch.Tensor(np.frombuffer(t_data, dtype=np.float16).copy())
            res_tensor = res_tensor.reshape(shape)
            return res_tensor
        time.sleep(0.002)
    return None



def set_tensor_data(conn, s_key, shape,d_key, _data):
    _shape = np.array(shape, dtype=np.int64).tobytes()
    _arr = _data.cpu().numpy()
    serialized_array = _arr.tobytes()
    conn.set(s_key, _shape)
    conn.set(d_key, serialized_array)


if __name__ == '__main__':
    conn = redis.Redis(host='localhost', port=6379, db=0)
    s_key = 'emb_shape'
    d_key = 'emb_data'
    emb_shape = [1, 4, 8]
    emb_tensor = torch.randn(emb_shape, dtype=torch.float16)
    set_tensor_data(conn, s_key, emb_shape, d_key, emb_tensor)
    out_tensor = get_tensor_data(conn, s_key, d_key)
    print("difference of the tensor's are: ")
    print(out_tensor - emb_tensor)


