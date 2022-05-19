"""
Created on 21/06/2018
@author: Nicolai Pedersen
"""

import os


def path_init():
    data_path = data_path_init()
    return data_path


def base_dir_init():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def data_path_init():
    return os.path.normpath(base_dir_init() + '/data/')


def test_data_path_init():
    return os.path.normpath(base_dir_init() + '/tests/data/')


def video_path_init():
    return os.path.normpath(data_path_init() + '/videos/')


def log_path_init():
    return os.path.normpath(data_path_init() + '/logs/')


def grid_path_init():
    return os.path.normpath(data_path_init() + '/grid/')


def lrs3_path_init():
    return os.path.normpath(data_path_init() + '/lrs3/')


def lrs3_pretrain_path_init():
    return os.path.normpath(data_path_init() + '/lrs3_pretrain/')


def lrs3_test_path_init():
    return os.path.normpath(data_path_init() + '/lrs3_test/')


def model_path_init():
    return os.path.normpath(base_dir_init() + '/python_code/models/')


if __name__ == '__main__':
    base_dir = base_dir_init()
    print('====== Current dir ======')
    print(base_dir)
