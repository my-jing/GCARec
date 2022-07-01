'''
Author: Jingmy
Date: 2022-01-05 09:11:08
LastEditors: jingmy
LastEditTime: 2022-06-20 09:36:15
'''
import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    conf = Configurator("./NeuRec.properties", default_section="hyperparameters")

    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]
    dataset = Dataset(conf)
    config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        my_module = importlib.import_module("model.general_recommender." + recommender)
            
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.global_variables_initializer())
        model.train_model()
