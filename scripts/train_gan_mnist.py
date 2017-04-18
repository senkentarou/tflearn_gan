# coding: utf-8

#
# train_gan_mnist.py
#  GANを学習するスクリプト
#  Tensorflow >= 1.0.0, tflearn >= 0.3.0
#

import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist
from skimage import io

import sys
import os

SCRIPT_NAME = "gan_mnist"
HOME_DIR = ".."  # os.path.expanduser("~")
MODEL_DIR = "%s/out_models/%s" % (HOME_DIR, SCRIPT_NAME)
MNIST_DIR = "%s/datasets/mnist" % HOME_DIR

LOG_DIR = "%s/logs" % MODEL_DIR
TENSORBOARD_DIR = "%s/models" % MODEL_DIR
CHECKPOINT_DIR = "%s/models/model" % MODEL_DIR

G_SCOPE = "G"
G_HIDDEN_SIZE = 256
G_OUTPUT_SIZE = 784
D_SCOPE = "D"
D_HIDDEN_SIZE = 256
D_OUTPUT_SIZE = 1

X_SIZE = 784
Z_SIZE = 200

SAMPLE_NUM = 50000
BATCH_SIZE = 32
N_EPOCH = 100

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(TENSORBOARD_DIR):
    os.makedirs(TENSORBOARD_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(MNIST_DIR):
    os.makedirs(MNIST_DIR)


def build_generator(inputs, scope="G", reuse=False):

    """
     Generative Model 構築

     args:
        inputs: ネットワークの入力(入力層)
        scope: ネットワークの名前
        reuse: 名前空間の再利用判定
            true: 再利用する(ネットワークの重み共有)
            false: 再利用しない

     return:
        generative model network
    """

    # Hidden Layer
    g_h = tflearn.fully_connected(inputs,
                                  G_HIDDEN_SIZE,
                                  activation="relu",
                                  reuse=reuse,
                                  scope="%s/g_hidden" % scope)
    
    # Output Layer
    g_o = tflearn.fully_connected(g_h,
                                  G_OUTPUT_SIZE,
                                  activation="sigmoid",
                                  reuse=reuse,
                                  scope="%s/g_output" % scope)

    return g_o


def build_discriminator(inputs, scope="D", reuse=False):

    """
     Discriminative Model 構築

     args:
        inputs: ネットワークの入力(入力層)
        scope: ネットワークの名前
        reuse: 名前空間の再利用判定
            true: 再利用する(ネットワークの重み共有)
            false: 再利用しない

     return:
        discriminative model network
    """

    # Hidden Layer
    d_h = tflearn.fully_connected(inputs,
                                  D_HIDDEN_SIZE,
                                  activation="relu",
                                  reuse=reuse,
                                  scope="%s/d_hidden" % scope)

    # Output Layer
    d_o = tflearn.fully_connected(d_h,
                                  D_OUTPUT_SIZE,
                                  activation="sigmoid",
                                  reuse=reuse,
                                  scope="%s/d_output" % scope)

    return d_o


def build_gan_trainer():

    target = None

    # Place Holder
    input_x = tflearn.input_data(shape=(None, X_SIZE), name="input_x")
    input_z = tflearn.input_data(shape=(None, Z_SIZE), name="input_z")

    # Generator
    G_sample = build_generator(input_z, scope=G_SCOPE)
    target = G_sample

    # Discriminator
    D_origin = build_discriminator(input_x, scope=D_SCOPE)
    D_fake = build_discriminator(G_sample, scope=D_SCOPE, reuse=True)

    # Loss
    D_loss = -tf.reduce_mean(tf.log(D_origin) + tf.log(1. - D_fake))
    G_loss = -tf.reduce_mean(tf.log(D_fake))

    # Optimizer
    G_opt = tflearn.Adam(learning_rate=0.001).get_tensor()
    D_opt = tflearn.Adam(learning_rate=0.001).get_tensor()

    # Vars
    G_vars = get_trainable_variables(G_SCOPE)
    D_vars = get_trainable_variables(D_SCOPE)

    # TrainOp
    G_train_op = tflearn.TrainOp(loss=G_loss,
                                 optimizer=G_opt,
                                 batch_size=BATCH_SIZE,
                                 trainable_vars=G_vars,
                                 name="Generator")
    
    D_train_op = tflearn.TrainOp(loss=D_loss,
                                 optimizer=D_opt,
                                 batch_size=BATCH_SIZE,
                                 trainable_vars=D_vars,
                                 name="Discriminator")

    # Trainer
    gan_trainer = tflearn.Trainer([D_train_op, G_train_op],
                                  tensorboard_dir=TENSORBOARD_DIR,
                                  max_checkpoints=1)

    return gan_trainer, target


def train_gan(x, z):

    # Train
    with tf.Graph().as_default():

        gan_trainer, target = build_gan_trainer()

        input_x = get_input_tensor_by_name("input_x")
        input_z = get_input_tensor_by_name("input_z")

        feed_dict = [{input_x: x, input_z: z},
                     {input_z: z}]

        gan_trainer.fit(feed_dict,
                        n_epoch=N_EPOCH,
                        run_id="GAN")

        # Save trained model
        gan_trainer.save(CHECKPOINT_DIR)
        np.save("%s/sample_z.npy" % CHECKPOINT_DIR, z)


def get_trainable_variables(scope):
    return [v for v in tflearn.get_all_trainable_variable()
            if scope + '/' in v.name]


def get_input_tensor_by_name(name):
    return tf.get_collection(tf.GraphKeys.INPUTS, scope=name)[0]


# メイン部
if __name__ == "__main__":

    print("%s: start" % SCRIPT_NAME)

    # 目的: MNIST文字生成
    X, Y, testX, testY = mnist.load_data(data_dir=MNIST_DIR)

    # ランダムで100次元
    Z = np.random.uniform(-1., 1., SAMPLE_NUM * Z_SIZE)

    # データを処理
    sample_X = X[:SAMPLE_NUM]
    sample_testX = testX[:SAMPLE_NUM]
    sample_X = np.reshape(sample_X, (SAMPLE_NUM, X_SIZE))
    sample_Z = np.reshape(Z, (SAMPLE_NUM, Z_SIZE))

    # Train
    train_gan(sample_X, sample_Z)

    print("%s: done" % SCRIPT_NAME)
