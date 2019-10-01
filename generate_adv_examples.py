"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with Keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils import AccuracyReport, other_classes

import matplotlib.pyplot as plt
import imageio
import os
import os.path as osp

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001


class CNNModel:
    def __init__(self, dataset, model_type='cnn_model', label_smoothing=.1):
        self.x_train, self.y_train = dataset.get_set('train')
        self.x_test, self.y_test = dataset.get_set('test')

        # Obtain Image Parameters
        self.img_rows, self.img_cols, self.nchannels = self.x_train.shape[1:4]
        self.nb_classes = self.y_train.shape[1]

        # Label smoothing
        self.y_train -= label_smoothing * (self.y_train - 1. / self.nb_classes)

        # Define Keras model
        self.model = cnn_model(img_rows=self.img_rows, img_cols=self.img_cols,
                          channels=self.nchannels, nb_filters=64,
                          nb_classes=self.nb_classes)
        print("Defined Keras model.")

        # To be able to call the model in the custom loss, we need to call it once
        # before, see https://github.com/tensorflow/tensorflow/issues/23769
        self.model(self.model.input)

    def compile(self, metrics=['accuracy'], loss='categorical_crossentropy',
                learning_rate=LEARNING_RATE):
        self.model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                           loss=loss,
                           metrics=metrics)

    def fit(self, batch_size=BATCH_SIZE,
            epochs=NB_EPOCHS,
            verbose=2):
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(self.x_test, self.y_test),
                       verbose=verbose)

    def evaluate(self, batch_size=BATCH_SIZE,
                 verbose=0):
        loss, acc, adv_acc = self.model.evaluate(self.x_test, self.y_test,
                                                 batch_size=batch_size,
                                                 verbose=verbose)
        return loss, acc, adv_acc

    def predict(self, x):
        return self.model.predict(x)


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   learning_rate=LEARNING_RATE, testing=False,
                   label_smoothing=0.1):
    """
    MNIST CleverHans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param testing: if true, training error is calculated
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)
    # Force TensorFlow to use single thread to improve reproducibility
    # config = tf.ConfigProto(intra_op_parallelism_threads=1,
    #                         inter_op_parallelism_threads=1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if keras.backend.image_data_format() != 'channels_last':
        raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Get MNIST test data
    mnist = MNIST(train_start=train_start, train_end=train_end,
                  test_start=test_start, test_end=test_end)

    report = gen_adv_fast_gradient_method(sess, mnist)
    return report


def gen_adv_fast_gradient_method(sess, dataset, fgsm_params=None,
                                 testing=False, adv_range=range(0, 20), output_dir='./adv_output'):
    # Object used to keep track of (and return) key accuracies
    attack_name = "fast_gradient_method"
    report = AccuracyReport()
    model = CNNModel(dataset)

    # Initialize the Fast Gradient Sign Method (FGSM) attack object
    wrap = KerasModelWrapper(model.model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    if fgsm_params is None:
        fgsm_params = {'eps': 0.3, 'clip_min': 0., 'clip_max': 1., 'y_target': None}

    adv_acc_metric = get_adversarial_acc_metric(model.model, fgsm, fgsm_params)
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy', adv_acc_metric])

    # Train an MNIST model
    model.fit()

    # Evaluate the accuracy on legitimate and adversarial test examples
    _, acc, adv_acc = model.evaluate()
    report.clean_train_clean_eval = acc
    report.clean_train_adv_eval = adv_acc

    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    for sample_ind in adv_range:
        sample = model.x_test[sample_ind:(sample_ind + 1)]
        current_class = int(np.argmax(model.y_test[sample_ind]))
        target_classes = other_classes(model.nb_classes, current_class)
        if not osp.isdir(osp.join(output_dir, attack_name)):
            os.mkdir(osp.join(output_dir, attack_name))
        fn = osp.join(output_dir, attack_name, str(sample_ind) + "_input.tiff")
        imageio.imwrite(fn, np.reshape(sample, (model.img_rows, model.img_cols)))
        for target in target_classes:
            one_hot_target = np.zeros((1, model.nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            fgsm_params['y_target'] = one_hot_target
            adv_x = fgsm.generate_np(sample, **fgsm_params)
            fn = osp.join(output_dir, attack_name, str(sample_ind) + "_adv{}.tiff".format(target))
            imageio.imwrite(fn, np.reshape(adv_x, (model.img_rows, model.img_cols)))

    # Calculate training error
    if testing:
        _, train_acc, train_adv_acc = model.evaluate()
        report.train_clean_train_clean_eval = train_acc
        report.train_clean_train_adv_eval = train_adv_acc

    return report

def generate_mnist_adv_examples():
    return

def generate_cifar_adv_examples():
    return

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    mnist_tutorial(nb_epochs=FLAGS.nb_epochs,
                   batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                         'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                       'Learning rate for training')
    tf.app.run()
