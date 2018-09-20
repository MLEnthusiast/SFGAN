import pickle as pkl
import time
#import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from models.research.slim.datasets import dataset_utils
from models.research.slim.nets import inception
import tensorflow as tf
import os
slim = tf.contrib.slim
from models.research.slim.preprocessing import inception_preprocessing

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

data_dir = 'data/'
url = "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
checkpoints_dir = "model_checkpoint/"

nrof_labelled = 1000

if not isdir(data_dir):
    raise Exception("Data directory doesn't exist!")

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

# Load the training and testing datasets
trainset = loadmat(data_dir + 'train_64x64.mat') # read the images as .mat format
testset = loadmat(data_dir + 'test_64x64.mat')
print("trainset shape:", trainset['X'].shape)
print("testset shape:", testset['X'].shape)

def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


class Dataset:
    def __init__(self, train, test, val_frac=0.1, shuffle=True, scale_func=None):
        split_idx = int(len(test['Y']) * (1 - val_frac))
        self.test_x, self.valid_x = test['X'][:, :, :, :split_idx], test['X'][:, :, :, split_idx:]
        self.test_y, self.valid_y = test['Y'][:split_idx], test['Y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['Y']
        # we pretend to use only a subset of labels
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[0:nrof_labelled] = 1

        self.train_x = np.rollaxis(self.train_x, axis=3)
        self.valid_x = np.rollaxis(self.valid_x, axis=3)
        self.test_x = np.rollaxis(self.test_x, axis=3)

        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler(self.train_x)
        self.valid_x = self.scaler(self.valid_x)
        self.test_x = self.scaler(self.test_x)
        self.shuffle = shuffle

    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        # Return the value of the named attribute of object
        num_examples = len(getattr(dataset, y_name))
        if self.shuffle:
            idx = np.arange(num_examples)
            np.random.shuffle(idx)
            setattr(dataset, x_name, getattr(dataset, x_name)[idx])
            setattr(dataset, y_name, getattr(dataset, y_name)[idx])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[idx]

        dataset_x = getattr(dataset, x_name)
        dataset_y = getattr(dataset, y_name)
        for ii in range(0, num_examples, batch_size):
            x = dataset_x[ii:ii + batch_size]
            y = dataset_y[ii:ii + batch_size]

            if which_set == "train":
                yield x, y, self.label_mask[ii:ii + batch_size]
            else:
                yield x, y


def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
    with tf.variable_scope('generator', reuse=reuse):
        # First fully connected layer
        x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
        # Reshape it to start the convolutional stack
        x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)

        x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)

        x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)

        x4 = tf.layers.conv2d_transpose(x3, size_mult, 5, strides=2, padding='same')
        x4 = tf.layers.batch_normalization(x4, training=training)
        x4 = tf.maximum(alpha * x4, x4)

        # Output layer
        logits = tf.layers.conv2d_transpose(x4, output_dim, 5, strides=2, padding='same')

        out = tf.tanh(logits)

        return out


def discriminator(x, reuse=False, alpha=0.2, drop_rate=0., num_classes=10, size_mult=64, x_inception=None):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.dropout(x, rate=drop_rate / 2.5)

        # Input layer is ?x64x64x3
        x0 = tf.layers.conv2d(x, size_mult, 3, strides=2, padding='same')
        relu0 = tf.maximum(alpha * x0, x0)
        relu0 = tf.layers.dropout(relu0, rate=drop_rate)  # [?x32x32x?]

        x1 = tf.layers.conv2d(relu0, size_mult, 3, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate)  # [?x16x16x?]

        x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)  # [?x8x8x?]
        relu2 = tf.maximum(alpha * bn2, bn2)

        x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same')  # [?x4x4x?]
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        relu3 = tf.layers.dropout(relu3, rate=drop_rate)

        x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)

        x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same')  # [?x4x4x?]
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)

        x6 = tf.layers.conv2d(relu5, filters=(2 * size_mult), kernel_size=3, strides=1, padding='valid')
        relu6 = tf.maximum(alpha * x6, x6)

        features = tf.reduce_mean(relu6, axis=[1, 2])
        inception_logits = tf.reduce_mean(x_inception, axis=[1,2])

        # concat the high level features
        features = tf.concat(axis=1, values=[features, inception_logits])

        class_logits = tf.layers.dense(features, num_classes)

        gan_logits = tf.reduce_logsumexp(class_logits, 1)

        out = tf.nn.softmax(class_logits) 

        return out, class_logits, gan_logits, features


def get_g_out(input_z, output_dim, alpha=0.2):
    g_size_mult = 32
    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
    return g_model


def model_loss(input_real, input_z, output_dim, y, num_classes, label_mask, alpha=0.2, drop_rate=0., smooth=0.1, x_inception_r=None, x_inception_f=None):
    
    g_size_mult = 32
    d_size_mult = 64

    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult, reuse=True)
    d_on_data = discriminator(input_real, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult, x_inception=x_inception_r)

    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data

    d_on_samples = discriminator(g_model, reuse=True, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult, x_inception=x_inception_f)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples

    real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data,
                                                                            labels=tf.ones_like(gan_logits_on_data) * (
                                                                            1 - smooth)))

    fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples,
                                                                            labels=tf.zeros_like(
                                                                                gan_logits_on_samples)))

    unsupervised_loss = real_data_loss + fake_data_loss

    y = tf.squeeze(y)
    suppervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                               labels=tf.one_hot(y, num_classes, dtype=tf.float32))

    label_mask = tf.squeeze(tf.to_float(label_mask))

    suppervised_loss = tf.reduce_sum(tf.multiply(suppervised_loss, label_mask))

    # get the mean
    suppervised_loss = suppervised_loss / tf.maximum(1.0, tf.reduce_sum(label_mask))
    d_loss = unsupervised_loss + suppervised_loss

    data_moments = tf.reduce_mean(data_features, axis=0)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
    eq = tf.equal(tf.squeeze(y), pred_class)
    correct = tf.reduce_sum(tf.to_float(eq))
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))

    return d_loss, g_loss, correct, masked_correct, g_model, pred_class


def model_opt(d_loss, g_loss, learning_rate, beta1):

    discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='d_optimizer').minimize(d_loss,
                                                                                            var_list=discriminator_train_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='g_optimizer').minimize(g_loss,
                                                                                            var_list=generator_train_vars)

    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)

    return d_train_opt, g_train_opt, shrink_lr

def train(dataset, epochs, batch_size, figsize=(5, 5)):

    gen_samples, train_accuracies, test_accuracies = [], [], []
    steps = 0
    z_dim = 100

    with tf.Graph().as_default():

        ###### place holders

        raw_input = tf.placeholder(dtype=tf.float32, shape=(64, 64, 3))
        raw_inputs_x = tf.placeholder(dtype=tf.float32, shape=(None, 299, 299, 3))

        inputs_real = tf.placeholder(tf.float32, (None, 64, 64, 3), name='input_real')

        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

        y = tf.placeholder(tf.int32, (None), name='y')

        label_mask = tf.placeholder(tf.int32, (None), name='label_mask')

        input_inception_real = tf.placeholder(tf.float32, (None, 8, 8, 2048), name='input_inception_real')
        input_inception_fake = tf.placeholder(tf.float32, (None, 8, 8, 2048), name='input_inception_fake')

        drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")

        lr_rate = 0.0003
        num_classes = 10
        learning_rate = tf.Variable(lr_rate, trainable=False)
        sample_z = np.random.normal(0, 1, size=(50, z_size))

        g_out = get_g_out(input_z=inputs_z, output_dim=real_size[2], alpha=0.2)

        loss_results = model_loss(inputs_real, inputs_z,
                                  real_size[2], y, num_classes,
                                  label_mask=label_mask,
                                  alpha=0.2,
                                  drop_rate=drop_rate,
                                  x_inception_r=input_inception_real,
                                  x_inception_f=input_inception_fake)

        d_loss, g_loss, correct, masked_correct, samples, pred_class = loss_results

        d_opt, g_opt, shrink_lr = model_opt(d_loss, g_loss, learning_rate, beta1=0.5)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits_inception, _ = inception.inception_v3_base(raw_inputs_x, final_endpoint='Mixed_7c')  # Mixed_7c

        init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
                                             slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            init_fn(sess)
            for e in range(epochs):
                print("Epoch", e)

                t1e = time.time()
                num_examples = 0
                num_correct = 0
                for x, _y, _label_mask in dataset.batches(batch_size):
                    assert 'int' in str(y.dtype)
                    steps += 1
                    num_examples += _label_mask.sum()

                    # Sample random noise for G
                    batch_z = np.random.normal(0, 1, size=(x.shape[0], z_size))

                    # Run optimizers
                    t1 = time.time()

                    gen_samples_out = sess.run(g_out, feed_dict={inputs_real: x, inputs_z: batch_z})

                    processed_batch_r = np.empty((0, 299, 299, 3))
                    processed_image = inception_preprocessing.preprocess_image(raw_input, height=299, width=299, is_training=False)
                    # real samples
                    for i in range(x.shape[0]):
                        temp = sess.run(processed_image, feed_dict={raw_input: x[i, :, :, :]})
                        processed_batch_r = np.append(processed_batch_r, np.reshape(temp, newshape=(1, 299, 299, 3)), axis=0)

                    # fake samples
                    processed_batch_f = np.empty((0, 299, 299, 3))
                    for i in range(x.shape[0]):
                        temp = sess.run(processed_image, feed_dict={raw_input: gen_samples_out[i, :, :, :]})
                        processed_batch_f = np.append(processed_batch_f, np.reshape(temp, newshape=(1, 299, 299, 3)), axis=0)

                    final_op_r = sess.run(logits_inception, feed_dict={raw_inputs_x: processed_batch_r})
                    final_op_f = sess.run(logits_inception, feed_dict={raw_inputs_x: processed_batch_f})

                    _, _, _correct = sess.run([d_opt, g_opt, masked_correct],
                                         feed_dict={inputs_real: x, inputs_z: batch_z, y: _y,
                                                    label_mask: _label_mask, input_inception_real: final_op_r,
                                                    input_inception_fake: final_op_f})

                    t2 = time.time()
                    num_correct += _correct

                sess.run([shrink_lr])

                train_accuracy = num_correct / float(num_examples)

                print("\t\tClassifier train accuracy: ", train_accuracy)

                num_examples = 0
                num_correct = 0
                for x, _y in dataset.batches(batch_size, which_set="valid"):
                    assert 'int' in str(y.dtype)
                    num_examples += x.shape[0]

                    processed_batch_r = np.empty((0, 299, 299, 3))
                    processed_image = inception_preprocessing.preprocess_image(raw_input, height=299, width=299, is_training=False)
                    # real samples
                    for i in range(x.shape[0]):
                        temp = sess.run(processed_image, feed_dict={raw_input: x[i, :, :, :]})
                        processed_batch_r = np.append(processed_batch_r, np.reshape(temp, newshape=(1, 299, 299, 3)), axis=0)

                    final_op_r = sess.run(logits_inception, feed_dict={raw_inputs_x: processed_batch_r})

                    _correct, = sess.run([correct], feed_dict={inputs_real: x,
                                                              y: _y,
                                                              drop_rate: 0.,
                                                              input_inception_real: final_op_r})
                    num_correct += _correct

                test_accuracy = num_correct / float(num_examples)
                print("\t\tClassifier test accuracy", test_accuracy)
                print("\t\tStep time: ", t2 - t1)
                t2e = time.time()
                print("\t\tEpoch time: ", t2e - t1e)
                
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

                gen_sample = sess.run(g_out, feed_dict={inputs_z: sample_z})
                gen_samples.append(gen_sample)

            y_predictions = []
            y_target = []
            num_examples = 0
            num_correct = 0
            for x, _y in dataset.batches(batch_size, which_set="test"):
                num_examples += x.shape[0]
                processed_batch_r = np.empty((0, 299, 299, 3))
                processed_image = inception_preprocessing.preprocess_image(raw_input, height=299, width=299,
                                                                           is_training=False)

                # real samples
                for i in range(x.shape[0]):
                    temp = sess.run(processed_image, feed_dict={raw_input: x[i, :, :, :]})
                    processed_batch_r = np.append(processed_batch_r, np.reshape(temp, newshape=(1, 299, 299, 3)),
                                                  axis=0)

                final_op_r = sess.run(logits_inception, feed_dict={raw_inputs_x: processed_batch_r})

                _correct, _y_pred, = sess.run([correct, pred_class], feed_dict={inputs_real: x, y: _y, drop_rate: 0., input_inception_real: final_op_r})
                num_correct += _correct

                y_predictions.append(_y_pred)
                y_target.append(_y)
            test_accuracy = num_correct / float(num_examples)
            print('Testing...')
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)


    return train_accuracies, test_accuracies, gen_samples, y_predictions, y_target

real_size = (64,64,3)
z_size = 100
learning_rate = 0.0003

dataset = Dataset(trainset, testset)

batch_size = 128
epochs = 30
train_accuracies, test_accuracies, gen_samples, y_predictions, y_target = train(dataset, epochs, batch_size, figsize=(10,5))

with open('val_acc_sfgan.pkl', 'wb') as f:
    pkl.dump(test_accuracies, f, protocol=2)

with open('y_predictions_sfgan.pkl', 'wb') as f:
    pkl.dump(y_predictions, f, protocol=2)

with open('y_target_sfgan.pkl', 'wb') as f:
    pkl.dump(y_target, f, protocol=2)

with open('gen_samples_sfgan.pkl', 'wb') as f:
    pkl.dump(gen_samples, f, protocol=2)

