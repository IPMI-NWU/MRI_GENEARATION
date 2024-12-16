import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

slim = tf.contrib.slim
PROJECT_PATH = os.path.dirname(os.path.abspath(os.getcwd()))
tf.app.flags.DEFINE_string('pretrained_model_path',
                           os.path.join(PROJECT_PATH, r'C:\Users\win10\Desktop\vgg_16_2016_08_28\vgg_16.ckpt'), '')
FLAGS = tf.app.flags.FLAGS


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch, img_name):
    feature_map = np.squeeze(img_batch)

    feature_map_combination = []
    plt.figure(figsize=(20, 20))
    plt.suptitle("Hidden layer feature map")

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, 10):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split, cmap="gray")
        plt.axis('off')
        plt.title('feature_map_{}'.format(i), fontdict={'size': 6})

    plt.savefig(img_name + '.png')
    plt.show()

    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)


def vgg_arg_scope(weight_decay=0.1):
    """定义 VGG arg scope.
    Args:
      weight_decay: The l2 regularization coefficient.
    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def styleloss(f1, f2, f3, f4):
    gen_f, real_f = tf.split(f1, 2, 0)
    size = tf.size(gen_f)
    style_loss = tf.nn.l2_loss(gram(gen_f) - gram(real_f)) * 2 / tf.to_float(size)

    gen_f, real_f = tf.split(f2, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(real_f)) * 2 / tf.to_float(size)

    gen_f, real_f = tf.split(f3, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(real_f)) * 2 / tf.to_float(size)

    gen_f, real_f = tf.split(f4, 2, 0)
    size = tf.size(gen_f)
    style_loss += tf.nn.l2_loss(gram(gen_f) - gram(real_f)) * 2 / tf.to_float(size)

    return style_loss


def vgg16(inputs, scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], ):
            # outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

            # with tf.variable_scope('relu1'):
            out1 = net

            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')

            out2 = net

            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')

            out3 = net

            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')

            out4 = net
            exclude = ['vgg_16/fc6', 'vgg_16/pool4', 'vgg_16/conv5', 'vgg_16/pool5', 'vgg_16/fc7',
                       'vgg_16/global_pool', 'vgg_16/fc8/squeezed', 'vgg_16/fc8']

            return out1, out2, out3, out4, exclude


def net(real_img, gen_img, ses):
    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    with slim.arg_scope(vgg_arg_scope()):
        f1, f2, f3, f4 = vgg16(input_image)
    init = tf.global_variables_initializer()
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with ses as sess:
        sess.run(init)
        if FLAGS.pretrained_model_path is not None:
            variable_restore_op(sess)

        imgs = cv.imread(r"G:\dwi_first_zuixin\spine06\test\20230415-104408\1081520_120breast.png")
        img_fake = imgs[:, imgs.shape[0] * 2:imgs.shape[0] * 3, :]
        img_real = imgs[:, imgs.shape[0] * 3:imgs.shape[0] * 4, :]
        f1, f2, f3, f4 == sess.run([f1, f2, f3, f4],
                                   feed_dict={input_image: tf.concat([gen_img, real_img], axis=0).eval()})
        perceptual_loss = styleloss(f1, f2, f3, f4)
        return perceptual_loss


if __name__ == '__main__':
    net()
    print(tf.trainable_variables())
