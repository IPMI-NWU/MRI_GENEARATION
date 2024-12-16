from __future__ import division
import os
import cv2
import logging
import collections
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow_utils as tf_utils
import utils as utils
from reader import Reader
from layer import (conv2d, deconv2d, max_pool_2x2, crop_and_concat, weight_xavier_init, bias_variable)
from cbam_module import cbam_block, channel_attention
import copy
from vgg16_pretain import *

slim = tf.contrib.slim

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)

train_flag = None


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """
    Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity of two batch of data,
    usually be used for binary image segmentation
    i.e. labels are binary.
    The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = 1. - tf.reduce_mean(dice)

    return dice


def dice_coe_path(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5, patch_nums=32, size=16):
    total = 0
    output_patch = tf.split(output, 16, 1)
    target_patch = tf.split(target, 16, 1)
    for each, each2 in zip(output_patch, target_patch):
        tem = tf.split(each, 16, 2)
        tem2 = tf.split(each2, 16, 2)
        for tem_output, tem_target in zip(tem, tem2):
            total += dice_coe(tem_output, tem_target)
    return total / 256


# # noinspection PyPep8Naming
class Pix2Pix(object):
    def __init__(self, sess, flags, img_size, data_path, log_path=None):
        self.sess = sess
        self.flags = flags
        img_size_trans = (img_size[0], img_size[0], int((int(img_size[1] / img_size[0] - 1) * img_size[2] / 3)))
        img_size_trans = list(img_size_trans)
        self.img_size = img_size_trans
        self.data_path = data_path
        self.log_path = log_path

        self.deep_supervision_weight = [0.3, 0.6]
        self.L3_lamba = 1
        self.style_cnt = 0
        self._gen_train_ops_1, self._gen_train_ops_2, self._gen_train_ops_3, = [], [], []
        self._seg_train_ops, self._dis_train_ops = [], []
        self.gen_c = [64, 128, 256, 512, 512, 512, 512, 512,
                      512, 512, 512, 512, 256, 128, 64, self.img_size[2]]
        self.start_decay_step = int(self.flags.iters / 2)
        self.decay_steps = self.flags.iters - self.start_decay_step
        self.eps = 1e-12

        self._init_logger()  # init logger
        self._build_net()  # init graph
        self._tensorboard()  # init tensorboard

    def _init_logger(self):
        if self.flags.is_train:
            tf_utils._init_logger(self.log_path)

    def calc_style_loss(self, real_imgs, fake_imgs):
        with slim.arg_scope(vgg_arg_scope()):
            f1, f2, f3, f4, exclude = vgg16(tf.concat([real_imgs, fake_imgs], axis=0))
            style_loss = styleloss(f1, f2, f3, f4)
            # load vgg model
            if (self.style_cnt == 0):
                vgg_model_path = r'C:\Users\win10\Desktop\vgg_16_2016_08_28/vgg_16.ckpt'
                vgg_vars = slim.get_variables_to_restore(include=['vgg_16'], exclude=exclude)
                init_fn = slim.assign_from_checkpoint_fn(vgg_model_path, vgg_vars)
                init_fn(self.sess)
                self.style_cnt += 1

        return style_loss

    def _build_net(self):

        self.x_1_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='x1_test_tfph')
        self.x_2_test_tfph = tf.placeholder(tf.float32, shape=[None, *self.img_size], name='x2_test_tfph')
        self.generator = Generator2(name='gen_1', gen_c=self.gen_c, image_size=self.img_size,
                                    _ops=self._gen_train_ops_1)
        data_reader = Reader(self.data_path, name='data', image_size=self.img_size, batch_size=self.flags.batch_size,
                             is_train=self.flags.is_train)

        self.x_t1_imgs, self.x_dwi_imgs, self.y_imgs, self.x_mask_img, self.tumor_img, \
        self.x_t1_imgs_ori, self.x_dwi_imgs_ori, self.y_imgs_ori, self.mask_imgs_ori, self.tumor_imgs_ori, self.img_name = data_reader.feed()

        one = tf.ones_like(self.tumor_img)
        zero = tf.zeros_like(self.tumor_img)
        label = tf.where(self.tumor_img > 0, x=one, y=zero)
        self.tumor_img = label

        self.x_1, self.x_2 = self.x_t1_imgs, self.x_dwi_imgs

        generator_output = self.generator(self.x_1, self.x_2)

        self.merge_generate_output = generator_output[0][0][0]
        self.merge_trainable_parameters = generator_output[0][1]
        self.merge_op = generator_output[0][2]

        self.dwi_sample_d2 = generator_output[1][0][1]
        self.dwi_sample_d4 = generator_output[1][0][2]
        self.dwi_final = generator_output[1][0][0]
        self.dwi_trainable_parameters = generator_output[1][1]
        self.dwi_op = generator_output[1][2]

        self.first_sample_d2 = generator_output[2][0][1]
        self.first_sample_d4 = generator_output[2][0][2]
        self.first_final = generator_output[2][0][0]
        self.first_trainable_parameters = generator_output[2][1]
        self.first_op = generator_output[2][2]

        self.g_dwi_d2_tumor_loss = dice_coe(self.dwi_sample_d2, self.y_imgs * self.tumor_img)

        self.g_dwi_d4_tumor_loss = dice_coe(self.dwi_sample_d4, self.y_imgs * self.tumor_img)

        self.g_dwi_final_tumor_loss = dice_coe(self.dwi_final, self.y_imgs * self.tumor_img)

        self.g_dwi_all_cost = (self.deep_supervision_weight[0] * self.g_dwi_d2_tumor_loss +
                               self.deep_supervision_weight[1] * self.g_dwi_d4_tumor_loss + self.g_dwi_final_tumor_loss)

        self.g_first_d2_loss = self.L3_lamba * tf.reduce_mean(tf.abs(self.y_imgs - self.first_sample_d2))

        self.g_first_d4_loss = self.L3_lamba * tf.reduce_mean(tf.abs(self.y_imgs - self.first_sample_d4))

        self.g_first_final_loss = self.L3_lamba * tf.reduce_mean(tf.abs(self.y_imgs - self.first_final))

        self.g_first_all_cost = (self.deep_supervision_weight[0] * self.g_first_d2_loss + self.deep_supervision_weight[
            1] * self.g_first_d4_loss) + self.g_first_final_loss

        self.g_merge_loss = self.L3_lamba * tf.reduce_mean(tf.abs(self.y_imgs - self.merge_generate_output))

        self.g_merge_all_cost = self.g_merge_loss

        gen_dwi_op = self.optimizer(loss=self.g_dwi_all_cost, variables=self.dwi_trainable_parameters,
                                    name='gen_dwi_parameters')
        gen_first_op = self.optimizer(loss=self.g_first_all_cost, variables=self.first_trainable_parameters,
                                      name='gen_first_parameters')
        gen_merge_op = self.optimizer(loss=self.g_merge_all_cost, variables=self.merge_trainable_parameters,
                                      name='gen_merge_parameters')

        gen_dwi_op_all = [gen_dwi_op] + self.dwi_op
        self.g_dwi_optim = tf.group(*gen_dwi_op_all)

        gen_first_op_all = [gen_first_op] + self.first_op
        self.g_first_optim = tf.group(*gen_first_op_all)

        gen_merge_op_all = [gen_merge_op] + self.merge_op
        self.g_merge_optim = tf.group(*gen_merge_op_all)

        self.fake_y_sample = self.generator(self.x_1_test_tfph, self.x_2_test_tfph)[0][0][0]
        self.fake_y_dwi = self.generator(self.x_1_test_tfph, self.x_2_test_tfph)[1][0][0]
        self.fake_y_first = self.generator(self.x_1_test_tfph, self.x_2_test_tfph)[2][0][0]
        self.input_tumor_region, self.first2 = self.generator(self.x_1_test_tfph, self.x_2_test_tfph)[3], \
                                               self.generator(self.x_1_test_tfph, self.x_2_test_tfph)[4]

    def optimizer(self, loss, variables, name='Adam'):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.flags.learning_rate
        end_learning_rate = 0.
        start_decay_step = self.start_decay_step
        decay_steps = self.decay_steps

        learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                  tf.train.polynomial_decay(starter_learning_rate,
                                                            global_step - start_decay_step,
                                                            decay_steps, end_learning_rate, power=1.0),
                                  starter_learning_rate))
        tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

        learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.flags.beta1, name=name). \
            minimize(loss, global_step=global_step, var_list=variables)

        return learn_step

    def _tensorboard(self):
        tf.summary.scalar('loss/gen_merge_total_loss', self.g_merge_all_cost)
        tf.summary.scalar('loss/gen_dwi_total_loss', self.g_dwi_all_cost)
        tf.summary.scalar('loss/gen_first_total_loss', self.g_first_all_cost)
        self.summary_op = tf.summary.merge_all()

    def train_step(self):

        gen_dwi_ops = [self.g_dwi_optim, self.g_dwi_all_cost, self.summary_op]

        gen_first_ops = [self.g_first_optim, self.g_first_all_cost, ]

        gen_merge_ops = [self.g_merge_optim, self.g_merge_all_cost, ]

        _, gan_dwi_loss, summary = self.sess.run(gen_dwi_ops)

        _, gan_first_loss, = self.sess.run(gen_first_ops)

        _, gan_merge_loss, = self.sess.run(gen_merge_ops)

        return [gan_dwi_loss, gan_first_loss, gan_merge_loss], summary

    def test_step(self):
        x1_vals, x2_vals, y_vals, img_name = self.sess.run(
            [self.x_t1_imgs_ori, self.x_dwi_imgs_ori, self.y_imgs_ori, self.img_name])
        fakes_y, fake_y_first, fake_y_dwi = self.sess.run(
            [self.fake_y_sample, self.fake_y_first, self.fake_y_dwi],
            feed_dict={self.x_1_test_tfph: x1_vals, self.x_2_test_tfph: x2_vals})
        fakes_y = tf.convert_to_tensor(fakes_y)
        y_vals = tf.convert_to_tensor(y_vals)
        fakes_y = tf.squeeze(fakes_y)
        y_vals = tf.squeeze(y_vals)
        max_value = 1
        batch_psnr = tf.image.psnr(fakes_y, y_vals, max_val=max_value)
        batch_ssim = tf.image.ssim(fakes_y, y_vals, max_val=max_value)

        fakes_y = tf.squeeze(fakes_y)
        y_vals = tf.squeeze(y_vals)
        fakes_y = fakes_y.eval(session=self.sess)
        y_vals = y_vals.eval(session=self.sess)
        fake_y_first_ori = copy.deepcopy(fake_y_first)

        return [x1_vals, x2_vals, fakes_y, y_vals, fake_y_first_ori, fake_y_first], img_name, [batch_ssim, batch_psnr]

    def sample_imgs(self):
        x1_vals, x2_vals, y_vals, tumor, tumor_region = self.sess.run(
            [self.x_t1_imgs, self.x_dwi_imgs, self.y_imgs, self.tumor_img, self.tumor_img * self.y_imgs])
        input_tumor_region, first2, fakes_y, fake_y_first, fake_y_dwi = self.sess.run(
            [self.input_tumor_region, self.first2,
             self.fake_y_sample, self.fake_y_first, self.fake_y_dwi],
            feed_dict={self.x_1_test_tfph: x1_vals, self.x_2_test_tfph: x2_vals})
        return [x1_vals, x2_vals, fakes_y, fake_y_first, fake_y_dwi, y_vals, tumor, tumor_region,
                input_tumor_region, first2]

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('dwi_gan_loss', loss[0]),
                                                  ('first_gan_loss', loss[1]),
                                                  ('merge_gan_loss', loss[2]),
                                                  # ('tumor_gan_loss', loss[3]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    @staticmethod
    def plots(imgs, iter_time, image_size, save_file):

        scale, margin = 0.02, 0.02
        n_cols, n_rows = len(imgs), imgs[0].shape[0]
        cell_size_h, cell_size_w = imgs[0].shape[1] * scale, imgs[0].shape[2] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)
        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                result = np.array(imgs[col_index][row_index])
                x = result
                if (x.shape[2]) == 3:
                    x = x[:, :, 0] * 0.299 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.114
                    result = x
                else:
                    result = x
                plt.imshow(result, cmap='Greys_r')
                plt.imshow((result).reshape(image_size[0], image_size[1]), cmap='Greys_r')

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time).zfill(5)), bbox_inches='tight')
        plt.close(fig)

    def plots_test(self, batch_imgs, img_name, save_file, eval_file, gt_file, ct_file):
        img_name_ = img_name[0].decode('utf-8')
        my_dpi = 96
        imgs = [np.squeeze(batch_imgs[idx]) for idx in range(len(batch_imgs))]
        concat_imgs = np.hstack(imgs)
        concat_imgs = concat_imgs[:, :, 0] * 0.299 + concat_imgs[:, :, 1] * 0.587 + concat_imgs[:, :, 2] * 0.114
        fig, ax = plt.subplots()
        ax.imshow(concat_imgs, aspect='equal')
        plt.axis("off")
        height, width = concat_imgs.shape
        fig.set_size_inches(width / my_dpi, height / my_dpi)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.imshow(concat_imgs, cmap='Greys_r')
        plt.savefig(os.path.join(save_file, img_name_), dpi=my_dpi, constrained_layout=True)
        plt.close("all")


class Generator2(object):
    def __init__(self, name=None, gen_c=None, image_size=(384, 384, 2), _ops=None):
        self.name = name
        self.name_merge = "gen_merge"
        self.gen_c = gen_c
        self.image_size = image_size
        self.gen_1_ops = []
        self.gen_2_ops = []
        self.drop_conv = 0.5
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.image_channel = image_size[2]
        self.n_class = 1

    def gen_deconv(self, batch_input, out_channels, name):
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2],
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf_utils.conv2d(resized_input, out_channels, d_h=1, d_w=1, name=name)

    def print_size_change(self, layer_name, name):
        print('{0} layer shape is:{1}'.format(name, layer_name.shape))

    def construct_generator_func(self, scope_name, reuse_flag, input, flag_type="soft"):
        self.gen_1_ops = []
        with tf.variable_scope(scope_name, reuse=reuse_flag):
            tf_utils.print_activations(input)
            X = input
            gen1_e0_conv2d = tf_utils.conv2d(X, self.gen_c[0], name='gen1_e0_conv2d')
            gen1_e0_lrelu = tf_utils.lrelu(gen1_e0_conv2d, name='gen1_e0_lrelu')
            gen1_e0_cbam = cbam_block(gen1_e0_lrelu, name='gen1_e0_cbam', ratio=8)

            gen1_e1_conv2d = tf_utils.conv2d(gen1_e0_cbam, self.gen_c[1], name='gen1_e1_conv2d')
            gen1_e1_batchnorm = tf_utils.batch_norm(gen1_e1_conv2d, name='gen1_e1_batchnorm', _ops=self.gen_1_ops)
            gen1_e1_lrelu = tf_utils.lrelu(gen1_e1_batchnorm, name='gen1_e1_lrelu')
            gen1_e1_cbam = cbam_block(gen1_e1_lrelu, name='gen1_e1_cbam', ratio=8)

            gen1_e2_conv2d = tf_utils.conv2d(gen1_e1_cbam, self.gen_c[2], name='gen1_e2_conv2d')
            gen1_e2_batchnorm = tf_utils.batch_norm(gen1_e2_conv2d, name='gen1_e2_batchnorm', _ops=self.gen_1_ops)
            gen1_e2_lrelu = tf_utils.lrelu(gen1_e2_batchnorm, name='gen1_e2_lrelu')
            gen1_e2_cbam = cbam_block(gen1_e2_lrelu, name='gen1_e2_cbam', ratio=8)

            gen1_e3_conv2d = tf_utils.conv2d(gen1_e2_cbam, self.gen_c[3], name='gen1_e3_conv2d')
            gen1_e3_batchnorm = tf_utils.batch_norm(gen1_e3_conv2d, name='gen1_e3_batchnorm', _ops=self.gen_1_ops)
            gen1_e3_lrelu = tf_utils.lrelu(gen1_e3_batchnorm, name='gen1_e3_lrelu')
            gen1_e3_cbam = cbam_block(gen1_e3_lrelu, name='gen1_e3_cbam', ratio=8)

            gen1_e4_conv2d = tf_utils.conv2d(gen1_e3_cbam, self.gen_c[4], name='gen1_e4_conv2d')
            gen1_e4_batchnorm = tf_utils.batch_norm(gen1_e4_conv2d, name='gen1_e4_batchnorm', _ops=self.gen_1_ops)
            gen1_e4_lrelu = tf_utils.lrelu(gen1_e4_batchnorm, name='gen1_e4_lrelu')
            gen1_e4_cbam = cbam_block(gen1_e4_lrelu, name='gen1_e4_cbam', ratio=8)

            gen1_e5_conv2d = tf_utils.conv2d(gen1_e4_cbam, self.gen_c[5], name='gen1_e5_conv2d')
            gen1_e5_batchnorm = tf_utils.batch_norm(gen1_e5_conv2d, name='gen1_e5_batchnorm', _ops=self.gen_1_ops)
            gen1_e5_lrelu = tf_utils.lrelu(gen1_e5_batchnorm, name='gen1_e5_lrelu')
            gen1_e5_cbam = cbam_block(gen1_e5_lrelu, name='gen1_e5_cbam', ratio=8)

            gen1_e6_conv2d = tf_utils.conv2d(gen1_e5_cbam, self.gen_c[6], name='gen1_e6_conv2d')
            gen1_e6_batchnorm = tf_utils.batch_norm(gen1_e6_conv2d, name='gen1_e6_batchnorm', _ops=self.gen_1_ops)
            gen1_e6_lrelu = tf_utils.lrelu(gen1_e6_batchnorm, name='gen1_e6_lrelu')
            gen1_e6_cbam = cbam_block(gen1_e6_lrelu, name='gen1_e6_cbam', ratio=8)

            gen1_d1_deconv = self.gen_deconv(gen1_e6_cbam, self.gen_c[9],
                                             name='gen1_d1_deconv2d')  # e6_lrelu  d0_relu
            gen1_shapeA = gen1_e5_batchnorm.get_shape().as_list()[1]
            gen1_shapeB = gen1_d1_deconv.get_shape().as_list()[1] - gen1_e5_batchnorm.get_shape().as_list()[1]
            gen1_d1_split, _ = tf.split(gen1_d1_deconv, [gen1_shapeA, gen1_shapeB], axis=1, name='gen1_d1_split')
            tf_utils.print_activations(gen1_d1_split)
            gen1_d1_batchnorm = tf_utils.batch_norm(gen1_d1_split, name='gen1_d1_batchnorm', _ops=self.gen_1_ops)
            gen1_d1_drop = tf.nn.dropout(gen1_d1_batchnorm, keep_prob=0.5, name='gen1_d1_dropout')
            gen1_d1_concat = tf.concat([gen1_d1_drop, gen1_e5_batchnorm], axis=3, name='gen1_gen1_d1_concat')
            gen1_d1_relu = tf_utils.relu(gen1_d1_concat, name='gen1_d1_relu')
            gen1_d1_cbam = cbam_block(gen1_d1_relu, name='gen1_d1_cbam', ratio=8)

            # Decoder
            gen1_d2_deconv = self.gen_deconv(gen1_d1_cbam, self.gen_c[10], name='gen1_d2_deconv2d')
            gen1_shapeA = gen1_e4_batchnorm.get_shape().as_list()[2]
            gen1_shapeB = gen1_d2_deconv.get_shape().as_list()[2] - gen1_e4_batchnorm.get_shape().as_list()[2]
            gen1_d2_split, _ = tf.split(gen1_d2_deconv, [gen1_shapeA, gen1_shapeB], axis=2, name='gen1_d2_split')
            tf_utils.print_activations(gen1_d2_split)
            gen1_d2_batchnorm = tf_utils.batch_norm(gen1_d2_split, name='gen1_d2_batchnorm', _ops=self.gen_1_ops)
            gen1_d2_drop = tf.nn.dropout(gen1_d2_batchnorm, keep_prob=0.5, name='gen1_d2_dropout')
            gen1_d2_concat = tf.concat([gen1_d2_drop, gen1_e4_batchnorm], axis=3, name='gen1_d2_concat')
            gen1_d2_relu = tf_utils.relu(gen1_d2_concat, name='gen1_d2_relu')
            gen1_d2_cbam = cbam_block(gen1_d2_relu, name='gen1_d2_cbam', ratio=8)

            gen1_d2_upsampling = tf.image.resize_bilinear(gen1_d2_cbam, (self.image_size[0], self.image_size[1]), \
                                                          align_corners=False, name='gen1_d3output')
            gen1_d2_upsampling = tf_utils.conv2d_11(gen1_d2_upsampling, self.gen_c[15], name='gen1_d3_conv2d_11')
            gen1_d2_output = tf_utils.sigmoid(gen1_d2_upsampling, name='gen1_d3_output_tanh')

            gen1_d3_deconv = self.gen_deconv(gen1_d2_cbam, self.gen_c[11], name='gen1_d3_deconv2d')
            gen1_shapeA = gen1_e3_batchnorm.get_shape().as_list()[1]
            gen1_shapeB = gen1_d3_deconv.get_shape().as_list()[1] - gen1_e3_batchnorm.get_shape().as_list()[1]
            gen1_d3_split_1, _ = tf.split(gen1_d3_deconv, [gen1_shapeA, gen1_shapeB], axis=1, name='gen1_d3_split_1')
            tf_utils.print_activations(gen1_d3_split_1)

            gen1_shapeA = gen1_e3_batchnorm.get_shape().as_list()[2]
            gen1_shapeB = gen1_d3_split_1.get_shape().as_list()[2] - gen1_e3_batchnorm.get_shape().as_list()[2]
            gen1_d3_split_2, _ = tf.split(gen1_d3_split_1, [gen1_shapeA, gen1_shapeB], axis=2, name='d3_split_2')
            tf_utils.print_activations(gen1_d3_split_2)
            gen1_d3_batchnorm = tf_utils.batch_norm(gen1_d3_split_2, name='gen1_d3_batchnorm', _ops=self.gen_1_ops)
            gen1_d3_concat = tf.concat([gen1_d3_batchnorm, gen1_e3_batchnorm], axis=3, name='gen1_d3_concat')
            gen1_d3_relu = tf_utils.relu(gen1_d3_concat, name='gen1_d3_relu')
            gen1_d3_cbam = cbam_block(gen1_d3_relu, name='gen1_d3_cbam', ratio=8)

            gen1_d4_deconv = self.gen_deconv(gen1_d3_cbam, self.gen_c[12], name='gen1_d4_deconv2d')
            gen1_shapeA = gen1_e2_batchnorm.get_shape().as_list()[2]
            gen1_shapeB = gen1_d4_deconv.get_shape().as_list()[2] - gen1_e2_batchnorm.get_shape().as_list()[2]
            gen1_d4_split, _ = tf.split(gen1_d4_deconv, [gen1_shapeA, gen1_shapeB], axis=2, name='gen1_d4_split')
            tf_utils.print_activations(gen1_d4_split)
            gen1_d4_batchnorm = tf_utils.batch_norm(gen1_d4_split, name='gen1_d4_batchnorm', _ops=self.gen_1_ops)
            gen1_d4_concat = tf.concat([gen1_d4_batchnorm, gen1_e2_batchnorm], axis=3, name='gen1_d4_concat')
            gen1_d4_relu = tf_utils.relu(gen1_d4_concat, name='gen1_d4_relu')
            gen1_d4_cbam = cbam_block(gen1_d4_relu, name='gen1_d4_cbam', ratio=8)
            gen1_d4_upsampling = tf.image.resize_bilinear(gen1_d4_relu, (self.image_size[0], self.image_size[1]),
                                                          align_corners=False, name='gen1_output')
            gen1_d4_upsampling = tf_utils.conv2d_11(gen1_d4_upsampling, self.gen_c[15], name='gen1_d4_conv2d_11')
            gen1_d4_output = tf_utils.sigmoid(gen1_d4_upsampling, name='gen1_d4_output_tanh')

            gen1_d5_deconv = self.gen_deconv(gen1_d4_cbam, self.gen_c[13], name='gen1_d5_deconv2d')
            gen1_shapeA = gen1_e1_batchnorm.get_shape().as_list()[1]
            gen1_shapeB = gen1_d5_deconv.get_shape().as_list()[1] - gen1_e1_batchnorm.get_shape().as_list()[1]
            gen1_d5_split, _ = tf.split(gen1_d5_deconv, [gen1_shapeA, gen1_shapeB], axis=1, name='gen1_d5_split')
            tf_utils.print_activations(gen1_d5_split)
            gen1_d5_batchnorm = tf_utils.batch_norm(gen1_d5_split, name='gen1_d5_batchnorm', _ops=self.gen_1_ops)
            gen1_d5_concat = tf.concat([gen1_d5_batchnorm, gen1_e1_batchnorm], axis=3, name='gen1_d5_concat')
            gen1_d5_relu = tf_utils.relu(gen1_d5_concat, name='gen1_d5_relu')
            gen1_d5_cbam = cbam_block(gen1_d5_relu, name='gen1_d5_cbam', ratio=8)

            gen1_d6_deconv = self.gen_deconv(gen1_d5_cbam, self.gen_c[14], name='gen1_d6_deconv2d')
            gen1_d6_batchnorm = tf_utils.batch_norm(gen1_d6_deconv, name='gen1_d6_batchnorm', _ops=self.gen_1_ops)
            gen1_d6_concat = tf.concat([gen1_d6_batchnorm, gen1_e0_conv2d], axis=3, name='gen1_d6_concat')
            gen1_d6_relu = tf_utils.relu(gen1_d6_concat, name='gen1_d6_relu')
            gen1_d7_deconv = self.gen_deconv(gen1_d6_relu, self.gen_c[15], name='gen1_d7_deconv2d')
            gen1_output = tf_utils.sigmoid(gen1_d7_deconv, name='gen1_output_tanh')

            self.gen_1_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            gen_1 = [gen1_output, gen1_d2_output, gen1_d4_output]
            middle_layer = [gen1_e0_cbam, gen1_e1_cbam, gen1_e2_cbam, gen1_e3_cbam, gen1_e4_cbam, gen1_e5_cbam,
                            gen1_e6_cbam, gen1_d1_cbam,
                            gen1_d2_cbam, gen1_d3_cbam, gen1_d4_cbam, gen1_d5_cbam, gen1_output]
            training_variable = self.gen_1_variables
        return gen_1, middle_layer, training_variable, self.gen_1_ops;

    def __call__(self, first_img, dwi_img):
        dce_by_dwi, dwi_middle_layer, dwi_training_variable, dwi_op = self.construct_generator_func('generator_dwi',
                                                                                                    tf.AUTO_REUSE,
                                                                                                    tf.concat([dwi_img,
                                                                                                               first_img],
                                                                                                              axis=3))
        dce_by_first, first_middle_layer, first_training_variable, first_op = self.construct_generator_func(
            'generator_first',
            tf.AUTO_REUSE, first_img)

        with tf.variable_scope(self.name_merge, reuse=tf.AUTO_REUSE):
            tmp = dce_by_dwi[0][0]
            tmp = tf.expand_dims(tmp, axis=0, name=None, dim=None)

            first = dce_by_first[0][0]
            first = tf.expand_dims(first, axis=0, name=None, dim=None)

            zero = tf.zeros_like(tmp)
            one = tf.ones_like(tmp)
            mask = tf.where(tmp < 0.4, x=zero, y=one)  # 0.5为阈值
            mask2 = tf.where(tmp < 0.4, x=one, y=zero)  # 0.5为阈值
            img_mask2 = first * mask2
            gen1_d7_deconv = tf.add(img_mask2, tmp * mask)

            img_mask1 = tmp * mask
            gen1_output = tf_utils.conv2d_11(gen1_d7_deconv, self.gen_c[15], name='gen1_d8_conv')
            gen1_all_output = [gen1_output]
            reuse_flag = True
            self.gen_2_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_merge)

        return [gen1_all_output, self.gen_2_variables, self.gen_2_ops], [dce_by_dwi, dwi_training_variable, dwi_op], [
            dce_by_first, first_training_variable, first_op], img_mask2, img_mask1
