import os
import time
import logging
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import pandas as pd

# noinspection PyPep8Naming
import tensorflow_utils as tf_utils
from dataset import Dataset
from pix2pix import Pix2Pix

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.iter_time = 0
        self._make_folders()
        self._init_logger()

        self.dataset = Dataset(self.flags.dataset, self.flags, log_path=self.log_out_dir)
        self.model = Pix2Pix(self.sess, self.flags, self.dataset.image_size, self.dataset(self.flags.is_train),
                             log_path=self.log_out_dir)

        self.sess.run(tf.global_variables_initializer())
        tf_utils.show_all_variables()
        self.saver = tf.train.Saver()
        variable_names = [v.name for v in tf.trainable_variables()]

    def _make_folders(self):
        if self.flags.is_train:  # train stage
            if self.flags.load_model is None:
                cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
                if not os.path.isdir(self.model_out_dir):
                    os.makedirs(self.model_out_dir)
            else:
                cur_time = self.flags.load_model
                self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, cur_time)
            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)

        elif not self.flags.is_train:  # test stage
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            self.eval_out_dir = "../eval/pix2pix"
            self.gt_out_dir = "../eval/gt"
            self.ct_out_dir = "../eval/ct"
            self.log_out_dir = "{}/logs/{}".format(self.flags.dataset, self.flags.load_model)

            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

            if not os.path.isdir(self.eval_out_dir):
                os.makedirs(self.eval_out_dir)

            if not os.path.isdir(self.gt_out_dir):
                os.makedirs(self.gt_out_dir)

            if not os.path.isdir(self.ct_out_dir):
                os.makedirs(self.ct_out_dir)

    def _init_logger(self):
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(self.log_out_dir, 'solver.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        if self.flags.is_train:
            logger.info('gpu_index: {}'.format(self.flags.gpu_index))
            logger.info('batch_size: {}'.format(self.flags.batch_size))
            logger.info('dataset: {}'.format(self.flags.dataset))

            logger.info('is_train: {}'.format(self.flags.is_train))
            logger.info('learning_rate: {}'.format(self.flags.learning_rate))
            logger.info('beta1: {}'.format(self.flags.beta1))

            logger.info('iters: {}'.format(self.flags.iters))
            logger.info('print_freq: {}'.format(self.flags.print_freq))
            logger.info('save_freq: {}'.format(self.flags.save_freq))
            logger.info('sample_freq: {}'.format(self.flags.sample_freq))
            logger.info('load_model: {}'.format(self.flags.load_model))

    def train(self):
        # load initialized checkpoint that provided
        if self.flags.load_model is not None:
            if self.load_model():
                print(' [*] Load SUCCESS!\n')
            else:
                print(' [!] Load Failed...\n')

        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        try:
            while self.iter_time < self.flags.iters:
                # samppling images and save them
                self.sample(self.iter_time)

                # train_step
                loss, summary = self.model.train_step()
                self.model.print_info(loss, self.iter_time)
                self.train_writer.add_summary(summary, self.iter_time)
                self.train_writer.flush()

                # save model
                self.save_model(self.iter_time)
                self.iter_time += 1

            self.save_model(self.flags.iters)

        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def test(self):
        if self.load_model():
            logger.info(' [*] Load SUCCESS!')
        else:
            logger.info(' [!] Load Failed...')
        ssim_lists = []
        psnr_lists = []
        # threads for tfrecord
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        iter_time = 0
        total_time = 0.

        try:
            while iter_time < self.dataset.num_tests:
                tic = time.time()
                imgs, img_names, metrics = self.model.test_step()
                margin = np.zeros((imgs[0].shape[0], 3))
                name = 'sample_ % d.png' % iter_time
                psnr_lists.append(metrics[0])
                ssim_lists.append(metrics[1])
                total_time += time.time() - tic

                self.model.plots_test(imgs, img_names, self.test_out_dir, self.eval_out_dir, self.gt_out_dir,
                                      self.ct_out_dir)
                iter_time += 1
            psnr_lists.append(np.mean(psnr_lists))
            ssim_lists.append(np.mean(ssim_lists))
            df = pd.DataFrame([psnr_lists, ssim_lists])

            mean_value = [np.mean(psnr_lists), np.mean(ssim_lists)]
            mean_value_pd = pd.DataFrame(mean_value)

            path = os.path.join(self.test_out_dir, 'testresult')
            if not os.path.exists(path):
                print("# path not exists")
                os.makedirs(path)
            whole_path = os.path.join(path, 'ssim_psnr.csv')
            mean_path = os.path.join(path, 'mean.csv')

            mean_value_pd.to_csv(mean_path)

            df.to_csv(whole_path)
            logger.info('Avg. PT: {:.2f} msec.'.format(total_time / self.dataset.num_tests * 1000.))
            print('predicted finished.......................')
        except KeyboardInterrupt:
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
        finally:
            # when done, ask the threads to stop
            coord.request_stop()
            coord.join(threads)

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, self.iter_time, self.dataset.image_size_single, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=self.iter_time)
            logger.info(' [*] Model saved! Iter: {}'.format(iter_time))

    def load_model(self):
        logger.info(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info(' [*] Load iter_time: {}'.format(self.iter_time))
            return True
        else:
            return False
