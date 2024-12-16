import os
import sys
import logging

logger = logging.getLogger(__name__)  # logger
logger.setLevel(logging.INFO)


def _init_logger(flags, log_path):
    if flags.is_train:
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        # file handler
        file_handler = logging.FileHandler(os.path.join(log_path, 'dataset.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        # add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


class SpineC2M(object):
    def __init__(self, flags):
        self.flags = flags
        self.name = 'day2night'
        self.divide_patch_nums = 4
        self.image_size = (384, 384 * 4, 3)
        self.image_size_single = (int(384), int(384), 3)
        self.num_tests = 1000
        self.train_tfpath = r'..\data\train_raw\train_tf.tfrecords'
        self.test_tfpath = r'..\data\test_raw\test_tf.tfrecords'
        logger.info('Initialize {} dataset SUCCESS!'.format(self.flags.dataset))
        logger.info('Img size: {}'.format(self.image_size))

    def __call__(self, is_train='True'):
        if is_train:
            if not os.path.isfile(self.train_tfpath):
                sys.exit(' [!] Train tfrecord file is not found...')
            return self.train_tfpath
        else:
            if not os.path.isfile(self.test_tfpath):
                sys.exit(' [!] Test tfrecord file is not found...')
            return self.test_tfpath


def Dataset(dataset_name, flags, log_path=None):
    if flags.is_train:
        _init_logger(flags, log_path)  # init logger

    if dataset_name == 'mri_modality_dataset':
        return SpineC2M(flags)
    else:
        raise NotImplementedError
