import os
import tensorflow as tf
from solver import Solver

tf.reset_default_graph()

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '1', 'gpu index if you have muliple gpus, default: 0')
tf.flags.DEFINE_integer('batch_size', 2, 'batch size, default: 1')
tf.flags.DEFINE_string('dataset', 'mri_modality_dataset', 'dataset name, default: spine06')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial leraning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 500, 'save frequency for model, default: 10000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to continue training '
                                           '(e.g. 20181203-1647), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    if not FLAGS.is_train:
        print('test model loading' * 10)
        solver.test()


if __name__ == '__main__':
    tf.app.run()
