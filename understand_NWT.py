import os
import argparse
import pandas as pd
import tensorflow as tf
from tensorboard import program
from tensorboard.plugins.hparams import api as hp

rootdir = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
###in production change reuqired to True
ap.add_argument('--filepath', required=False,
	help='path to log directory of PEST++ output files')
args = vars(ap.parse_args())
filepath = args['filepath']

###FOR TESTING###
filepath = rootdir + '/input/NWT_Explore_out.csv'
#################

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.compat.v1.summary.FileWriter(logdir)
        self.logdir = logdir

    def open(self):
    	tb = program.TensorBoard()
    	tb.configure(argv=[None, '--logdir', '/' + os.path.join(*self.logdir.split('/')[0:-1]) + '/'])
    	url = tb.launch()
    	print('TensorBoard is opened at :', url)
    	print('\nPress ^C to close Tensorboard')

    def close(self):
    	self.writer.close()
    	print('\nI hope your ending me because your model ran successfully!')

    def log_hparams(self, hparams, Elapsed_min):
        with tf.summary.create_file_writer(self.logdir + '/hparam_tuning/' + hparams).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar(METRIC_SPEED, Elapsed_min, step=1)

def saveHyperParams(tb):
    hparamdf = pd.read_csv(filepath)
    # HP_LINEMETH = hp.HParam('linemeth', hp.Discrete([1, 2]))
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_SPEED = 'Elapsed_min'

    with tf.summary.create_file_writer(rootdir + '/logs/hparam_tuning').as_default():
        hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
                          metrics=[hp.Metric(METRIC_SPEED, display_name='Elapsed Minutes')])
    # for num_units in HP_NUM_UNITS.domain.values:
    #     for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    #         for optimizer in HP_OPTIMIZER.domain.values:

    # for row in hparamdf
        hparams = {
          HP_NUM_UNITS: 16,
          HP_DROPOUT: .12,
          HP_OPTIMIZER: 'adam',
        }
        Elapsed_min = 30.2
        tb.log_hparams(hparams, Elapsed_min)

if __name__ == "__main__":
    tensorboard = Tensorboard(rootdir + '/logs')
    saveHyperParams(tensorboard)
    tensorboard.open()
