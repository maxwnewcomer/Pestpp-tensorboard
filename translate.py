import os
import io
import sys
import time
import signal
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard import default
from tensorboard import program
from tensorboard.plugins.hparams import api as hp

rootdir = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
###in production change reuqired to True
ap.add_argument('--inputdir', required=False,
	help='path to log directory of PEST++ output files')
ap.add_argument('--logname', required=False, default='runname',
    help='log file names for model iteration identification\nOptions\n — datetime\n — runname\n')
args = vars(ap.parse_args())
inputdir = args['inputdir']

###FOR TESTING###
inputdir = rootdir + '/input'
#################

class Tensorboard:
    def __init__(self, logdir):
        self.writer = tf.compat.v1.summary.FileWriter(logdir)
        self.logdir = logdir

    def open(self):


    def close(self):
        self.writer.close()

    def log_scalar(self, tag, value, global_step):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_histogram(self, tag, values, global_step, bins):
        counts, bin_edges = np.histogram(values, bins=bins)

        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, histo=hist)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()

    def log_plot(self, tag, figure, global_step):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=global_step)
        self.writer.flush()
    def log_hparams(self, hparams, run_dir):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

# To end program with ^c
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGINT, signal_handler)

# nameFormat
if args['logname'].lower() == 'datetime':
    nameFormat = time.strftime("%b %d %Y %H:%M:%S", time.gmtime())
if args['logname'].lower() == 'runname':
    for file in os.listdir(inputdir):
        if file.endswith(".iobj"):
            nameFormat = file[:-5]

# variable initializtion
iobjepoch = 0
iobjupdate = False
old_iobj = pd.DataFrame()
def check_iobj():
    global old_iobj
    global iobjupdate
    global iobjepoch

    for file in os.listdir(inputdir):
        if file.endswith(".iobj"):
            iobj = pd.read_csv(os.path.join(inputdir, file))

    if not iobj.equals(old_iobj):
        iobjupdate = True
        for column in iobj.columns:
            if column not in ['iteration', 'model_runs_completed', 'regul_kp1']:
                tensorboard.log_scalar(column, iobj[column][iobjepoch], iobjepoch)

    old_iobj = iobj[0:iobjepoch+1]
    if iobjupdate:
        iobjepoch += 1

iparepoch = 0
iparupdate = False
old_ipar = pd.DataFrame()
def check_ipar():
    global old_ipar
    global iparupdate
    global iparepoch

    for file in os.listdir(inputdir):
        if file.endswith(".ipar"):
            ipar = pd.read_csv(os.path.join(inputdir, file))

    if not ipar.equals(old_ipar):
        iparupdate = True
        tensorboard.log_histogram(tag='Parameter Tuning', values=ipar.iloc[[iparepoch], 1:].to_numpy(),
                                  bins='auto',
                                  global_step=iparepoch)
        for column in ipar.columns:
            if column not in ['iteration']:
                tensorboard.log_scalar(column, ipar[column][iparepoch], iparepoch)

    old_ipar = ipar[0:iparepoch+1]
    if iparupdate:
        iparepoch += 1

isenepoch = 0
isenupdate = False
old_isen = pd.DataFrame()
def check_isen():
    global old_isen
    global isenupdate
    global isenepoch

    for file in os.listdir(inputdir):
        if file.endswith(".isen"):
            isen = pd.read_csv(os.path.join(inputdir, file))

    if not isen.equals(old_isen):
        isenupdate = True
        tensorboard.log_histogram(tag='Parameter Sensitivity', values=isen.iloc[[isenepoch], 1:].to_numpy(),
                                  bins='auto',
                                  global_step=isenepoch)

    old_isen = isen[0:isenepoch+1]
    if isenupdate:
        isenepoch += 1

def saveHyperParams():
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
                          metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')])
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                  HP_NUM_UNITS: num_units,
                  HP_DROPOUT: dropout_rate,
                  HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)

tensorboard = Tensorboard(rootdir + ('/logs/{}'.format(nameFormat)))
tensorboard.open()
interrupted = False
print('Press ^C to quit translation of data to Tensorboard')

while(True):
    time.sleep(5) #for demo
    check_iobj()
    check_ipar()
    check_isen()
    if interrupted:
        tensorboard.close()
        print('\nI hope your ending me because your model ran successfully!')
        break
