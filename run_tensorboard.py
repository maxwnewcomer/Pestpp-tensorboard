import os
import io
import sys
import time
import signal
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard import program

##Disable annoying tensorboard http requests
logging.disable(logging.ERROR)

cwd = os.getcwd()
ap = argparse.ArgumentParser()
ap.add_argument('--inputdir', required=True,
	help='path to log directory of PEST++ output files')
ap.add_argument('--logname', required=False, default='runname',
    help='log file names for model iteration identification\nOptions\n — datetime\n — runname\n')
args = vars(ap.parse_args())
inputdir = args['inputdir']

###FOR TESTING###
# inputdir = os.path.join(cwd, 'input')
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

# nameFormat
if args['logname'].lower() == 'datetime':
	nameFormat = time.strftime("%b %d %Y %H:%M:%S", time.gmtime())
if args['logname'].lower() == 'runname':
	for file in os.listdir(inputdir):
		if file.endswith(".iobj"):
			nameFormat = file[:-5]
	for file in os.listdir(os.path.join(cwd, 'logs')):
		if file.endswith(nameFormat):
			nameFormat = nameFormat + ' ' + time.strftime("%b %d %Y %H:%M:%S", time.gmtime())

tensorboard = Tensorboard(os.path.join(cwd, ('logs/{}'.format(nameFormat))))
tensorboard.open()

# To end program with ^c
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGINT, signal_handler)
interrupted = False

while(not interrupted):
    # time.sleep(.25) #for demo
    check_iobj()
    check_ipar()
    check_isen()

tensorboard.close()
