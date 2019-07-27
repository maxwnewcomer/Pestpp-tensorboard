import os
import argparse
import signal
import logging
import pandas as pd
import tensorflow as tf
from tensorboard import program
from tensorboard.plugins.hparams import api as hp

##Disable annoying tensorboard http requests
logging.disable(logging.ERROR)

rootdir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
ap = argparse.ArgumentParser()
ap.add_argument('--filepath', required=True,
	help='path to log directory of PEST++ output files')
args = vars(ap.parse_args())
filepath = args['filepath']

# ###FOR TESTING###
# filepath = rootdir + '/input/NWT_Explore_out.csv'
# #################

class Tensorboard:
    def __init__(self, logdir):
        self.logdir = logdir

    def open(self):
    	tb = program.TensorBoard()
    	tb.configure(argv=[None, '--logdir', '/' + os.path.join(*self.logdir.split('/')[0:-1]) + '/'])
    	url = tb.launch()
    	print('TensorBoard is opened at :', url)
    	print('\nPress ^C to close Tensorboard')

    def log_hparams(self, hparams, Elapsed_min, id):
        with tf.summary.create_file_writer(self.logdir + '/' + id).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar('Elapsed_min', Elapsed_min, step=1)

def saveHyperParams(tb):
	print(os.path.join(cwd, filepath))
	hparamdf = pd.read_csv(filepath)
	maxrun = int(hparamdf[['Test']].max()[0])

	HP_LINMETH = hp.HParam('linmeth', hp.Discrete([1, 2]))
	HP_BACKFLAG = hp.HParam('backflag', hp.Discrete([0, 1]))
	HP_BACKREDUCE = hp.HParam('backreduce', hp.RealInterval(.00001, 1.))
	HP_BACKTOL = hp.HParam('backtol', hp.RealInterval(1., 2.))
	HP_DBDGAMMA = hp.HParam('dbdgamma', hp.RealInterval(0., 0.0001))
	HP_DBDKAPPA = hp.HParam('dbdkappa', hp.RealInterval(.00001, .0001))
	HP_DBDTHETA = hp.HParam('dbdtheta', hp.RealInterval(.4, 1.))
	HP_EPSRN = hp.HParam('epsrn', hp.RealInterval(.00005, .001))
	HP_FLUXTOL = hp.HParam('fluxtol', hp.RealInterval(5000., 1000000.))
	HP_HCLOSEXMD = hp.HParam('hclosexmd', hp.RealInterval(.00001, .001))
	HP_HEADTOL = hp.HParam('headtol', hp.RealInterval(.01, 5.))
	HP_IACL = hp.HParam('iacl', hp.IntInterval(0, 2))
	HP_IBOTAV = hp.HParam('ibotav', hp.Discrete([0, 1]))
	HP_IDROPTOL = hp.HParam('idroptol', hp.Discrete([0, 1]))
	HP_ILUMETHOD = hp.HParam('ilumethod', hp.Discrete([1, 2]))
	HP_IPRNWT = hp.HParam('iprnwt', hp.IntInterval(0, 2))
	HP_IREDSYS= hp.HParam('iredsys', hp.Discrete([0, 1]))
	HP_LEVEL= hp.HParam('level', hp.IntInterval(0, 10))
	HP_LEVFILL= hp.HParam('levfill', hp.IntInterval(0, 10))
	HP_MAXBACKITER = hp.HParam('maxbackiter', hp.IntInterval(10, 50))
	HP_MAXITEROUT = hp.HParam('maxiterout', hp.IntInterval(100, 400))
	HP_MAXITINNER = hp.HParam('maxitinner', hp.IntInterval(25, 1000))
	HP_MOMFACT = hp.HParam('momfact', hp.RealInterval(0., .1))
	HP_MSDR = hp.HParam('msdr', hp.IntInterval(5, 20))
	HP_MXITERXMD = hp.HParam('mxiterxmd', hp.IntInterval(5, 25))
	HP_NORDER = hp.HParam('norder', hp.IntInterval(0, 2))
	HP_NORTH = hp.HParam('north', hp.IntInterval(2, 10))
	HP_OPTIONS = hp.HParam('options', hp.Discrete(['SPECIFIED', 'CONTINUE']))
	HP_RRCTOLS = hp.HParam('rrctols', hp.RealInterval(0., .0001))
	HP_STOPTOL = hp.HParam('stoptol', hp.RealInterval(.000000000001, .00000001))
	HP_THICKFACT = hp.HParam('thickfact', hp.RealInterval(.000001, .0005))

	METRIC_SPEED = 'Elapsed_min'

	with tf.summary.create_file_writer(rootdir + '/logs/hparam_tuning').as_default():
	    hp.hparams_config(hparams=[HP_BACKFLAG,
	                               HP_BACKREDUCE, HP_BACKTOL, HP_DBDGAMMA,
	                               HP_DBDKAPPA, HP_DBDTHETA, HP_EPSRN, HP_FLUXTOL,
	                               HP_HCLOSEXMD, HP_HEADTOL, HP_IACL, HP_IBOTAV,
	                               HP_IDROPTOL, HP_ILUMETHOD, HP_IPRNWT, HP_IREDSYS,
	                               HP_LEVEL, HP_LEVFILL, HP_LINMETH, HP_MAXBACKITER,
	                               HP_MAXITEROUT, HP_MAXITINNER, HP_MOMFACT, HP_MSDR,
	                               HP_MXITERXMD, HP_NORDER, HP_NORTH, HP_OPTIONS,
	                               HP_RRCTOLS, HP_STOPTOL, HP_THICKFACT],
	                      metrics=[hp.Metric(METRIC_SPEED, display_name='Elapsed Minutes')])
	for run in hparamdf.itertuples():
	    hparams = {
	        HP_BACKFLAG: run.backflag,
	        HP_BACKREDUCE: run.backreduce,
	        HP_BACKTOL: run.backtol,
	        HP_DBDGAMMA: run.dbdgamma,
	        HP_DBDKAPPA: run.dbdkappa,
	        HP_DBDTHETA: run.dbdtheta,
	        HP_EPSRN: run.epsrn,
	        HP_FLUXTOL: run.fluxtol,
	        HP_HCLOSEXMD: run.hclosexmd,
	        HP_HEADTOL: run.headtol,
	        HP_IACL: run.iacl,
	        HP_IBOTAV: run.ibotav,
	        HP_IDROPTOL: run.idroptol,
	        HP_ILUMETHOD: run.ilumethod,
	        HP_IPRNWT: run.iprnwt,
	        HP_IREDSYS: run.iredsys,
	        HP_LEVEL: run.level,
	        HP_LEVFILL: run.levfill,
	        HP_LINMETH: run.linmeth,
	        HP_MAXBACKITER: run.maxbackiter,
	        HP_MAXITEROUT: run.maxiterout,
	        HP_MAXITINNER: run.maxitinner,
	        HP_MOMFACT: run.momfact,
	        HP_MSDR: run.msdr,
	        HP_MXITERXMD: run.mxiterxmd,
	        HP_NORDER: run.norder,
	        HP_NORTH: run.north,
	        HP_OPTIONS: run.options,
	        HP_RRCTOLS: run.rrctols,
	        HP_STOPTOL: run.stoptol,
	        HP_THICKFACT: run.thickfact,
	    }
	    Elapsed_min = run.Elapsed_min
	    tb.log_hparams(hparams, Elapsed_min, id=str(run.Test))
tensorboard = Tensorboard(rootdir + '/logs/hparam_tuning')
saveHyperParams(tensorboard)
tensorboard.open()

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGINT, signal_handler)

while(not interrupted):
    pass
