import os
import time
import fileinput
from subprocess import call
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

cwd = os.getcwd()
# namefile = 'haskell.nam'
# listfile  = 'haskell.list'
# initnwt = 'haskell.nwt'
for file in os.listdir(cwd):
    if file.endswith('.nam'):
        namefile = file
    if file.endswith('.list'):
        listfile = file
    if file.endswith('.nwt')
        initnwt = file

hparams = [
    hp.choice('linmeth',
        [
            {'linmeth': 1,
               'maxitinner': hp.choice('maxitinner', range(25, 1001)),
               'ilumethod': hp.choice('ilumethod', [1, 2]),
               'levfill': hp.choice('levfill', range(0, 11)),
               'stoptol': hp.uniform('stoptol', .000000000001, .00000001),
               'msdr': hp.choice('msdr', range(5, 21))
            },
            {'linmeth': 2,
                'iacl': hp.choice('iacl', [0, 1, 2]),
                'norder': hp.choice('norder', [0, 1, 2]),
                'level': hp.choice('level', range(0, 11)),
                'north': hp.choice('north', range(2, 11)),
                'iredsys': hp.choice('iredsys', [0, 1]),
                'rrctols': hp.uniform('rrctols', 0., .0001),
                'idroptol': hp.choice('idroptol', [0, 1]),
                'epsrn': hp.uniform('epsrn', .00005, .001),
                'hclosexmd': hp.uniform('hclosexmd', .00001, .001),
                'mxiterxmd': hp.choice('mxiterxmd', range(25,  1001))
            }
        ]),
    hp.uniform('headtol', .01, 5.),
    hp.uniform('fluxtol', 5000, 1000000),
    hp.choice('maxiterout', range(100, 401)),
    hp.uniform('thickfact', .000001, .0005),
    hp.choice('iprnwt', [0, 2]),
    hp.choice('ibotav', [0, 1]),
    hp.choice('options', ['SPECIFIED']),
    hp.uniform('dbdtheta', .4, 1.),
    hp.uniform('dbdkappa', .00001, .0001),
    hp.uniform('dbdgamma', 0., .0001),
    hp.uniform('momfact', 0., .1),
    hp.choice('backflag', [0, 1]),
    hp.choice('maxbackiter', range(10,51)),
    hp.uniform('backtol', 1., 2.),
    hp.uniform('backreduce', .00001, 1.),
]

NWTNUM = -1
try:
    os.mkdir(os.path.join(cwd, 'nwts'))
except:
    os.rmdir(os.path.join(cwd, 'nwts'))
    os.mkdir(os.path.join(cwd, 'nwts'))
    print('[INFO] removed previous nwts')

def inputHp2nwt(inputHp):
    global NWTNUM
    NWTNUM += 1
    with open(os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM))), 'w') as file:
        file.write(('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(inputHp[1], inputHp[2], inputHp[3], inputHp[4], inputHp[0]['linmeth'], inputHp[5],
                   inputHp[6], inputHp[7], inputHp[8], inputHp[9], inputHp[10], inputHp[11],
                   inputHp[12], inputHp[13], inputHp[14], inputHp[15])) + '\n')
    if inputHp[0]['linmeth'] == 1:
        with open(os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM))), 'a') as file:
           file.write(('{} {} {} {} {}'.format(inputHp[0]['maxitinner'], inputHp[0]['ilumethod'], inputHp[0]['levfill'],
                      inputHp[0]['stoptol'], inputHp[0]['msdr'])))
    elif inputHp[0]['linmeth'] == 2:
        with open(os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM))), 'a') as file:
           file.write(('{} {} {} {} {} {} {} {} {} {}'.format(inputHp[0]['iacl'], inputHp[0]['norder'], inputHp[0]['level'],
                      inputHp[0]['north'], inputHp[0]['iredsys'], inputHp[0]['rrctols'],
                      inputHp[0]['idroptol'], inputHp[0]['epsrn'], inputHp[0]['hclosexmd'],
                      inputHp[0]['mxiterxmd'])))
    return os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM)))

def trials2csv(trials):
    df = pd.DataFrame(trials.results).drop('loss', axis=1)
    df.to_csv(os.path.join(cwd, 'nwt_performance.csv'))

def runModel(pathtonwt, initnwt):
    # DOES NOT WORK
    with fileinput.FileInput(os.path.join(cwd, namefile), inplace=True) as file:
        for line in file:
            line.replace(os.path.split(initnwt)[-1], os.path.split(pathtonwt)[-1])
    # works
    call(['./mfnwt', namefile])

def getdata():
    return min_elapased, iterations

def objective(inputHp):
    global initnwt
    pathtonwt = inputHp2nwt(inputHp)
    runModel(pathtonwt, initnwt)
    min_elapased, iterations = getdata()
    initnwt = pathtonwt
    return {'loss': min_elapased,
            'status':  STATUS_OK,
            'eval_time': time.time(),
            'mass_balance': inputHp[1],
            'min_elapased': inputHp[2],
            'iterations': 1}

trials = Trials()
bestHp = fmin(fn=objective,
              space=hparams,
              algo=tpe.suggest,
              max_evals=25,
              trials=trials)

trials2csv(trials)
