import os
import time
import fileinput
from subprocess import call
from shutil import copyfile, rmtree
import ray
# from ray.tune import run
# from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.tune.suggest.hyperopt import HyperOptSearch
import ray.tune as tune
from ray.tune.hpo_scheduler import HyperOptScheduler
import pandas as pd
from hyperopt import fmin, rand, tpe, hp, STATUS_OK, Trials

cwd = os.getcwd()
for file in os.listdir(cwd):
    if file.endswith('.nam'):
        namefile = file
    if file.endswith('.list'):
        listfile = file
    if file.endswith('.nwt'):
        initnwt = file

hparams = [
    hp.choice('linmeth',
        [
            {'linmeth': 1,
               'maxitinner': hp.quniform('maxitinner', 25, 1000, 1),
               'ilumethod': hp.choice('ilumethod', [1, 2]),
               'levfill': hp.quniform('levfill', 0, 10, 1),
               'stoptol': hp.uniform('stoptol', .000000000001, .00000001),
               'msdr': hp.quniform('msdr', 5, 20, 1)
            },
            {'linmeth': 2,
                'iacl': hp.choice('iacl', [0, 1, 2]),
                'norder': hp.choice('norder', [0, 1, 2]),
                'level': hp.quniform('level', 0, 10, 1),
                'north': hp.quniform('north', 2, 10, 1),
                'iredsys': hp.choice('iredsys', [0, 1]),
                'rrctols': hp.uniform('rrctols', 0., .0001),
                'idroptol': hp.choice('idroptol', [0, 1]),
                'epsrn': hp.uniform('epsrn', .00005, .001),
                'hclosexmd': hp.uniform('hclosexmd', .00001, .001),
                'mxiterxmd': hp.quniform('mxiterxmd', 25,  100, 1)
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
    hp.quniform('maxbackiter', 10, 50, 1)),
    hp.uniform('backtol', 1., 2.),
    hp.uniform('backreduce', .00001, 1.),
]

NWTNUM = -1
try:
    os.mkdir(os.path.join(cwd, 'nwts'))
except:
    rmtree(os.path.join(cwd, 'nwts'))
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
    # print('[INFO] pulling nwt from', os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM))))
    return os.path.join(cwd, 'nwts', ('nwt_{}.nwt'.format(NWTNUM)))

def trials2csv(trials):
    df = pd.DataFrame(trials.results).drop('loss', axis=1)
    df.to_csv(os.path.join(cwd, 'nwt_performance.csv'))

def runModel(pathtonwt, initnwt):
    copyfile(pathtonwt, initnwt)
    call(['./mfnwt', namefile])

def getdata():
    mbline, timeline, iterline = '', '', ''
    with open(listfile, 'r') as file:
        mbfound = False
        for line in reversed(list(file)):
            if 'Error in Preconditioning' in line:
                return 99999999, -1, -1
            if 'PERCENT DISCREPANCY' in line and mbfound == False:
                mbfound = True
                mbline = line
            if 'Elapsed run time' in line:
                timeline = line
            if 'OUTER ITERATIONS' in line:
                iterline = line
                break
    for val in mbline.split(' '):
        try:
            mass_balance = float(val)
            break
        except:
            pass
    foundmin, foundsec = False, False
    min, sec = 0, 0
    for val in timeline.split(' '):
        if foundmin == False:
            try:
                min = float(val)
                foundmin = True
            except:
                pass
        else:
            try:
                sec = float(val)
                foundsec = True
                break
            except:
                pass
    if foundsec:
        sec_elapsed = min * 60 + sec
    else:
        sec_elapsed = min

    for val in iterline.split(' '):
        try:
            iterations = float(val)
            break
        except:
            pass

    print('[SECONDS]:', sec_elapsed)
    print('[MASS BALANCE]:', mass_balance)
    print('[TOTAL ITERATIONS]:', iterations)
    return sec_elapsed, iterations, mass_balance

# def objective(inputHp):
#     global initnwt
#     pathtonwt = inputHp2nwt(inputHp)
#     runModel(pathtonwt, initnwt)
#     sec_elapsed, iterations, mass_balance = getdata()
#     return {'loss': sec_elapsed + mass_balance ** 2,
#             'status':  STATUS_OK,
#             'eval_time': time.time(),
#             'mass_balance': mass_balance,
#             'sec_elapsed': sec_elapsed,
#             'iterations': iterations}
def objective(config, reporter):
     global initnwt
    pathtonwt = inputHp2nwt(inputHp)
    runModel(pathtonwt, initnwt)
    sec_elapsed, iterations, mass_balance = getdata()
    reporter(sec_elapsed=sec_elapsed,
             iterations=iterations,
             mass_balance=mass_balance)
if __name__ == '__main__':
    # trials = Trials()
    # bestHp = fmin(fn=objective,
    #               space=hparams,
    #               algo=tpe.suggest,
    #               max_evals=100,
    #               trials=trials)
    # bestRandHp = fmin(fn=objective,
    #               space=hparams,
    #               algo=rand.suggest,
    #               max_evals=100,
    #               trials=trials)
    # trials2csv(trials)
    ray.init()
    tune.register_trainable('objective', objective)
    tune.run_experiments({"experiment": {
        "run": "objective",
        "repeat": 100,
        "config": {"space": hparams}}}, scheduler=HyperOptScheduler())
