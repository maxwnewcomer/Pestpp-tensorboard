import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

def inputHp2nwt(inputHp):
    return nwtpath

def objective(inputHp):
    # model(inputHp2nwt(inputHp))
    #
    # read model output
    # output.accuracy
    return {'loss': inputHp[2],
            'status':  STATUS_OK,
            'eval_time': time.time(),
            'mass_balance': inputHp[1]}

trials = Trials()
bestHp = fmin(fn=objective,
              space=hparams,
              algo=tpe.suggest,
              max_evals=50,
              trials=trials)

print(bestHp)
