import time
from hyperopt import fmin, tpe, hp, STATUS_OK, trials

space = hp.choice(''
)

def inputHp2csv(inputHp):
    return csvpath

def objective(inputHp):
    model(inputHp2csv(inputHp))

    read model output
    output.accuracy
    return {'loss': run.accuracy,
            'status':  STATUS_OK,
            'eval_time': time.time(),
            'mass_balance': run.massbalance}

trials = Trials()
bestHp = fmin(fn=objective,
              space=hparams,
              algo=tpe.suggest,
              maxevals=50,
              trials=trials)

print(bestHp)
