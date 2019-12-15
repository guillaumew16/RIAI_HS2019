# should have more or less the same interface as art.attacks.Attack (except for initialization)

"""
Args:
    eps_step_ratio (float): eps_step / eps, where eps_step is the step size in FGSM (see adv.py)
    k (int): nb of projections in PGD
    num_restarts (int): number of times to try PGD with a different random seed
"""
class MyPgdAttacker():
    def __init__(self, eps_step_ratio, k, num_restarts):
        raise NotImplementedError

    