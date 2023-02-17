import numpy as np
from cvxopt import matrix, solvers
from loguru import logger
from datetime import datetime
import os

from RSSA_safety_index_learning import RSSASafetyIndexLearning as LatentSafetyIndexLearning
from panda_rod_env.panda_latent_SI_learning_env import PandaLatentSILearningEnv
from fly_inv_pend_env.fly_inv_pend_SI_learning_env import FlyingInvertedPendulumLatentSILearningEnv

def turn_on_log(log_root_path):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.mkdir(log_path)
    logger.add(log_path + '/log.log')
    return log_path



if __name__ == '__main__':
    turn_on_log(log_root_path='./src/pybullet-dynamics/fly_inv_pend_env/log/')
    env = FlyingInvertedPendulumLatentSILearningEnv(device='cuda')
    SI_learning = LatentSafetyIndexLearning(
        env=env, 
        epoch=10,
        elite_ratio=0.1,
        populate_num=100,
        init_sigma_ratio=0.3,
        noise_ratio=0.01,
        init_params=env.init_params,
        param_bounds={'a_1': [0.1, 5.0], 'a_2': [0.1, 5.0], 'a_3': [0.001, 1.0]},
    )
    SI_learning.learn()
        