import numpy as np

from panda_rod_env import PandaRodEnv
from panda_rod_utils import get_largest_singular_value, to_np

class PandaRodDifferentMassEnvs:
    def __init__(
        self,
        mass_list=[0.2, 0.4, 0.6, 0.8, 1.0],
    ):
        self.envs = []
        self.mass_list = mass_list
        for mass in self.mass_list:
            self.envs.append(PandaRodEnv(
                change_end_rod_mass=True,
                end_rod_mass=mass
            ))

        for env in self.envs:
            env.empty_step()
        
    def get_fs(self, q, dq):
        fs = []
        gs = []

        for env in self.envs:
            f, g = env.compute_true_f_and_g(q, dq)
            fs.append(f)
            gs.append(g)
        
        fs = to_np(fs)
        gs = to_np(gs)

        return fs, gs
    
    def get_f_sigma(self, fs):
        return np.cov(fs.T)
    
    def get_g_sigma(self, gs):
        return np.cov(gs.reshape(gs.shape[0], -1).T)
    


if __name__ == '__main__':
    env = PandaRodEnv()
    diff_envs = PandaRodDifferentMassEnvs(mass_list=np.linspace(0.1, 1.0, 8))

    env.empty_step()
    for i in range(100):
        q, dq = env.robot.get_joint_states()
        fs, gs = diff_envs.get_fs(q, dq)
         
        f_sigma = diff_envs.get_f_sigma(fs)
        g_sigma = diff_envs.get_g_sigma(gs)

        f_v = get_largest_singular_value(np.array([f_sigma]))
        g_v = get_largest_singular_value(np.array([g_sigma]))
        print(f'f_v: {f_v}, g_v: {g_v}')

        env.step([0.0] * 7)


