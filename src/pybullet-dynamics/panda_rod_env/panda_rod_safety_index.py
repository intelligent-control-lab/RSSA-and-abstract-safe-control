from sys import prefix
import numpy as np
import torch as th
import time
import pickle

from panda_rod_env import PandaRodEnv
from panda_rod_utils import video_record

from learned_dynamics.panda_rod_dynamics_model import *
from learned_dynamics.panda_rod_differentiable_dynamics_chain import *
from learned_dynamics.panda_rod_dataset import PandaRodDataset

class PandaRodSafetyIndex:
    def __init__(
        self,
        env: PandaRodEnv,
        d_min=0.05,
        eta=1,
        k_v=1,
        lamda=0.0,
        use_asnyc_model=True,
    ):
        # self.env = PandaRodEnv()    # when debugging done, change it to self.env = env
        self.env = env

        self.d_min = d_min
        self.eta = eta
        self.dT = self.env.dT
        self.k_v = k_v
        self.lamda = lamda

        # self.dims = PandaRodDataset(data_path='./src/pybullet-dynamics/panda_rod_env/model/env_data_test_bn_1/').get_dim()
        self.dims = {'f_dim': 14, 'g_flat_dim': 14*7, 'u_dim': 7}
        self.f_dim = self.dims['f_dim']
        self.g_flat_dim = self.dims['g_flat_dim']
        self.u_dim = self.dims['u_dim']
        self.x_dim = self.f_dim

        self.use_async_model = use_asnyc_model
        self.num_models = 0

    def load_nn_ensemble(
        self, 
        prefix,
        n_ensemble, 
        num_layer, 
        hidden_dim,
        mode='partial_model',
    ):  
        self.num_models = n_ensemble
        self.mode = mode


        if self.mode == 'partial_model':
            self.models = [PandaRodDifferentiableDynamicsChain(self.env.physics_client_id, device, self.env.robot.id,) 
                            for _ in range(n_ensemble)]
            for i in range(n_ensemble):
                load_path = prefix + str(i) + '_best.pth'
                self.models[i].load_state_dict(th.load(load_path))
                self.models[i].eval()

        elif self.mode == 'unknown_model':
            if not self.use_async_model:
                self.models = [PandaRodDynamicsModel(num_layer, hidden_dim, self.dims) for _ in range(n_ensemble)]
                
                for i in range(n_ensemble):
                    load_path = prefix + str(i) + '_best.pth'
                    self.models[i].load_state_dict(th.load(load_path, map_location=th.device('cpu')))
                    self.models[i].eval()
            else:
                self.models = []
                for i in range(n_ensemble):
                    self.models.append({
                        'f': PandaRodDynamicsModelF(num_layer, hidden_dim, self.dims),
                        'g': PandaRodDynamicsModelG(num_layer, hidden_dim, self.dims),
                    })
                    f_load_path = prefix + 'f/' + str(i) + '_best.pth'
                    g_load_path = prefix + 'g/' + str(i) + '_best.pth'
                    self.models[i]['f'].load_state_dict(th.load(f_load_path))
                    self.models[i]['g'].load_state_dict(th.load(g_load_path))
                    self.models[i]['f'].eval()
                    self.models[i]['g'].eval()
        

    def ensemble_inference(self, Xr):
        fxs = np.zeros((self.num_models, self.x_dim))
        gxs = np.zeros((self.num_models, self.g_flat_dim))

        with th.no_grad():
            if self.mode == 'partial_model':
                q = th.tensor(Xr[:self.env.robot.dof].reshape(1, -1)).to(device).float()
                dq = th.tensor(Xr[self.env.robot.dof:].reshape(1, -1)).to(device).float()
                for i in range(self.num_models):
                    fx, gx = self.models[i].get_f_and_g_flat(q, dq)
                    fxs[i, :] = fx
                    gxs[i, :] = gx

            elif self.mode == 'unknown_model':
                if not self.use_async_model:
                    for i in range(self.num_models):
                        fx, gx = self.models[i].get_f_and_g_flat(Xr)
                        fxs[i, :] = fx
                        gxs[i, :] = gx
                else:
                    for i in range(self.num_models):
                        fx = self.models[i]['f'].get_f(Xr)
                        gx = self.models[i]['g'].get_g_flat(Xr)
                        fxs[i, :] = fx
                        gxs[i, :] = gx

        return fxs, gxs

    def phi(self, Mh):
        '''
        Just need to parse Mh, because Xr and Mr both can be obtained directly from self.env.
        Also, there might be multiple obstacles.
        '''
        dM_p = self.env.Mr[:3, 0] - Mh[:3, 0]
        dM_v = self.env.Mr[3:, 0] - Mh[3:, 0]

        d = np.linalg.norm(dM_p)
        d = 1e-3 if d == 0 else d

        dot_d = (dM_p.T @ dM_v).item() / d
        dot_d = 1e-3 if dot_d == 0 else dot_d
        phi = self.d_min**2 + self.lamda * self.dT + self.eta * self.dT - d**2 - self.k_v * dot_d

        self.d = d
        self.dot_d = dot_d
        self.Phi = phi

        return phi

    def compute_grad(self, Mh):
        '''
        Just need to parse Mh, because Xr and Mr both can be obtained directly from self.env.
        Also, there might be multiple obstacles.
        Note that p_Mr_p_Xr can be obtained in self.env.
        '''
        dM_p = self.env.Mr[:3, 0] - Mh[:3, 0]
        dM_v = self.env.Mr[3:, 0] - Mh[3:, 0]

        d = np.linalg.norm(dM_p)
        d = 1e-3 if d == 0 else d

        dot_d = (dM_p.T @ dM_v).item() / d
        dot_d = 1e-3 if dot_d == 0 else dot_d

        p_d_p_Mr = np.zeros((1, 6))
        p_d_p_Mr[:, :3] = dM_p / d
        p_d_p_Xr = p_d_p_Mr @ self.env.p_Mr_p_Xr

        p_dot_d_p_Mr = np.zeros((1, 6))
        p_dot_d_p_Mr[:, :3] = dM_v.T / d - (dM_p.T @ dM_v).item() * dM_p.T / (d**3)
        p_dot_d_p_Mr[:, 3:] = dM_p.T / d
        p_dot_d_p_Xr = p_dot_d_p_Mr @ self.env.p_Mr_p_Xr

        return d, dot_d, p_d_p_Xr, p_dot_d_p_Xr

    def render_image(self):
        image_1 = self.env.render(
            height=512, width=512,
            cam_dist=2,
            camera_target_position=[0, 0.5, 0],
            cam_yaw=30, cam_pitch=-30, cam_roll=0, 
        )
        image_2 = self.env.render(
            height=512, width=512,
            cam_dist=2,
            camera_target_position=[0, 0, 0],
            cam_yaw=120, cam_pitch=-50, cam_roll=0,
        )
        image = np.concatenate((image_1, image_2), axis=1)
        # visually split two perspectives
        image[:, 512, :] = 0
        # cv2.imwrite('./src/pybullet-dynamics/panda_rod_env/imgs/test.jpg', image)
        return image





