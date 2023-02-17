from matplotlib import pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from loguru import logger

from SCARA_env.SCARA_parameter_learning import SCARAParameterLearningEnv
from SegWay_env.SegWay_parameter_learning import SegWayParameterLearningEnv
from SegWay_env.SegWay_env import SegWayEnv

solvers.options['show_progress'] = False

class SafetyIndex:
    def __init__(
        self,
        env: SegWayParameterLearningEnv,
        safety_index_params={
            'k_v': 0.5,
            'eta': 0.05,
        },
        use_true_param=False,
        gamma=0.1,
        rho=0.0,
    ):
        self.env = env
        self.safety_index_params = safety_index_params
        self.use_true_param = use_true_param
        self.gamma = gamma
        self.rho = rho

    def safe_control(self, u_ref):
        def filter_u(u, phi_list, p_phi_p_Xr_list):
            LfP_list = []
            LgP_list = []
            if self.use_true_param:
                for phi, p_phi_p_Xr in zip(phi_list, p_phi_p_Xr_list):
                    LfP = p_phi_p_Xr @ self.env.f
                    LgP = p_phi_p_Xr @ self.env.g
                    LfP_list.append(LfP)
                    LgP_list.append(LgP)
            else:
                true_param = self.env.get_param()
                param_mean, _ = self.env.param_pred()
                self.env.set_param(param_mean)
                for phi, p_phi_p_Xr in zip(phi_list, p_phi_p_Xr_list):
                    LfP = p_phi_p_Xr @ self.env.f
                    LgP = p_phi_p_Xr @ self.env.g
                    LfP_list.append(LfP)
                    LgP_list.append(LgP)
                self.env.set_param(true_param)
                
            n = len(u_ref)

            # a @ u <= b
            a_list = []
            b_list = []
            for LfP, LgP in zip(LfP_list, LgP_list):
                if phi >= 0:
                    a = LgP
                    b = -LfP - np.array([[self.gamma * phi]])
                else:
                    a = np.zeros((1, n))
                    b = np.ones((1, 1))
                a_list.append(a)
                b_list.append(b)
            a_list = np.vstack(a_list)
            b_list = np.vstack(b_list)

            # add additional constraints:
            # I @ u <= u_max
            # -I @ u <= -u_min
            # a @ u - u_slack <= 0
            # u_slack >= 0
            a_slack = np.zeros((1, n + 1))
            a_slack[0, -1] = -1
            b_slack = np.zeros((1, 1))
            
            
            G = np.vstack([
                np.hstack([a_list, -np.ones((len(a_list), 1))]),
                np.eye(n, n + 1),
                -np.eye(n, n + 1),
                a_slack,
            ])
            h = np.vstack([
                b_list,
                np.asanyarray(self.env.u_limit['high']).reshape(-1, 1),
                -np.asanyarray(self.env.u_limit['low']).reshape(-1, 1),
                b_slack,
            ])

            # object:
            # 0.5 * u.T @ P @ u + q.T @ u
            w = 1000
            P = np.eye(n + 1).astype(float) 
            P[-1, -1] = w
            q = np.vstack([
                -u_ref.reshape(-1, 1),
                np.zeros((1, 1)),
            ])

            sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
            u = np.squeeze(sol_obj['x'])
            if np.abs(u[-1]) > 1e-3:
                # logger.debug(f'Safety index infeasible: {u}')
                self.if_infeasible = True
            else:
                self.if_infeasible = False
            # print(u)
            u = u[:-1]
            return u
        
        phi = self.env.get_phi(self.safety_index_params) + self.rho
        self.phi = phi
        p_phi_p_Xr = self.env.get_p_phi_p_Xr(self.safety_index_params)
        
        phi_list = [phi]
        p_phi_p_Xr_list = [p_phi_p_Xr]
        try:
            phi = self.env.get_phi_another(self.safety_index_params) + self.rho
            p_phi_p_Xr = self.env.get_p_phi_p_Xr_another(self.safety_index_params) + self.rho
            phi_list.append(phi)
            p_phi_p_Xr_list.append(p_phi_p_Xr)
        except:
            pass
        u = filter_u(u_ref, phi_list, p_phi_p_Xr_list)
        return u


if __name__ == '__main__':
    # env = SCARAParameterLearningEnv()
    env = SegWayEnv()
    ssa = SafetyIndex(env)

    # q_way_points = np.linspace(start=[np.pi/2 - 0.05, 0.0], stop=[-np.pi/3, -np.pi], num=400)
    # for i, q_d in enumerate(q_way_points):
    #     u = env.robot.computed_torque_control(q_d=q_d)

    #     for _ in range(3):
    #         u = ssa.safe_control(u)
    #         env.step(u)
    #         if env.detect_collision():
    #             print('COLLISION!')
    #         phi = env.get_phi(ssa.safety_index_params)
    #         print(phi)
    #     env.render(img_name=str(i) + '.jpg', save_path='./src/pybullet-dynamics/SCARA_env/imgs/safety_index/')

    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    a_list = []
    dr_list = []
    u_list = []
    for i in range(960):
        u = env.robot.PD_control(q_d, dq_d)
        u = ssa.safe_control(u)
        print(env.Xr)
        u_list.append(u[0])
        a_list.append(env.robot.q[1])
        dr_list.append(env.robot.dq[0])
        env.step(u)

    plt.plot(a_list)
    plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/safety_index_test/a.jpg')
    plt.close()
    plt.plot(dr_list)
    plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/safety_index_test/dr.jpg')
    plt.close()
    plt.plot(u_list)
    plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/safety_index_test/u.jpg')
    plt.close()