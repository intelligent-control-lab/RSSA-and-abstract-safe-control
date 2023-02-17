import numpy as np
import time
from cvxopt import matrix, solvers

from panda_rod_safety_index import PandaRodSafetyIndex
from panda_rod_env import PandaRodEnv
from panda_latent_env import PandaLatentEnv
from panda_rod_utils import *
from online_adaptation.panda_rod_bayesian_regression import PandaRodBayesianRegression
from online_adaptation.panda_rod_rls import *
from online_adaptation.panda_rod_linear_regression import PandaRodLinearRegression

solvers.options['show_progress'] = False

monitor = Monitor()
# monitor.start()

class PandaRodSSA(PandaRodSafetyIndex):
    def __init__(
        self, 
        env: PandaLatentEnv, 
        prefix,
        d_min=0.05, 
        eta=0.0, 
        k_v=1.0, 
        lamda=0.0,
        use_gt_dynamics=True,
        n_ensemble=10,
        num_layer=3,
        hidden_dim=256,
    ):
        super().__init__(env, d_min, eta, k_v, lamda)
        self.use_gt_dynamics = use_gt_dynamics
        self.prefix = prefix
        if not self.use_gt_dynamics:
            self.load_nn_ensemble(
                prefix=prefix,
                n_ensemble=n_ensemble,
                num_layer=num_layer,
                hidden_dim=hidden_dim,
            )
        
        self.u_dim = env.robot.dof
        self.x_dim = env.robot.dof * 2
        
        self.last_M = env.M
    
    def nn_fg(self, Xr):
        fxs, gxs = self.ensemble_inference(Xr)
        fx_mu = np.mean(fxs, axis=0).reshape(self.f_dim, 1)
        gx_mu = np.mean(gxs, axis=0).reshape(self.x_dim, self.u_dim)

        # TODO: compare nn_fg and true fg
        # compare(fx_mu, self.env.f, ratio_flag=False)
        # compare(gx_mu, self.env.g, ratio_flag=False)
        # compare(
        #     fx_mu + gx_mu @ self.env.target_torque, 
        #     self.env.f + self.env.g @ self.env.target_torque
        # )
        # END TODO

        return fx_mu, gx_mu
    
    def grad_phi(self, Xr, Mh):
        d, dot_d, p_d_p_Xr, p_dot_d_p_Xr = self.compute_grad(Mh)
        p_phi_p_Xr = -2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr

        if self.use_gt_dynamics:
            fx = self.env.f
            gx = self.env.g
        else:
            fx, gx = self.nn_fg(Xr)
        
        LfP = p_phi_p_Xr @ fx
        LgP = p_phi_p_Xr @ gx
        return LfP, LgP
    
    def generate_safety_con(self, Xr, Mh):
        Xr = np.vstack(Xr)
        Mh = np.vstack(Mh)

        p = self.phi(Mh)
        LfP, LgP = self.grad_phi(Xr, Mh)

        # a*u <= b
        a = LgP
        # b = -LfP - self.eta - self.lamda
        b = -LfP.item() - self.eta - self.lamda

        if p < 0:
            return np.zeros_like(a), np.ones_like(b)
        else:
            print('UNSAFE!')
            return a, b

    def safe_control(self, uref):
        ''' safe control
        Input:
            uref: reference control 
        '''
        # if the next state is unsafe, then trigger the safety control 

        # solve QP 
        # Compute the control constraints
        # Get f(x), g(x); note it's a hack for scalar u

        # compute the QP 
        # objective: 0.5*u*P*u + q*u
        
        # constraint
        # A u <= b  
        # u <= umax 
        # -u <= umax
        # c >= 0, c is the slack variable 

        Xr = self.env.Xr

        Abs = [self.generate_safety_con(Xr, Mh) for Mh in self.env.obstacles if self.phi(Mh) > 0]
        if len(Abs) > 0:
            A = np.vstack([Ab[0] for Ab in Abs])
            b = np.vstack([Ab[1] for Ab in Abs])
        else:
            A = np.zeros((1, self.u_dim))
            b = np.ones((1, 1))

        ### TODO: test SSA - use prior method
        # uref = uref.reshape(-1, 1)
        # if self.phi(self.env.Mh) <= 0 or (A @ uref).item() < b.item():
        #     u = uref
        # else:   
        #     u = uref - ((A @ uref - b).item() * A.T) / (A @ A.T).item()
        # u = np.squeeze(u)
        # return u
        ### END TODO

        ### TODO: test SSA - without slack variable and limits on u
        # G = A
        # h = b
        # P = np.eye(self.u_dim)
        # q = -uref.reshape((-1, 1))
        # sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        # u = np.array(sol_obj['x'])
        # return u
        ### END TODO

        A_slack = np.zeros((1, self.u_dim + 1))
        A_slack[0, -1] = -1
        b_slack = np.zeros((1, 1))

        G = np.vstack([
            np.hstack([A, -np.ones((A.shape[0], 1))]),
            -np.eye(self.u_dim, self.u_dim + 1),
            np.eye(self.u_dim, self.u_dim + 1),
            A_slack,
        ]).astype(float)
        h = np.vstack([
            b,
            self.env.max_u.reshape((-1, 1)),
            self.env.max_u.reshape((-1, 1)),
            b_slack,
        ])

        w = 1e8
        P = np.eye(self.u_dim + 1).astype(float)
        P[-1, -1] = w
        q = np.vstack([-uref.reshape((-1, 1)), [0.0]]).astype(float)

        sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
        u = np.array(sol_obj['x'])
        # print(u)
        u = u[:-1]
        return u

    def compute_monitor_data(self, Xr, Mh):
        Xr = np.vstack(Xr)
        Mh = np.vstack(Mh)

        f_nn, g_nn = self.nn_fg(Xr)

        self.f_nn = f_nn
        self.g_nn = g_nn

        d, _, p_d_p_Xr, p_dot_d_p_Xr = self.compute_grad(Mh)
        p_phi_p_Xr = -2 * d * p_d_p_Xr - self.k_v * p_dot_d_p_Xr
        self.LfP_nn = (p_phi_p_Xr @ f_nn).item()
        self.LgP_nn = p_phi_p_Xr @ g_nn

        self.LfP = (p_phi_p_Xr @ self.env.f).item()
        self.LgP = p_phi_p_Xr @ self.env.g
        
    def latent_safe_control(self, uref, env: PandaLatentEnv):
        dot_d = env.z[1, 0]
        M = env.M
        dot_M = (M - self.last_M) / self.dT
        self.eta = -self.k_v * dot_d * dot_M / M**2
        self.last_M = M
        return self.safe_control(uref)
        


def evaluate_latent_ssa(
    k_v,
    goal_pose, obstacle_pose, 
    robot_file_path='src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf', dof=7, 
    step_num=960,
):
    env = PandaLatentEnv(
        render_flag=True, 
        goal_pose=goal_pose, obstacle_pose=obstacle_pose, 
        robot_file_path=robot_file_path, dof=dof,
    )
    env.update_latent_info()
    ssa = PandaRodSSA(
        env, 
        prefix='./src/pybullet-dynamics/panda_rod_env/model/env_diff/', 
        use_gt_dynamics=True,
        n_ensemble=1,
        k_v=k_v,
    )
    for _ in range(step_num):
        u = env.compute_naive_torque()
        env.update_latent_info()
        u = ssa.latent_safe_control(u, env)
        u = np.squeeze(u)
        env.step(u)
        if env.z[0, 0] <= env.d_min:
            print('---COLLSION!---')
        


if __name__ == '__main__':
    k_v = 133
    goal_pose_lim = {'low': [0.6, 0.2, 0.3], 'high': [0.8, 0.4, 0.5]}
    obstacle_pose_lim = {'low': [0.35, 0.0, 0.45], 'high': [0.55, 0.2, 0.65]}
    for i in range(100):
        print(i)
        goal_pose = np.random.uniform(low=goal_pose_lim['low'], high=goal_pose_lim['high'])
        obstacle_pose = np.random.uniform(low=obstacle_pose_lim['low'], high=obstacle_pose_lim['high'])
        evaluate_latent_ssa(
            k_v=k_v,
            goal_pose=goal_pose, obstacle_pose=obstacle_pose,
            robot_file_path='src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf', dof=7, 
        )
        import ipdb; ipdb.set_trace()
    
    
    # env = PandaRodEnv(render_flag=False, goal_pose=[0.65, 0.1, 0.5], obstacle_pose=[0.5, 0.0, 0.4])
    # env = PandaRodEnv(render_flag=True, goal_pose=[0.65, 0.1, 0.5], obstacle_pose=[0.5, 0.0, 0.4])
    # env = PandaLatentEnv(
    #     render_flag=False, 
    #     goal_pose=[0.7, 0.3, 0.4], 
    #     obstacle_pose=[0.45, 0.1, 0.55], 
    #     robot_file_path='src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf',
    #     dof=7,
    # )
    # env.update_latent_info()
    # ssa = PandaRodSSA(
    #     env, 
    #     prefix='./src/pybullet-dynamics/panda_rod_env/model/env_diff/', 
    #     use_gt_dynamics=True,
    #     n_ensemble=1,
    #     k_v=8.0,
    # )
    
    # images = []
    # env.reset()
    # for i in range(960):
    #     print(i)
    #     u = env.compute_naive_torque()
    #     env.update_latent_info()
    #     u = ssa.latent_safe_control(u, env)
    #     u = np.squeeze(u)
    #     print(env.robot.get_end_eff_pose())
    #     env.step(u)
    #     time.sleep(1/240 * 10)
        # rgb = env.render_image()
        # images.append(rgb)
    # video_record(movie_name='./src/pybullet-dynamics/panda_rod_env/movies/latent_example/test.mp4', images=images)
    
    # rls = PandaRodRLSLinearNullL2(
    #     lbd=0.998,
    #     initial_W=np.zeros((ssa.x_dim, ssa.x_dim + ssa.u_dim)),
    #     initial_lr=10000,
    #     beta=1e-5,
    # )
    # # lr = PandaRodLinearRegression()

    # monitor.update(ssa_args=ssa.__dict__)
    # monitor.update(env_args=env.__dict__)
    # monitor.update(adapt_args=rls.__dict__)

    # images = []

    # env.reset()
    # for i in range(2000):
    #     print(i)

    #     u = env.compute_underactuated_torque()
    #     # print(f'before: {to_np(u)}')
    #     u = ssa.safe_control(u)
    #     env.target_torque = u
    #     # print(f'after: {to_np(u)}')

    #     _, _, _, _ = env.step(u)

    #     # img = ssa.render_image()
    #     # images.append(img)



    #     ssa.compute_monitor_data(env.Xr, env.Mh)
    #     dot_Xr_residual = env.dot_Xr.reshape(-1, 1) - (ssa.f_nn + ssa.g_nn @ u.reshape(-1, 1))
    #     rls.adapt(dot_x_residual=dot_Xr_residual, x=env.Xr, u=u)
    #     f_residual, g_residual = rls.pred(env.Xr)
    #     monitor.update(
    #         f=env.f, g=env.g, dot_Xr=env.dot_Xr, u=u, LfP=ssa.LfP, LgP=ssa.LgP,
    #         f_nn=ssa.f_nn, g_nn=ssa.g_nn, LfP_nn=ssa.LfP_nn, LgP_nn=ssa.LgP_nn,
    #         d=ssa.d, dot_d=ssa.dot_d, Phi=ssa.Phi,
    #         f_res=f_residual, g_res=g_residual,
    #     )
    #     # print(f'd: {ssa.d}, dot_d: {ssa.dot_d}, Phi: {ssa.Phi}')
    #     print(f'f_res: {to_np(f_residual).max()}')
    #     print(f'g_res: {to_np(g_residual).max()}')
    #     # print(f'f: {env.f}')
    #     # print(f'f_nn: {ssa.f_nn}')
    #     compare(env.g, ssa.g_nn)


    #     if env.if_success():
    #         print('--SUCCESS!--')
    #         break

    #     if env.robot.get_contact_info():
    #         print('--COLLSION!--')
    #         break
            
    #     time.sleep(1/240)
    
    # # video_record('./src/pybullet-dynamics/panda_rod_env/movies/ssa_change_mass_1.mp4', images)
    # monitor.close()
