from matplotlib import pyplot as plt
import numpy as np
import yaml
from loguru import logger
import time
from datetime import datetime
import os
import shutil
import pickle
from tqdm import tqdm
from scipy import stats
from interval import interval, inf

from RSSA_safety_index import SafetyIndex
from MMRSSA_additive import MMAddRSSA
from MMRSSA_multiplicative import MMMulRSSA
from MMRSSA_gaussian_multiplicative import GaussianMulRSSA
from MMRSSA_gaussian_additive import GaussianAddRSSA
from RSSA_utils import Monitor
from MMRSSA_utils import draw_ellipsoid, draw_rectangle

from SegWay_env.SegWay_multimodal_env import SegWayAdditiveNoiseEnv, SegWayMultiplicativeNoiseEnv
from SCARA_env.SCARA_utils import draw_GP_confidence_interval
from SegWay_env.SegWay_utils import generate_gif


def evaluate_in_MM_SegWay(
    rssa_types,
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_multimodal_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/evaluation/',
    num_steps=100,
    render=False
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.makedirs(log_path)
    shutil.copy(src=yaml_path, dst=log_path + '/SegWay_multimodal_params.yaml')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SegWay_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SegWay_kwargs['robot_kwargs']
    safe_control_kwargs = SegWay_kwargs['safe_control_kwargs']
    speed_kwargs = SegWay_kwargs['speed_kwargs']
    
    Env = SegWayAdditiveNoiseEnv if 'additive' in rssa_types[0] else SegWayMultiplicativeNoiseEnv
    print(f'Env: {Env.__name__}')
    env = Env(
            dt=robot_kwargs['dt'],
            K_m=robot_kwargs['K_m'],
            K_b=robot_kwargs['K_b'],
            m_0=robot_kwargs['m_0'],
            m=robot_kwargs['m'],
            J_0=robot_kwargs['J_0'],
            L=robot_kwargs['L'],
            l=robot_kwargs['l'],
            R=robot_kwargs['R'],
            g=robot_kwargs['g'],
            q_limit=robot_kwargs['q_limit'],
            dq_limit=robot_kwargs['dq_limit'],
            u_limit=robot_kwargs['u_limit'],
            a_safe_limit=robot_kwargs['a_safe_limit'],
        )   
    
    q_d = np.asanyarray(speed_kwargs['q_d'])
    dq_d = np.asanyarray(speed_kwargs['dq_d'])    
    store_data = {}
    # rssa_type_list = safe_control_kwargs['rssa_types']
    for rssa_type in rssa_types:
        print(f'Now evaluating: {rssa_type}')
        env.reset()
        env.robot.q = np.asanyarray(speed_kwargs['q_init'])
        
        rssa = get_rssa(rssa_type, env, safe_control_kwargs)
        monitor = Monitor()
        for i in tqdm(range(num_steps)):
            u_ref = env.robot.PD_control(q_d=q_d, dq_d=dq_d)
            if rssa is not None:
                u = rssa.safe_control(u_ref)
            else:
                u = u_ref
            env.step(u)
            monitor.update(
                q=env.robot.q,
                dq=env.robot.dq,
                u_ref=u_ref,
                u=u,
                dis_a_limit=env.a_safe_limit['high'] - env.robot.q[1],
            )
            if render:
                env.render(img_name=str(i) + '.jpg', save_path=f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/')
        store_data[rssa_type] = monitor.data
        if render:
            generate_gif(rssa_type + '.gif', f'./src/pybullet-dynamics/SegWay_env/imgs/mm_evaluate/{rssa_type}/',
                        f'./src/pybullet-dynamics/SegWay_env/movies/mm_evaluate/', num_fig=num_steps)
    
    with open(log_path + '/SegWay_safe_control.pkl', 'wb') as file:
        pickle.dump(store_data, file)
    pkl_path = log_path + '/SegWay_safe_control.pkl'    
    return pkl_path, log_path
    
def get_rssa(rssa_type, env, safe_control_kwargs):
    param_dict = safe_control_kwargs['param_dict']
    if 'none' in rssa_type:
        rssa = None
    elif rssa_type == 'gaussian_additive_mmrssa':
        rssa = GaussianAddRSSA(
            env=env,
            safety_index_params=param_dict,
            p_gaussian=safe_control_kwargs['p_gaussian'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'gaussian_multiplicative_mmrssa':
        rssa = GaussianMulRSSA(
            env=env,
            safety_index_params=param_dict,
            p_gaussian=safe_control_kwargs['p_gaussian'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    elif rssa_type == 'additive_mmrssa':
        rssa = MMAddRSSA(
            env=env,
            safety_index_params=param_dict,
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
        )
    elif rssa_type == 'multiplicative_mmrssa':
        rssa = MMMulRSSA(
            env=env,
            safety_index_params=param_dict,
            p_init=safe_control_kwargs['p_init'],
            sample_points_num=safe_control_kwargs['sample_points_num'],
            gamma=safe_control_kwargs['gamma'],
            fast_SegWay=safe_control_kwargs['fast_SegWay'],
        )
    else:
        raise Exception('Known safe control')
    return rssa

    
def draw_phi(data, rssa_types, truncate=-1):
    for rssa_type in rssa_types:
        values = np.array(data[rssa_type]['q'])[:truncate, 1]
        plt.plot(values, label=rssa_type)
        plt.xlabel('step')
    plt.ylabel('$\phi$')
    plt.legend()
    plt.plot(np.ones_like(values)*0.1, linestyle='--', c='k', linewidth=0.75)
    plt.savefig(log_path + '/phi.png')
    plt.show()
    plt.close()

def draw_u(data, rssa_types, truncate=-1):
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    axes = [ax1, ax2, ax3]
    axes = axes[:len(rssa_types)]
    for ax, rssa_type in zip(axes, rssa_types):
        u = np.array(data[rssa_type]['u'])[:truncate]
        u_ref = np.array(data[rssa_type]['u_ref'])[:truncate]
        ax.plot(u, label='u')
        ax.plot(u_ref, label='u_ref', linestyle='--', c='k', linewidth=0.75)
        ax.set_xlabel('step')
        ax.legend()
        ax.set_ylabel(rssa_type)

    plt.savefig(log_path + '/u.png')
    plt.show()
    plt.close()

def draw_interval_in_MM_SegWay(
    env_type='additive',
    yaml_path = './src/pybullet-dynamics/SegWay_env/SegWay_multimodal_params.yaml',
    log_root_path='./src/pybullet-dynamics/SegWay_env/log/draw_interval/',
    q = [0, 0.001],
    dq = [0, 0]
):
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d__%H-%M-%S')
    log_path =log_root_path + date_time
    os.makedirs(log_path)
    shutil.copy(src=yaml_path, dst=log_path + '/SegWay_multimodal_params.yaml')
    
    with open(yaml_path, 'r', encoding='utf-8') as file:
        file_data = file.read()
    SegWay_kwargs = yaml.load(file_data, Loader=yaml.FullLoader)
    robot_kwargs = SegWay_kwargs['robot_kwargs']
    safe_control_kwargs = SegWay_kwargs['safe_control_kwargs']
    speed_kwargs = SegWay_kwargs['speed_kwargs']
    
    Env = SegWayAdditiveNoiseEnv if env_type == 'additive' else SegWayMultiplicativeNoiseEnv
    print(f'Env: {Env.__name__}')
    env = Env(
            dt=robot_kwargs['dt'],
            K_m=robot_kwargs['K_m'],
            K_b=robot_kwargs['K_b'],
            m_0=robot_kwargs['m_0'],
            m=robot_kwargs['m'],
            J_0=robot_kwargs['J_0'],
            L=robot_kwargs['L'],
            l=robot_kwargs['l'],
            R=robot_kwargs['R'],
            g=robot_kwargs['g'],
            q_limit=robot_kwargs['q_limit'],
            dq_limit=robot_kwargs['dq_limit'],
            u_limit=robot_kwargs['u_limit'],
            a_safe_limit=robot_kwargs['a_safe_limit'],
        )   
    
    fig1 = plt.figure(figsize=(5, 8))
    ax_model = fig1.add_subplot(1,1,1)

    fig2 = plt.figure(figsize=(5, 8))
    ax_u = fig2.add_subplot(1,1,1)

    env.reset()
    env.robot.q = np.asanyarray(q)
    env.robot.dq = np.array(dq)

    if env_type == 'additive':
        mmrssa: MMAddRSSA = get_rssa('additive_mmrssa', env, safe_control_kwargs)
        gaussian_rssa: GaussianAddRSSA = get_rssa('gaussian_additive_mmrssa', env, safe_control_kwargs)
        mmrssa.safe_control(np.array([0]))
        gaussian_rssa.safe_control(np.array([0]))
        for modal_param, k in zip(mmrssa.modal_params_pred, mmrssa.k_list):
            weight, mu, sigma = modal_param
            # mu = mu[2:] + env.f[2:]
            # sigma = sigma[2:, 2:]
            mu = mu[[1,3]] + env.f[[1,3]]
            sigma = sigma[[1,3]][:,[1,3]]
            draw_ellipsoid(ax_model, mu, sigma, k=k, color='green')
        f_points = gaussian_rssa.f_points
        ax_model.plot(f_points[:, 1], f_points[:, 3], 'o', color='orange', markersize=3, alpha=0.8)
        mu = np.mean(f_points[:, [1, 3]], axis=0).reshape(-1, 1)
        sigma = np.cov(f_points[:, [1, 3]].T)
        draw_ellipsoid(ax_model, mu, sigma, confidence=gaussian_rssa.p_gaussian, color='cornflowerblue')
        ax_model.set_aspect('equal')
        ax_model.set_xlabel('$\mathrm{f_2(rad)}$')
        ax_model.set_ylabel('$\mathrm{f_4(rad/s)}$')

        # draw feasible u
        mm_u_limit = [-40, 40]
        gaussian_u_limit = [-40, 40]
        if mmrssa.grad_phi_mul_g>0:
            mm_u_limit[1]=min(mm_u_limit[1], mmrssa.RHS/mmrssa.grad_phi_mul_g)
        else:
            mm_u_limit[0]=max(mm_u_limit[0], mmrssa.RHS/mmrssa.grad_phi_mul_g)

        if gaussian_rssa.grad_phi_mul_g>0:
            gaussian_u_limit[1]=min(gaussian_u_limit[1], gaussian_rssa.RHS/gaussian_rssa.grad_phi_mul_g)
        else:
            gaussian_u_limit[0]=max(gaussian_u_limit[0], gaussian_rssa.RHS/gaussian_rssa.grad_phi_mul_g)

        # Draw intervals as vertical lines
        ax_u.vlines(1, mm_u_limit[0], mm_u_limit[1], colors='green', linewidth=50)  
        ax_u.vlines(2, gaussian_u_limit[0], gaussian_u_limit[1], colors='cornflowerblue', linewidth=50)
        ax_u.set_xticks([1, 2], ['$U_r$ of \n Multimodal RSSA', '$U_r$ of \n Gaussian RSSA'])
        ax_u.set_ylabel('$\mathrm{u(Nm)}$')
        # Set limits
        ax_u.axhline(y=-40, color='black', label='u=-40', linestyle='--')
        ax_u.axhline(y=40, color='black', label='u=40', linestyle='--')
        ax_u.set_xlim([0.5, 2.5])
        ax_u.set_ylim([-50, 50])

        fig1.savefig(log_path + '/f_bound.png')
        fig2.savefig(log_path + '/u_bound.png')

    else:
        mmrssa: MMMulRSSA = get_rssa('multiplicative_mmrssa', env, safe_control_kwargs)
        gaussian_rssa: GaussianMulRSSA = get_rssa('gaussian_multiplicative_mmrssa', env, safe_control_kwargs)
        u_mmrssa = mmrssa.safe_control(np.array([-20]))
        u_gaussian = gaussian_rssa.safe_control(np.array([-20]))
        print(u_mmrssa, u_gaussian)
        for modal_param, p in zip(mmrssa.modal_params_pred, mmrssa.optimal_p):
            # f_mu = modal_param['f_mu']
            # f_sigma = modal_param['f_sigma']
            g_mu = modal_param['g_mu']
            g_sigma = modal_param['g_sigma']
            mu = g_mu[[2,3]]
            sigma = g_sigma[[2,3]][:,[2,3]]
            draw_ellipsoid(ax_model, mu, sigma, confidence=p, color='green')
        g_points_flat = gaussian_rssa.g_points_flat
        ax_model.plot(g_points_flat[:, 2], g_points_flat[:, 3], 'o', color='orange', markersize=2, alpha=0.6)
        mu = np.mean(g_points_flat[:, [2, 3]], axis=0).reshape(-1, 1)
        sigma = np.cov(g_points_flat[:, [2, 3]].T)
        draw_ellipsoid(ax_model, mu, sigma, confidence=gaussian_rssa.p_gaussian, color='cornflowerblue')
        ax_model.set_xlabel('$\mathrm{g_3(rad)}$')
        ax_model.set_ylabel('$\mathrm{g_4(rad/s)}$')

        # draw feasible u
        mm_u_limit = [-40, 40]
        mm_u_interval = interval([mm_u_limit[0], mm_u_limit[1]])
        gaussian_u_limit = [-40, 40]
        gaussian_u_interval = interval([gaussian_u_limit[0], gaussian_u_limit[1]])
        ######
        # (L^2-c^2)*u^2 - 2cdu - d^2 <= 0
        ######
        u_mmrssa = 0
        u_gaussian = 0
        for con in mmrssa.safety_conditions:
            # print(abs(con['L']*u_mmrssa)<=con['c']*u_mmrssa+con['d']+0.01)
            L, c, d = con['L'], con['c'], con['d']
            if L+c>0:
                mm_u_interval = mm_u_interval & interval([-d/(L+c), inf])
            else:
                mm_u_interval = mm_u_interval & interval([-inf, -d/(L+c)])
            if L-c>0:
                mm_u_interval = mm_u_interval & interval([-inf, d/(L-c)])
            else:
                mm_u_interval = mm_u_interval & interval([d/(L-c), inf])
        
        con = gaussian_rssa.safety_conditions
        # print(abs(con['L']*u_gaussian)<=con['c']*u_gaussian+con['d']+0.01)
        L, c, d = con['L'], con['c'], con['d']
        if L+c>0:
            gaussian_u_interval = gaussian_u_interval & interval([-d/(L+c), inf])
        else:
            gaussian_u_interval = gaussian_u_interval & interval([-inf, -d/(L+c)])
        if L-c>0:
            gaussian_u_interval = gaussian_u_interval & interval([-inf, d/(L-c)])
        else:
            gaussian_u_interval = gaussian_u_interval & interval([d/(L-c), inf])
        # Draw intervals as vertical lines
        for sub_interval in mm_u_interval:
            u_l, u_r = sub_interval.inf, sub_interval.sup
            ax_u.vlines(1, u_l, u_r, colors='green', linewidth=50)
        for sub_interval in gaussian_u_interval:
            u_l, u_r = sub_interval.inf, sub_interval.sup
            ax_u.vlines(2, u_l, u_r, colors='cornflowerblue', linewidth=50)
        ax_u.set_xticks([1, 2], ['$U_r$ of \n Multimodal RSSA', '$U_r$ of \n Gaussian RSSA'])
        ax_u.set_ylabel('$\mathrm{u(Nm)}$')
        # Set limits
        ax_u.axhline(y=-40, color='black', label='u=-40', linestyle='--')
        ax_u.axhline(y=40, color='black', label='u=40', linestyle='--')
        ax_u.set_xlim([0.5, 2.5])
        ax_u.set_ylim([-50, 50])

        fig1.savefig(log_path + '/g_bound.png')
        fig2.savefig(log_path + '/u_bound.png')

    
    plt.show()
    return

if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 300  # 500

    # additive_rssa_types = ['additive_mmrssa', 'gaussian_additive_mmrssa', 'additive_none']
    # multiplicative_rssa_types=['multiplicative_mmrssa', 'gaussian_multiplicative_mmrssa', 'multiplicative_none']

    # rssa_types = additive_rssa_types
    # # rssa_types = multiplicative_rssa_types
    # pkl_path, log_path = evaluate_in_MM_SegWay(rssa_types=rssa_types, num_steps=1000)
    # with open(pkl_path, 'rb') as file:
    #     data = pickle.load(file)
    # draw_phi(data, rssa_types)
    # draw_u(data, rssa_types)

    ### draw interval
    # draw_interval_in_MM_SegWay(env_type='additive', q=[0, 0.06], dq=[0, 0.5])    
    draw_interval_in_MM_SegWay(env_type='multiplicative', q=[0, 0.2], dq=[0, 0.5])    