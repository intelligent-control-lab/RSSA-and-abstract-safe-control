from dataclasses import replace
from typing import Dict, List, Tuple
import numpy as np
import pickle
import cv2
import heapq
import time

from panda_latent_env import PandaLatentEnv


def collect_M_and_dot_M_max(
    env: PandaLatentEnv, 
    num=int(1e5),
    path='src/pybullet-dynamics/panda_rod_env/data/M_and_dot_M_max/test/',
):
    def create_data(env: PandaLatentEnv) -> Dict:
        q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
        dq = np.random.uniform(low=env.dq_limit['low'], high=env.dq_limit['high'])
        env.robot.set_joint_states(q, dq)
        env.update_latent_info()
        M = env.M
        dot_M_max = env.get_dot_M_max_per_state()
        data = {
            'q': q,
            'M': M,
            'dot_M_max': dot_M_max,
            'J': env.robot.get_jacobian(),
        }
        return data
    
    for i in range(num):
        print(i)
        data = create_data(env)
        with open(path + str(i) + '.pkl', 'wb') as file:
            pickle.dump(data, file)
            

def collect_M(env: PandaLatentEnv):
    warmup_num = int(1e4)
    total_num = int(1e7)
    record_interval = warmup_num
    
    def create_data(env: PandaLatentEnv) -> Dict:
        q = np.random.uniform(low=env.q_limit['low'], high=env.q_limit['high'])
        env.robot.set_joint_states(q, np.zeros_like(q) + 1e-4)
        env.update_latent_info()
        data = {
            'q': q,
            'a': env.a,
            'C': env.C,
            'M': env.M,
            'J': env.robot.get_jacobian(),
        }
        return data
    
    heap_list = []
    for i in range(warmup_num):
        data = create_data(env)
        heap_data = (-data['M'], data)
        heap_list.append(heap_data)
    heapq.heapify(heap_list)
    print('Warm up finish!')
    
    for i in range(total_num // record_interval):
        print(f'{i} record interval')
        for j in range(record_interval):
            data = create_data(env)
            heap_data = (-data['M'], data)
            heapq.heappushpop(heap_list, heap_data)
       
        with open('src/pybullet-dynamics/panda_rod_env/data/abstract_data/large_test.pkl', 'wb') as file:
            pickle.dump(heap_list, file)
        

def analyze_M(env: PandaLatentEnv):
    with open('src/pybullet-dynamics/panda_rod_env/data/abstract_data/large_test.pkl', 'rb') as file:
        data_list = pickle.load(file)
        
    def slice_data(data_list: List[Tuple], name: str):
        return [data[1][name] for data in data_list]
    
    M = slice_data(data_list, 'M')
    J = slice_data(data_list, 'J')
    q = slice_data(data_list, 'q')
    a = slice_data(data_list, 'a')
    C = slice_data(data_list, 'C')
    
    M = np.asanyarray(M)
    M_indices = np.argsort(M)
    for i in range(10):
        M_min = M[M_indices[i]]
        # print(M_min)
        q_min = q[M_indices[i]]
        dq = env.dq_limit['high']
        env.robot.set_joint_states(q_min, dq)
        print(env.robot.get_end_eff_velocity())
        env.update_latent_info()
        print(env.M)
        # rgb = env.render_image()
        # cv2.imwrite(f'src/pybullet-dynamics/panda_rod_env/imgs/abstract/{i}.png', rgb)
        
        # C_min = C[M_indices[i]]
        # a_min = a[M_indices[i]]
        # pinv_C = np.linalg.pinv([C_min])
        # Q_v = (pinv_C.T @ env.Q_u @ pinv_C).item()
        # res = a_min * Q_v * a_min
        # print(res)
      
        
def dump_data(
    pkl_name,
    num=1e5, 
    root_path='src/pybullet-dynamics/panda_rod_env/data/M_and_dot_M_max/cpu/',
):
    num = int(num)
    totol_num = int(2.5e5)
    indices = np.random.choice(totol_num, size=num, replace=False)
    data = []
    for i in indices:
        with open(root_path + str(i) + '.pkl', 'rb') as file:
            tmp = pickle.load(file)
        data.append(tmp)
    
    final_path = 'src/pybullet-dynamics/panda_rod_env/data/final_data/'
    # with open(final_path + 'M_and_dot_M_max_distribution.pkl', 'wb') as file:
    #     pickle.dump(data, file)
    with open(final_path + pkl_name, 'wb') as file:
        pickle.dump(data, file)
        
    
    
        
        
if __name__ == '__main__':
    env = PandaLatentEnv(
        render_flag=False, 
        goal_pose=[0.7, 0.3, 0.4], 
        obstacle_pose=[0.45, 0.1, 0.55], 
        if_task=True,
        robot_file_path='src/pybullet-dynamics/panda_rod_env/urdf/panda_without_rod.urdf',
        dof=7
    )
    
    # q = np.array([ 2.2122625 , -1.23601788, -1.44122092, -2.2912187 ,  2.45808416,
    #      0.41332959,  2.43699197])
    # env.robot.set_joint_states(q, np.zeros_like(q))
    # rgb = env.render(
    #         height=512, width=512,
    #         cam_dist=1.2,
    #         camera_target_position=[0, -0.2, 0.2],
    #         cam_yaw=-45, cam_pitch=-40, cam_roll=0, 
    #     )
    # img_path = 'src/pybullet-dynamics/panda_rod_env/imgs/final/'
    # cv2.imwrite(img_path + 'dot_M_max_pose.jpg', rgb)
    # exit(0)
    
    # with open('src/pybullet-dynamics/panda_rod_env/data/final_data/M_and_dot_M_max_distribution.pkl', 'rb') as file:
    #     data = pickle.load(file)
    
    start = time.time()
    
    root_path = 'src/pybullet-dynamics/panda_rod_env/data/M_and_dot_M_max/'
    collect_M_and_dot_M_max(env, num=int(1e5), path=root_path + 'cpu/')
    
    end = time.time()
    print(f'total time: {end - start} seconds')
    
    # collect_M(env)
    # analyze_M(env)
    # dump_data(pkl_name='fix_6_7.pkl', num=1e5)
    