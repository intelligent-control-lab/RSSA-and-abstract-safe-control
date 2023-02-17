import os
import numpy as np
import gym
import pandas as pd
import progressbar
import gym_dynamics
class GymDynamicsDataset(object):
    def __init__(self, env_name, size=100000):
        self.env_name = env_name
        self.size = size
        self.data_path = '../data/'+self.env_name+'_'+str(self.size)+'.zip'
        self.load_dataset()

    def load_dataset(self, generate=False):
        print("Loading ", self.data_path)
        if not os.path.exists(self.data_path) or generate:
            self.generate_dataset()
        self.df = pd.read_pickle(self.data_path)
        
    def generate_dataset(self):
        env = gym.make(self.env_name)

        state, reward, done, info = env.reset()
        step = 0

        df = pd.DataFrame(columns=['state', 'action', 'dot_state', 'reward', 'done'])
        print("Generating dynamics data")
        for step in progressbar.progressbar(range(self.size)):
        # for step in range(self.size):
            action = env.sample_action()
            new_state, reward, done, info = env.step(action)
            df.loc[step] = [state.astype(np.float32), action.astype(np.float32), info["dot_state"].astype(np.float32), reward, done]
            
            state = new_state
            if done:
                state, reward, done, info = env.reset()
        
        df.to_pickle(self.data_path, compression='zip')

    def __getitem__(self, idx):
        return np.squeeze(np.vstack([self.df.iat[idx, 0], self.df.iat[idx, 1]])), np.squeeze(self.df.iat[idx, 2])

    def __len__(self):
        return len(self.df)

class GymAffineDynamicsDataset(GymDynamicsDataset):
    
    def __getitem__(self, idx):
        # print(self.df.iat[idx, 0])
        # print(self.df.iat[idx, 1])
        # print(self.df.iat[idx, 2])
        # print(np.vstack([self.df.iat[idx, 0], self.df.iat[idx, 1]]).T)
        # exit()
        return np.squeeze(self.df.iat[idx, 0]).astype(np.float32), {"u":np.squeeze(self.df.iat[idx, 1]).astype(np.float32), "dot_x":np.squeeze(self.df.iat[idx, 2]).astype(np.float32)}


class UncertainUnicycleDataset(GymAffineDynamicsDataset):
    def __init__(self, env_name, size=100000, region_idx = [1,2,3]):
        self.env_name = env_name
        self.size = size
        self.data_path = '../data/'+self.env_name+'-'+str(size)+'-'+str(region_idx)+'.zip'
        self.region_idx = region_idx
        self.load_dataset(generate=True)

    def generate_dataset(self):
        env = gym.make(self.env_name)
        regions = [
            {'min_x':0, 'max_x':10, 'min_y':0, 'max_y':10},  # leave out [0, 10, 0, 10] for training
            {'min_x':-10, 'max_x':0, 'min_y':0, 'max_y':10},
            {'min_x':-10, 'max_x':0, 'min_y':-10, 'max_y':0},
            {'min_x':0, 'max_x':10, 'min_y':-10, 'max_y':0}
        ]    

        state, reward, done, info = env.reset(obs_num=0, uncertainty=5, **regions[self.region_idx[0]])
        step = 0

        df = pd.DataFrame(columns=['state', 'action', 'dot_state', 'reward', 'done'])
        print("Generating dynamics data")
        reset_cnt = 0
        for step in progressbar.progressbar(range(self.size)):
        # for step in range(self.size):                
            action = env.sample_action()
            new_state, reward, done, info = env.step(action)
            df.loc[step] = [state.astype(np.float32), action.astype(np.float32), info["dot_state"].astype(np.float32), reward, done]
            
            state = new_state

            # print(state)
            if state[0] < env.min_x or state[0] > env.max_x or state[1] < env.min_y or state[1] > env.max_y:
                done = True

            if done:
                reset_cnt += 1
                state, reward, done, info = env.reset(obs_num=0, uncertainty=5, **regions[self.region_idx[reset_cnt % len(self.region_idx)]])
                
            # env.render()
        df.to_pickle(self.data_path, compression='zip')



if __name__ == '__main__':
    
    # DynamicsDataset("Ant-v2")
    # UnicycleDynamicsDataset()
    # GymDynamicsDataset("Free-Ant-v0")
    UncertainUnicycleDataset("Uncertain-Unicycle-v0")