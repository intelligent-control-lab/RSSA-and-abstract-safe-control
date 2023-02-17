import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
from .uncertain_unicycle_env import UncertainUnicycleEnv
class UncertainUnicycleHitEnv(UncertainUnicycleEnv):
    def __init__(self):
        self.max_vr = 2
        self.max_ar = 4
        self.max_vt = np.pi
        self.dt = 0.1
        self.max_u =  np.vstack([self.max_ar, self.max_vt])
        self.max_x = 10
        self.max_y = 10
        self.render_initialized=False
        self.initialize()

    def initialize(self, state=[-5,-5,1,np.pi/4], goal = [5,5], obs=[0,0,0,0], uncertainty=1, obs_radius=1, min_x=-10, max_x=10, min_y=-10, max_y=10):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.step_cnt = 0
        
        np.random.seed(0)
        self.init_state = np.vstack(state)
        self.state = np.vstack(state)
        self.goal = np.vstack(goal)
        self.obs = np.vstack(obs)
        self.obs_max_v = 0
        self.obs_radius = obs_radius
        self.col_cnt = 0
        self.col_flag = np.zeros(1)

        self.wall_bounded = False
        
        self.g_uncertainty=uncertainty
        self.f_uncertainty=uncertainty * 0.00


    def init_obstacle(self):
        self.obstacles = [np.vstack(self.obs).astype(np.float32)]
        self.wall_obs = [
            np.vstack([self.state[0], self.max_y-1, self.state[2], 0]),
            np.vstack([self.state[0],-self.max_y+1, self.state[2], 0]),
            np.vstack([ self.max_x-1, self.state[1], 0, self.state[3]]),
            np.vstack([-self.max_x+1, self.state[1], 0, self.state[3]]),
        ]
        
    def update_obstacle(self):
        for i in range(len(self.obstacles)):
            self.obstacles[i][[0,1]] += self.obstacles[i][[2,3]] * self.dt
            if self.obstacles[i][0] > self.max_x or self.obstacles[i][0] < -self.max_x or self.obstacles[i][1] < -self.max_y or self.obstacles[i][1] > self.max_y:
                self.obstacles[i] = self.generate_obs()
                self.col_flag[i] = False
            if np.linalg.norm(self.obstacles[i][[0,1]] - self.state[[0,1]]) < self.obs_radius:
                if not self.col_flag[i]:
                    self.col_cnt += 1
                self.col_flag[i] = True
        self.wall_obs = [
            np.vstack([self.state[0], self.max_y, self.state[2], 0]),
            np.vstack([self.state[0],-self.max_y, self.state[2], 0]),
            np.vstack([ self.max_x, self.state[1], 0, self.state[3]]),
            np.vstack([-self.max_x, self.state[1], 0, self.state[3]]),
        ]
    
    def step(self, u):
        # print("fuck")
        # print("self.state")
        # print(self.state)
        u = self.filt_action(u)

        # print("self.state")
        # print(self.state)
        
        dot_state = self.rk4(self.state, u, self.dt)
        self.state = self.state + dot_state * self.dt

        self.state = self.filt_state(self.state)

        self.step_cnt += 1
        done = False
        if self.step_cnt > 1000:
            done = True
        
        if np.linalg.norm(self.state[:2] - self.goal) < 5e-1:
            done = True
        
        self.update_obstacle()

        info={
            "goal": self.goal,
            "obs_state": self.obstacles,
        }
        
        return self.state, 0, done, info
        
    def reset(self, state=[-5,-5,1,np.pi/4], goal = [5,5], obs=[0,0,0,0], uncertainty=1, obs_radius=1):
        self.initialize(state, goal, obs, uncertainty=uncertainty, obs_radius=obs_radius)
        self.init_obstacle()
        
        info={
            "goal": self.goal,
            "obs_state": self.obstacles,
        }
        
        return self.state, 0, False, info

    def render(self, mode, phis=None, save=False):
        if not self.render_initialized:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            plt.xlim([-self.max_x*1.5,self.max_x*1.5])
            plt.ylim([-self.max_y*1.5,self.max_y*1.5])
            x,y = self.get_unicycle_plot()
            self.unicycle_line, = self.ax.plot(x, y)
            self.unicycle_goal_dot = self.ax.scatter([self.goal[0]], [self.goal[1]], s=30)
            self.obs_state_dots = self.ax.scatter([obs[0] for obs in self.obstacles], [obs[1] for obs in self.obstacles], s=314 * self.obs_radius**2)
            # self.obs_goal_dot = self.ax.scatter([self.obs_goal[0]], [self.obs_goal[1]], s=10)
            self.render_initialized = True
        x,y = self.get_unicycle_plot()
        self.unicycle_line.set_xdata(x)
        self.unicycle_line.set_ydata(y)
        self.unicycle_goal_dot.set_offsets(np.column_stack([self.goal[0], self.goal[1]]))
        self.obs_state_dots.set_offsets(np.column_stack([[obs[0] for obs in self.obstacles], [obs[1] for obs in self.obstacles]]))
        if phis:
            c = ['orange' if phi < 0 else 'red' for phi in phis]
            self.obs_state_dots.set_color(c)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if save:
            plt.savefig("../results/img"+f"{self.step_cnt:05d}"+".png")
        plt.pause(0.0001)
