import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from julia import LazySets
class UncertainUnicycleEnv(gym.Env):
    def __init__(self):
        self.max_vr = 2
        self.max_ar = 4
        self.max_vt = np.pi
        self.dt = 0.1
        self.max_u =  np.vstack([self.max_ar, self.max_vt])
        self.render_initialized=False
        self.initialize()
        
    def initialize(self,obs_num=5, uncertainty=1, min_x=-10, max_x=10, min_y=-10, max_y=10):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.step_cnt = 0
        self.max_step = 100
        
        # np.random.seed(0)
        self.goal = np.random.rand(2) * [self.max_x, self.max_y] * 2 - [self.max_x, self.max_y]

        self.obs_num = obs_num        
        self.obs_max_v = 0
        self.obs_radius = 1
        self.col_cnt = 0
        self.col_flag = np.zeros(obs_num)

        self.wall_bounded = False
        
        self.g_uncertainty=uncertainty
        self.f_uncertainty=uncertainty * 0.00

    def rk4(self, s, u, dt):
        dot_s1 = self.dynamics(s, u, dt)
        dot_s2 = self.dynamics(s + 0.5*dt*dot_s1, u, dt)
        dot_s3 = self.dynamics(s + 0.5*dt*dot_s2, u, dt)
        dot_s4 = self.dynamics(s + dt*dot_s3, u, dt)
        dot_s = (dot_s1 + 2*dot_s2 + 2*dot_s3 + dot_s4)/6.0
        return dot_s
    
    def f(self, s, perturb=False):
        v = s[2,0]
        theta = s[3,0]
        # print("f state")
        # print(s)
        # print(v)
        # print(theta)
        dot_x = v * np.cos(theta) + np.random.rand() * self.f_uncertainty - self.f_uncertainty/2
        dot_y = v * np.sin(theta) + np.random.rand() * self.f_uncertainty - self.f_uncertainty/2
        # print(np.vstack([dot_x, dot_y, 0, 0]))
        return np.vstack([dot_x, dot_y, 0, 0])
    
    def g(self, s, perturb=True):
        # M = 1 / (1 + np.random.rand() * (self.g_uncertainty)) if perturb else 1.0
        # T = 1/ (1 + np.random.rand() * (self.g_uncertainty)) if perturb else 1.0

        M = 1 / (1 + (s[0,0] - self.min_x)/(self.max_x - self.min_x) * (self.g_uncertainty)) if perturb else 1.0
        T = 1 / (1 + (s[1,0] - self.min_y)/(self.max_y - self.min_y) * (self.g_uncertainty)) if perturb else 1.0
        
        # print("M, T")
        # print(M, T)
        return np.vstack([[0,0,M,0], [0,0,0,T]]).T
    
    def get_1_bits(self,n):
        ret = []
        i = 0
        while n:
            if n & 1:
                ret.append(i)
            i += 1
            n >>= 1
        return ret

    def hyperrect_2_vrep_hrep(self, lb, ub):
        n = len(lb)
        vrep = [np.zeros(n) for i in range(2**n)]
        # print("lb")
        # print(lb)
        # print("ub")
        # print(ub)
        for i in range(2**n):
            vrep[i][:] = lb[:]
            index = self.get_1_bits(i)
            # print("i")
            # print(i)
            # print("index")
            # print(index)
            vrep[i][index] = ub[index]
        
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.hstack([ub, -lb]).T

        return vrep, A, b

    def f_hull(self, s):
        v = s[2,0]
        theta = s[3,0]
        lb = np.array([v * np.cos(theta) - self.f_uncertainty/2, v * np.sin(theta) - self.f_uncertainty/2, 0., 0.])
        ub = np.array([v * np.cos(theta) + self.f_uncertainty/2, v * np.sin(theta) + self.f_uncertainty/2, 0., 0.]) + 1e-8
        
        vrep, A, b = self.hyperrect_2_vrep_hrep(lb, ub)
        return vrep, A, b
        # return LazySets.Hyperrectangle(low=lb, high=ub)
        # return [np.vstack([dot_x, dot_y,0,0]) for dot_x in dot_x_lim for dot_y in dot_y_lim]

    def g_hull(self, s):
        lb = np.hstack([[0,0,1/(1+self.g_uncertainty),0], [0,0,0,1/(1+self.g_uncertainty)]]).astype(np.float)
        # print(lb)
        ub = np.hstack([[0,0,1,0], [0,0,0,1]]).astype(np.float) + 1e-8
        vrep, A, b = self.hyperrect_2_vrep_hrep(lb, ub)
        # print(lb)
        # print(ub)
        # print("g x vrep")
        # print(vrep)
        return vrep, A, b
        # return LazySets.Hyperrectangle(low=lb, high=ub)
        # return [np.vstack([[0,0,M,0], [0,0,0,T]]).T for M in [1/(1+self.g_uncertainty), 1] for T in [1/(1+self.g_uncertainty), 1]]

    def dynamics(self, s, u, dt):
        dot_s = self.f(s) + self.g(s) @ u
        return np.vstack(dot_s)

    def compute_u_ref(self):
        s = self.state
        dp = self.goal - s[:2]
        dis = np.linalg.norm(dp)
        
        theta_R = s[3]
        v = np.array([s[2] * np.cos(theta_R), s[2] * np.sin(theta_R)])
        
        if(np.linalg.norm(v) < 1e-5):
            v[0] = 1e-5 * np.cos(theta_R)
            v[1] = 1e-5 * np.sin(theta_R)
        
        
        d_theta = np.arctan2(dp[1], dp[0]) - theta_R
        
        while d_theta > np.pi:
            d_theta = d_theta - np.pi*2
        while d_theta < -np.pi:
            d_theta = d_theta + np.pi*2

        u0 = np.zeros((2,1))
        
        sgn = 1
        # print("dp")
        # print(dp)
        # print("v")
        # print(v)
        if (dp.T @ v).item() < 0:
            sgn = -1
        
        k_v = 5
        k_theta = 2
        # print("sgn")
        # print(sgn)
        # print("dp[0] * np.cos(theta_R) + dp[1] * np.sin(theta_R)")
        # print(dp[0] * np.cos(theta_R) + dp[1] * np.sin(theta_R))
        # print("np.linalg.norm(v)")
        # print(np.linalg.norm(v))
        u0[0,0] = (dp[0] * np.cos(theta_R) + dp[1] * np.sin(theta_R)) - k_v * np.linalg.norm(v) * sgn
        # u0[0,0] = dp[0] * np.cos(theta_R) + dp[1] * np.sin(theta_R) - k_v * np.linalg.norm(v) * sgn
        u0[1,0] = k_theta * d_theta
        
        return u0
        
    def generate_obs(self):
        
        xylim = np.array([self.max_x, self.max_y]) * 2
        lb = np.array([self.max_x, self.max_y])
        # print(np.random.rand(2) * xylim)
        obs = np.hstack([np.random.rand(2) * xylim - lb, np.random.rand(2)*self.obs_max_v])

        return np.vstack(obs)
        k = np.random.randint(4)

        if k == 0:
            obs[0] = self.max_x
            obs[2] = min(obs[2], -obs[2])
        elif k == 1:
            obs[0] = self.min_x
            obs[2] = max(obs[2], -obs[2])
        elif k == 2:
            obs[1] = self.max_y
            obs[3] = min(obs[3], -obs[3])
        elif k == 3:
            obs[1] = self.min_y
            obs[3] = max(obs[3], -obs[3])

        return np.vstack(obs)

    def init_obstacle(self):
        self.obstacles = [self.generate_obs() for i in range(self.obs_num)]
        self.wall_obs = [
            np.vstack([self.state[0], self.max_y-1, self.state[2], 0]),
            np.vstack([self.state[0],self.min_y+1, self.state[2], 0]),
            np.vstack([ self.max_x-1, self.state[1], 0, self.state[3]]),
            np.vstack([self.min_x+1, self.state[1], 0, self.state[3]]),
        ]
        
    def update_obstacle(self):
        for i in range(self.obs_num):
            self.obstacles[i][[0,1]] += self.obstacles[i][[2,3]] * self.dt
            if self.obstacles[i][0] > self.max_x or self.obstacles[i][0] < self.min_x or self.obstacles[i][1] < self.min_y or self.obstacles[i][1] > self.max_y:
                self.obstacles[i] = self.generate_obs()
                self.col_flag[i] = False
            if np.linalg.norm(self.obstacles[i][[0,1]] - self.state[[0,1]]) < self.obs_radius:
                if not self.col_flag[i]:
                    self.col_cnt += 1
                self.col_flag[i] = True
        self.wall_obs = [
            np.vstack([self.state[0], self.max_y, self.state[2], 0]),
            np.vstack([self.state[0],self.min_y, self.state[2], 0]),
            np.vstack([ self.max_x, self.state[1], 0, self.state[3]]),
            np.vstack([self.min_x, self.state[1], 0, self.state[3]]),
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
        if self.step_cnt > self.max_step:
            done = True
        
        if np.linalg.norm(self.state[:2] - self.goal) < 5e-1:
            self.goal = np.vstack(np.random.rand(2) * [self.max_x, self.max_y] * 1.6 - np.array([self.max_x, self.max_y]) * 0.8)
        
        for obs in self.obstacles:
            if np.linalg.norm(obs[:2] - self.goal) < 1.5:
                self.goal = np.vstack(np.random.rand(2) * [self.max_x, self.max_y] * 1.6 - np.array([self.max_x, self.max_y]) * 0.8)

        self.update_obstacle()

        info={
            "goal": self.goal,
            "obs_state": self.obstacles,
            "dot_state": dot_state,
        }
        
        return self.state, 0, done, info
        
    def reset(self, obs_num=0, uncertainty=5, min_x=-10, max_x=10, min_y=-10, max_y=10):
        self.initialize(obs_num, uncertainty, min_x, max_x, min_y, max_y)
        
        min_state = np.vstack([self.min_x, self.min_y, -self.max_vr, -np.pi])
        max_state = np.vstack([self.max_x, self.max_y, self.max_vr, np.pi])
        self.state = np.vstack(np.random.uniform(min_state, max_state))
        self.goal = np.vstack(np.random.rand(2) * [self.max_x, self.max_y] * 1.6 - np.array([self.max_x, self.max_y]) * 0.8)
        
        self.init_obstacle()
        
        info={
            "goal": self.goal,
            "obs_state": self.obstacles,
        }
        
        return self.state, 0, False, info

    def sample_action(self):
        action = np.vstack(np.random.uniform(-self.max_u, self.max_u))
        return action

    def filt_action(self, u):
        u = np.minimum(u,  self.max_u)
        u = np.maximum(u, -self.max_u)
        return u

    def filt_state(self, s):
        if self.wall_bounded:
            s[0] = np.clip(s[0], self.min_y, self.max_y)
            s[1] = np.clip(s[1], self.min_x, self.max_x)
        while s[3] > 3*np.pi:
            s[3] = s[3] - 2 * np.pi
        while s[3] < -3*np.pi:
            s[3] = s[3] + 2 * np.pi
        return s

    def get_unicycle_plot(self):
        theta = self.state[3]
        ang = (-self.state[3] + np.pi/2) / np.pi * 180
        s = self.state[:2]
        t = self.state[:2] + np.vstack([np.cos(theta), np.sin(theta)])
        c = s
        s = s - (t-s)
        return np.vstack([s[0], t[0]]), np.vstack([s[1], t[1]])

    def render(self, mode, phis=None):
        if not self.render_initialized:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            # plt.xlim([self.min_x*1.5,self.max_x*1.5])
            # plt.ylim([self.min_y*1.5,self.max_y*1.5])
            plt.xlim([-15,15])
            plt.ylim([-15,15])
            x,y = self.get_unicycle_plot()
            self.unicycle_line, = self.ax.plot(x, y)
            self.unicycle_goal_dot = self.ax.scatter([self.goal[0]], [self.goal[1]], s=30)
            self.obs_state_dots = self.ax.scatter([obs[0] for obs in self.obstacles], [obs[1] for obs in self.obstacles], s=314)
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
        plt.pause(0.0001)
