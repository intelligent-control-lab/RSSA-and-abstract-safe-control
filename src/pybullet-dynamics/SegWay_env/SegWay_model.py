from matplotlib import pyplot as plt
import numpy as np
from numpy import sin, cos

class SegWayModel:
    '''
    SeyWay robot dynamics model:    

    state x:
    - q = [r, a], where r is the position of the wheel, a is the angle of the frame
    - dq = [dr, da] 

    control input u: one dimension
    '''
    def __init__(
        self, 
        dt,
        K_m=2.524, 
        K_b=0.189,
        m_0=52.710, m=44.798, J_0=5.108,
        L=0.169, l=0.75, R=0.195,
        g=9.81,
    ):
        self.dt = dt
        self.K_m = K_m
        self.K_b = K_b
        self.m_0 = m_0
        self.m = m
        self.J_0 = J_0
        self.L = L
        self.l = l
        self.R = R
        self.g = g

        self.q = np.zeros(2)
        self.dq = np.zeros(2)

    @property
    def b_t(self):
        return self.K_m * self.K_b / self.R

    @property
    def B(self):
        K_m = self.K_m
        B = np.zeros(2)
        B[0] = K_m / self.R
        B[1] = -K_m
        return B

    @property
    def M(self):
        M = np.zeros((2, 2))
        a = self.q[1]
        M[0, 0] = self.m_0
        M[1, 1] = self.J_0
        M[0, 1] = M[1, 0] = self.m * self.L * cos(a)
        return M

    @property
    def H(self):
        b_t = self.b_t
        a = self.q[1]
        dr, da = self.dq
        H = np.zeros(2)
        H[0] = -self.m * self.L * sin(a) * da**2 + b_t / self.R * (dr - self.R * da)
        H[1] = -self.m * self.g * self.L * sin(a) - b_t * (dr - self.R * da)
        return H

    @property
    def p(self):
        # position of the frame's tip
        p = self.calculate_p_given_q(self.q)
        return p

    @property
    def J(self):
        J = self.calculate_J_give_q(self.q)
        return J

    @property
    def dp(self):
        return self.J @ self.q

    @property
    def p_dp_p_q(self):
        a = self.q[1]
        da = self.q[1]
        p_dp_p_q = np.zeros((2, 2))
        p_dp_p_q[0, 1] = -self.l * da * cos(a)
        p_dp_p_q[1, 1] = -self.l * da * sin(a)
        return p_dp_p_q

    def calculate_J_give_q(self, q):
        a = q[1]
        J = np.zeros((2, 2))
        J[0, 0] = 1
        J[0, 1] = self.l * cos(a)
        J[1, 1] = -self.l * sin(a)
        return J

    def calculate_p_given_q(self, q):
        r, a = q
        p = np.zeros(2)
        p[0] = r + self.l * sin(a)
        p[1] = self.R + self.l * cos(a)
        return p

    def step(self, u):
        # shape of u: (1,)
        M_inv = np.linalg.pinv(self.M)
        ddq = M_inv @ (self.B * u - self.H)
        self.q += self.dq * self.dt
        self.dq += ddq * self.dt

    def set_joint_states(self, q, dq=[0.0, 0.0]):
        self.q = np.asanyarray(q)
        self.dq = np.asanyarray(dq)

    def PD_control(self, q_d, dq_d):
        K_dr = 8
        K_a = 40
        K_da = 10
        dr_d = dq_d[0]
        a_d = q_d[1]

        a = self.q[1]
        dr, da = self.dq

        u = K_dr * (dr - dr_d) + K_a * (a - a_d) + K_da * da
        return np.array([u])



if __name__ == '__main__':
    robot = SegWayModel(dt=1/240)

    q_d = np.array([0, 0])
    dq_d = np.array([1, 0])
    a_list = []
    dr_list = []
    u_list = []
    for i in range(960):
        u = robot.PD_control(q_d, dq_d)
        robot.step(u)
        print(robot.q, robot.dq)
        u_list.append(u[0])
        a_list.append(robot.q[1])
        dr_list.append(robot.dq[0])
    
    # plt.plot(a_list)
    # plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/model_test/a.jpg')
    # plt.close()
    # plt.plot(dr_list)
    # plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/model_test/dr.jpg')
    # plt.close()
    # plt.plot(u_list)
    # plt.savefig('./src/pybullet-dynamics/SegWay_env/imgs/model_test/u.jpg')
    # plt.close()


    
        