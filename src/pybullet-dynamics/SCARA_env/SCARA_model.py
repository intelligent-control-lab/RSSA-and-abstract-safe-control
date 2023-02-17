import numpy as np
from numpy import cos, sin

class SCARAModel:
    '''
    u = M(q) @ ddq + H(q, dq) - J(q).T @ F_{ext} \\
    A, B, C represents dynamic parameters \\
    p and dp represents end-effector's position and velocity
    '''
    def __init__(
        self,
        dt,
        m_1, l_1,
        m_2, l_2,
    ):  
        self.dt = dt
        self.m_1 = m_1
        self.l_1 = l_1
        self.m_2 = m_2
        self.l_2 = l_2

        self.q = np.zeros(2)
        self.dq = np.zeros(2)

    @property
    def I_1(self):
        return 1 / 12 * self.m_1 * self.l_1**2

    @property
    def I_2(self):
        return 1 / 12 * self.m_2 * self.l_2**2
    
    @property
    def A(self):
        return 1/8 * self.m_1 * self.l_1**2 + 1/2 * self.I_1 + 1/2 * self.m_2 * self.l_1**2
    
    @property
    def B(self):
        return 1/8 * self.m_2 * self.l_2**2 + 1/2 * self.I_2
    
    @property
    def C(self):
        return 1/2 * self.m_2 * self.l_1 * self.l_2

    @property
    def M(self):
        M = np.zeros((2, 2))
        M[0, 0] = 2 * self.A + 2 * self.B + 2 * self.C * np.cos(self.q[1])
        M[0, 1] = M[1, 0] = 2 * self.B + self.C * np.cos(self.q[1])
        M[1, 1] = 2 * self.B
        return M

    @property
    def H(self):
        H = np.zeros(2)
        H[0] = -self.C * np.sin(self.q[1]) * (2 * self.dq[0] + self.dq[1]) * self.dq[1]
        H[1] = self.C * np.sin(self.q[1]) * self.dq[0]**2      
        return H

    @property
    def J(self):
        J = self.calculate_J_give_q(self.q)
        return J

    @property
    def p(self):
        p = self.calculate_p_given_q(self.q)
        return p

    @property
    def dp(self):
        return self.J @ self.dq

    @property
    def p_dp_p_q(self):
        # calculate p_(J(q)dq)_p_q
        q_1, q_2 = self.q
        dq_1, dq_2 = self.dq
        l_1 = self.l_1
        l_2 = self.l_2
        p_dp_p_q = np.zeros((2, 2))
        p_dp_p_q[0, 0] = -dq_1 * (l_1 * cos(q_1) + l_2 * cos(q_1 + q_2)) - dq_2 * l_2 * cos(q_1 + q_2)
        p_dp_p_q[0, 1] = -dq_1 * l_2 * cos(q_1 + q_2) - dq_2 * l_2 * cos(q_1 + q_2)
        p_dp_p_q[1, 0] = -dq_1 * (l_1 * sin(q_1) + l_2 * sin(q_1 + q_2)) - dq_2 * l_2 * sin(q_1 + q_2)
        p_dp_p_q[1, 1] = -dq_1 * l_2 * sin(q_1 + q_2) - dq_2 * l_2 * sin(q_1 + q_2)
        return p_dp_p_q

    def calculate_J_give_q(self, q):
        J = np.zeros((2, 2))
        J[0, 0] = -self.l_1 * np.sin(q[0]) - self.l_2 * np.sin(q[0] + q[1])
        J[0, 1] = -self.l_2 * np.sin(q[0] + q[1])
        J[1, 0] = self.l_1 * np.cos(q[0]) + self.l_2 * np.cos(q[0] + q[1])
        J[1, 1] = self.l_2 * np.cos(q[0] + q[1])
        return J

    def calculate_p_given_q(self, q):
        p = np.zeros(2)
        p[0] = self.l_1 * np.cos(q[0]) + self.l_2 * np.cos(q[0] + q[1])
        p[1] = self.l_1 * np.sin(q[0]) + self.l_2 * np.sin(q[0] + q[1])
        return p

    def step(self, u, F_ext):
        M_inv = np.linalg.pinv(self.M)
        ddq = M_inv @ (u - self.H + self.J.T @ F_ext)
        self.q += self.dq * self.dt
        self.dq += ddq * self.dt

    def set_joint_states(self, q, dq=[0.0, 0.0]):
        self.q = np.asanyarray(q)
        self.dq = np.asanyarray(dq)

    def PD_control(self, q_d, dq_d):
        Kp = 64.0
        Kd = 16.0
        u = Kp * (q_d - self.q) + Kd * (dq_d - self.dq)
        return u

    def computed_torque_control(self, q_d, dq_d=np.zeros(2), F_ext=np.zeros(2)):
        Kp = 10.0
        Kd = 5.0
        fake_ddq = Kp * (q_d - self.q) + Kd * (dq_d - self.dq)
        u = self.M @ fake_ddq + self.H - self.J.T @ F_ext
        return u

    def solve_inverse_kinematics(
        self, p_d, init_q, 
        lr=1e-1, max_steps=500, eps=2e-3
    ):
        q = init_q
        p_calc = self.calculate_p_given_q(q)
        count = 0
        while count < max_steps and np.linalg.norm(p_d - p_calc) > eps:
            count += 1
            p_calc = self.calculate_p_given_q(q)
            J = self.calculate_J_give_q(q) 
            grad = 2 * (p_calc - p_d) @ J
            q = q - lr * grad
        if count == max_steps:
            return None
        else:
            return q


if __name__ == '__main__':
    robot = SCARAModel(
        dt=1/240,
        m_1=1.0, l_1=1.0, I_1=1/12,
        m_2=1.0, l_2=1.0, I_2=1/12,
    )

    q_d = np.array([0.5, 0.5])
    dq_d = np.zeros_like(q_d)
    F_ext = np.array([0.0, 0.0])

    robot.solve_inverse_kinematics(p_d=np.array([0.6, 1.2]), init_q=np.ones(2) * 0.5)

    for i in range(100):
        # u = robot.PD_control(q_d, dq_d)
        u = robot.computed_torque_control(q_d, dq_d, F_ext)
        # print(u)
        robot.step(u, F_ext)