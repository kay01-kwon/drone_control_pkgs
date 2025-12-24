import numpy as np

class ControlAllocator:
    def __init__(self, Param):
        l = Param['arm_length']
        self.C_T = Param['rotor_const']
        k_m = Param['moment_const']
        self.T_max = Param['T_max']
        self.T_min = Param['T_min']
        self.acc_max = Param['acc_max']
        self.acc_min = Param['acc_min']

        cos_pi_3 = np.cos(np.pi/3)
        sin_pi_3 = np.sin(np.pi/3)

        ly1 = l*cos_pi_3
        ly2 = l
        ly3 = l*cos_pi_3

        ly4 = -l*cos_pi_3
        ly5 = -l
        ly6 = -l*cos_pi_3

        lx1 = l*sin_pi_3
        lx2 = 0
        lx3 = -l*sin_pi_3

        lx4 = -l*sin_pi_3
        lx5 = 0
        lx6 = l*sin_pi_3

        K_forward = np.array([
            [1, 1, 1, 1, 1, 1],
            [ly1, ly2, ly3, ly4, ly5, ly6],
            [-lx1, -lx2, -lx3, -lx4, -lx5, -lx6],
            [-k_m, k_m, -k_m, k_m, -k_m, k_m]
        ])

        self.K_inv = np.linalg.pinv(K_forward)

    def compute_des_rpm(self, f, M):
        u = np.hstack([f,M])
        rotors_thrust = self.K_inv @ u
        rotors_thrust = self.clamp(rotors_thrust)
        rotors_speed = np.sqrt(rotors_thrust/self.C_T)
        return rotors_speed

    # Consider the limitation of actuator
    def compute_relaxed_des_rpm(self, f, M, rotor_speed_prev, dt):
        u = np.hstack([f,M])
        rotors_thrust = self.K_inv @ u
        rotors_thrust = self.clamp(rotors_thrust)
        rotors_speed = np.sqrt(rotors_thrust/self.C_T)
        
        # Acceleration saturation
        for i in range(len(rotors_speed)):
            if rotors_speed[i] - rotor_speed_prev[i] > dt*self.acc_max:
                rotors_speed[i] = rotor_speed_prev[i] + dt*self.acc_max
            elif rotors_speed[i] - rotor_speed_prev[i] < -dt*self.acc_min:
                rotors_speed[i] = rotor_speed_prev[i] - dt*self.acc_min

        return rotors_speed

    def clamp(self, rotors_thrust):
        for i in range(len(rotors_thrust)):
            if self.T_max < rotors_thrust[i]:
                rotors_thrust[i] = self.T_max
            if self.T_min > rotors_thrust[i]:
                rotors_thrust[i] = self.T_min
        return rotors_thrust