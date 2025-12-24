import numpy as np

def otimes(q1, q2):
    q1_w = q1[0]
    q1_x = q1[1]
    q1_y = q1[2]
    q1_z = q1[3]

    q1_mat = np.array([
        [q1_w, -q1_x, -q1_y, -q1_z],
        [q1_x, q1_w, -q1_z, q1_y],
        [q1_y, q1_z, q1_w, -q1_x],
        [q1_z, -q1_y, q1_x, q1_w]
    ])

    result = q1_mat@q2
    return result

def conjugate(q):
    q_res = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_res

def quaternion_to_rotm(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    rotm = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qy * qx + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qz * qx - qw * qy), 2 * (qz * qy + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
    ])

    return rotm

def quaternion_to_angle_axis_vec(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    q_vec = np.array([qx, qy, qz])
    q_vec_norm = np.sqrt(q_vec.dot(q_vec))
    theta = 2*np.arctan2(qw, q_vec_norm)
    angle_axis_vec = np.zeros((3,))
    sign = signum(qw)
    if theta > 1e-30:
        angle_axis_vec[0] = sign*theta*q_vec[0]/q_vec_norm
        angle_axis_vec[1] = sign*theta*q_vec[1]/q_vec_norm
        angle_axis_vec[2] = sign*theta*q_vec[2]/q_vec_norm
    else:
        angle_axis_vec[0] = 0
        angle_axis_vec[1] = 0
        angle_axis_vec[2] = 0

    return angle_axis_vec


def skew_symm(v):
    vx = v[0]
    vy = v[1]
    vz = v[2]

    return np.array([
        [0, -vz, vy],
        [vz, 0, -vx],
        [-vy, vx, 0]
    ])

def skew_symm_to_vec(R):
    vx = -R[1,2]
    vy = R[0,2]
    vz = -R[0,1]
    return np.array([vx, vy, vz])

def signum(x):
    return 1 if x >= 0 else -1