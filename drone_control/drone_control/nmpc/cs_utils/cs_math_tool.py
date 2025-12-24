import casadi as cs

'''
Math tool for casadi format
'''

def quaternion_to_rotm(q):
    '''
    Convert quaternion to rotation.
    :param q: qw, qx, qy, qz
    :return: rotation matrix
    '''
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    rotm = cs.vertcat(
        cs.horzcat(1-2*(qy*qy + qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        cs.horzcat(2*(qy*qx+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)),
        cs.horzcat(2*(qz*qx-qw*qy), 2*(qz*qy+qw*qx), 1-2*(qx*qx+qy*qy))
    )
    return rotm

def otimes(q1, q2):
    '''
    Compute multiplication of two quaternions.
    :param q1: Left quaternion (qw, qx, qy, qz)
    :param q2: Right quaternion (qw, qx, qy, qz)
    :return: multiplication result
    '''
    q1_w = q1[0]
    q1_x = q1[1]
    q1_y = q1[2]
    q1_z = q1[3]

    q1_L = cs.vertcat(
        cs.horzcat(q1_w, -q1_x, -q1_y, -q1_z),
        cs.horzcat(q1_x, q1_w, -q1_z, q1_y),
        cs.horzcat(q1_y, q1_z, q1_w, -q1_x),
        cs.horzcat(q1_z, -q1_y, q1_x, q1_w)
    )

    return cs.mtimes(q1_L, q2)