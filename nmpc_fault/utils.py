import numpy as np
import casadi as cs
from pyquaternion import Quaternion

def quaternion_to_euler(q):
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).
    :param q: Quaternion in format (w, x, y, z)
    :return: List of Euler angles [roll, pitch, yaw]
    """
    q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def unit_quat(q):
    """
    Normalizes a quaternion to unit length.
    :param q: Quaternion in array or CasADi format
    :return: Normalized quaternion
    """
    q_norm = np.linalg.norm(q) if isinstance(q, np.ndarray) else cs.norm_2(q)
    return q / q_norm

def v_dot_q(v, q):
    """
    Rotates a vector by a quaternion.
    :param v: 3D vector
    :param q: Quaternion
    :return: Rotated vector
    """
    rot_mat = q_to_rot_mat(q)
    return rot_mat.dot(v) if isinstance(q, np.ndarray) else cs.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    """
    Converts a quaternion to a rotation matrix.
    :param q: Quaternion
    :return: 3x3 rotation matrix
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2))
        )
    return rot_mat
    

def quaternion_inverse(q):
    """ Returns the inverse of a quaternion. """
    if isinstance(q, np.ndarray):
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        q_norm_sq = np.sum(q ** 2)
    else:
        q_conj = cs.vertcat(q[0], -q[1], -q[2], -q[3])
        q_norm_sq = cs.sumsqr(q)
    return q_conj/q_norm_sq


def q_dot_q(q1, q2):
    """
    Performs quaternion multiplication between two quaternions.
    :param q1: First quaternion
    :param q2: Second quaternion
    :return: Resulting quaternion from multiplication
    """
    if isinstance(q1, np.ndarray) and isinstance(q2, np.ndarray):
        w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        return np.array([w, x, y, z])
    else:
        w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        return cs.vertcat(w, x, y, z)

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector.
    :param v: 3D vector
    :return: 4x4 skew-symmetric matrix
    """
    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])
    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0)
    )

def decompose_quaternion(q):
    """
    Decomposes a quaternion into z and xy rotations.
    :param q: Quaternion
    :return: Tuple of xy and z rotation quaternions
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    qz = unit_quat(cs.vertcat(w, 0, 0, z)) if isinstance(q, cs.MX) else unit_quat(np.array([w, 0, 0, z]))
    qxy = q_dot_q(q, quaternion_inverse(qz))
    return qxy, qz

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.
    :param q1: First quaternion
    :param q2: Second quaternion
    :return: Resulting quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])



def quaternion_error(q, q_ref):
    """
    Calculates the quaternion error between two quaternions.
    :param q: Quaternion
    :param q_ref: Reference quaternion
    :return: Quaternion error
    """
    q_error = q_dot_q(q_ref, quaternion_inverse(q))
    return q_error
