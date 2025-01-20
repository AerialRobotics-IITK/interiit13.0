import os
import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from quadrotor import Quadrotor3D
from utils import skew_symmetric, v_dot_q, quaternion_inverse, decompose_quaternion, quaternion_error
from casadi import SX, vertcat
import indi_utils.Butterworth2LowPass as Butterworth2LowPass

STABILIZATION_INDI_FILT_CUTOFF_w = 8.0
STABILIZATION_INDI_FILT_CUTOFF_tau = 8.0
PERIODIC_FREQUENCY = 250.0
sample_time = 1.0/PERIODIC_FREQUENCY

filters = np.zeros((2, 3))  

def calc_alpha(omega_history, thrust):
    quad = Quadrotor3D()
    h = np.cos(np.pi / 4) * quad.length
    k = quad.c
    quantity = np.zeros(3)
    I_v = np.array([[quad.J[0], 0, 0], 
                [0, quad.J[1], 0], 
                [0, 0, quad.J[2]]])

    quantity[0] = -h*thrust[0] + h*thrust[1] + h*thrust[2] - h*thrust[3] - np.cross(quad.a_rate, np.dot(I_v, quad.a_rate))[0]
    quantity[1] = h*thrust[0] - h*thrust[1] + h*thrust[2] - h*thrust[3] - np.cross(quad.a_rate, np.dot(I_v, quad.a_rate))[1]
    quantity[2] = k*thrust[0] + k*thrust[1] - k*thrust[2] - k*thrust[3] - np.cross(quad.a_rate, np.dot(I_v, quad.a_rate))[2]
    return calc_Td(quantity, omega_history, thrust)


def calc_Td(quantity, omega_history, thrust):
    quad = Quadrotor3D()
    h = np.cos(np.pi / 4) * quad.length
    k = quad.c
    tau = np.zeros(3)
    I_v = np.array([[quad.J[0], 0, 0], 
                [0, quad.J[1], 0], 
                [0, 0, quad.J[2]]])
    omega = omega_history[-1]
    dt = 0.01 #######From Controller
    omega_dot = finite_difference(omega, omega_history[-2], dt)

    tau[0] = -h*thrust[0] + h*thrust[1] + h*thrust[2] - h*thrust[3]
    tau[1] = +h*thrust[0] - h*thrust[1] + h*thrust[2] - h*thrust[3]
    tau[2] = k*thrust[0] + k*thrust[1] - k*thrust[2] - k*thrust[3]

    w_dot_f, tau_f = update_filter(omega_dot, tau)

    tau_d = np.zeros(3)
    tau_d[0] = tau_f[0] + quantity[0]-np.dot(I_v, w_dot_f)[0]
    tau_d[1] = tau_f[1] + quantity[1]-np.dot(I_v, w_dot_f)[1]
    tau_d[2] = tau_f[2] + quantity[2]-np.dot(I_v, w_dot_f)[2]

    T = thrust[0]+thrust[1]+thrust[2]+thrust[3]
    return indi_thrust(tau_d, T)


def indi_thrust(tau_d, T):
    quad = Quadrotor3D()
    h = np.cos(np.pi / 4) * quad.length
    k = quad.c
    G = np.array([[1, 1, 0, 1], 
              [-h, h, 0, -h], 
              [h, -h, 0, -h], 
              [k, k, 0, -k]])

    G_inv = np.linalg.pinv(G)
    thrust_matrix = np.zeros(4)
    thrust_matrix[0] = T
    thrust_matrix[1] = tau_d[0]
    thrust_matrix[2] = tau_d[1]
    thrust_matrix[3] = tau_d[2]
    u = np.dot(G_inv, thrust_matrix)

    return u


def finite_difference_from_filter(filter, dt):
    output = []
    for i in range(3):
        output[i] = (filter[i].o[0] - filter[i].o[1])*PERIODIC_FREQUENCY
        # output[i] = (filter[i].o[0] - filter[i].o[1]) / dt
    return output

def finite_difference(new1, old, dt):
    output = []
    for i in range(3):
        output[i] = (new1[i] - old[i])*PERIODIC_FREQUENCY
        # output[i] = (new1[i] - old[i]) / dt;
    return output

def indi_init_filters():
    global filters     # w_df, tau_df

    w_est = 1.0/(2.0*np.pi*STABILIZATION_INDI_FILT_CUTOFF_w)
    tau_est = 1.0/(2.0*np.pi*STABILIZATION_INDI_FILT_CUTOFF_tau)
    tau = [w_est, tau_est]

    for i in range(2):
        filters[i, 0].init(tau[i], sample_time, 0.0)
        filters[i, 1].init(tau[i], sample_time, 0.0)
        filters[i, 2].init(tau[i], sample_time, 0.0)
    return

def update_filter(w, tau):
    w_new = np.zeros(3)
    tau_new = np.zeros(3)
    est_new = [w_new, tau_new]

    for i in range(2):
        w_new[i, 0] = filters[i, 0].init(est_new[i], sample_time, 0.0)
        w_new[i, 1] = filters[i, 1].init(est_new[i], sample_time, 0.0)
        w_new[i, 2] = filters[i, 2].init(est_new[i], sample_time, 0.0)
    
    w_new = filter[0].update(w)    
    tau_new = filter[1].update(tau)
    return w_new, tau_new

def finite_difference_from_filter(filter, dt):
    output = []
    for i in range(3):
        output[i] = (filter[i].o[0] - filter[i].o[1])*PERIODIC_FREQUENCY # output[i] = (filter[i].o[0] - filter[i].o[1]) / dt
    return output

def finite_difference(new1, old, dt):
    output = []
    for i in range(3):
        output[i] = (new1[i] - old[i])*PERIODIC_FREQUENCY # output[i] = (new1[i] - old[i]) / dt;
    return output
