import numpy as np
from quadrotor import Quadrotor3D

dt = 1/20
k = 4

def INDI(omega_x, omega_y, omega_z, actuator_vel, ideal_w, u):

    assert actuator_vel is not None, "Received None actuator velocities"

    # print("##############################################################################")
    quad = Quadrotor3D()
    
    omega = [0, 0, 0]
    print(f"omega_x: {omega_x[-k:]}")
    print(np.shape(omega_x[-k:]))
    # print(f"omega_y: {omega_y[-k:]}")
    # print(f"omega_z: {omega_z[-k:]}")

    
    omega[0] = np.average(omega_x[-k:], weights=[1/32, 1/32, 1/16, 14/16])
    omega[1] = np.average(omega_y[-k:], weights=[1/32, 1/32, 1/16, 14/16])
    omega[2] = np.average(omega_z[-k:], weights=[1/32, 1/32, 1/16, 14/16])
    
    omega_prev = [0, 0, 0]
    omega_prev[0] = np.average(omega_x[-k-1:-1], weights=[1/32, 1/32, 1/16, 14/16])
    omega_prev[1] = np.average(omega_y[-k-1:-1], weights=[1/32, 1/32, 1/16, 14/16])
    omega_prev[2] = np.average(omega_z[-k-1:-1], weights=[1/32, 1/32, 1/16, 14/16])
    
    omega = np.array(omega).reshape(3,1)
    omega_prev = np.array(omega_prev).reshape(3,1)

    omega_dot = (omega - omega_prev) / dt #first order forward difference, 
    # omega_dot = (omega - omega_prev_prev) / (2*dt) #second order difference
    

    omega_f = omega #(3,1)
    omega_dot_f = omega_dot #(3,1)

    # print(f"omega_f: {omega_f}")
    # print(f"omega_dot_f: {omega_dot_f}")
    print(f"actuator_vel: {actuator_vel}")

    


    actuator_thrust = (quad.motor_const * np.square(actuator_vel)).reshape(4,1) #thrust from motor velocity
    tau_f = np.array([
        (actuator_thrust.T @ quad.y_f)  ,
        (actuator_thrust.T @ quad.x_f)  ,
        (actuator_thrust.T @ quad.z_l_tau)  
    ]).reshape(3,1)
    

    thrust = (np.array(u) * quad.max_thrust).reshape(4,1) #thrust from control input
    tau = np.array([
        (thrust.T @ quad.y_f)  ,
        (thrust.T @ quad.x_f)  ,
        (thrust.T @ quad.z_l_tau)  
    ]).reshape(3,1)

    print("--- Difference in thrusts ---")
    print(f"actuator_thrust: {actuator_thrust}")
    print(f"thrust: {thrust}")
    print(f"u: {u}")
    print("--- Difference in torques ---")
    print(tau_f) # set the difference between tau (acc. to dynamics) and tau_f (acc. to actuator) as disturbance
    print(tau)
    print("-------------------------------")
    
    r = ideal_w.reshape(3,1)
    print(f"r: {r}")

    #check minus plus
    alpha_d = np.array([
            ((thrust.T @ quad.y_f)     -     (quad.J[1] - quad.J[2]) * r[1] * r[2])   / quad.J[0]  ,
            ((thrust.T @ quad.x_f)     -     (quad.J[2] - quad.J[0]) *  r[2] * r[0])  / quad.J[1]  ,
            ((thrust.T @ quad.z_l_tau) -     (quad.J[0] - quad.J[1]) * r[0] * r[1])   / quad.J[2]                                   
            ]).reshape(3,1)
    
    print(f"alpha_d: {alpha_d}")
    print(f"diff in alpha: {alpha_d - omega_dot_f}")
    

    tau_d = tau_f + quad.J @ (alpha_d - omega_dot_f)
    T_d = np.sum(thrust) #check minus plus

    G = np.array([
        [1, 1, 1, 1],
        quad.y_f,
        quad.x_f,
        quad.z_l_tau
    ]).reshape(4,4)

    print(G)
    # G[:, 2] = 0

    # print(f"thrust shape: {np.shape(thrust.T)}")
    # print(f"Ginv shape: {np.shape(np.linalg.pinv(G))}")

    thrust_indi = np.linalg.pinv(G) @ (np.vstack((T_d, tau_d))).reshape(4,1)

    # print(f"thrust: {thrust}")
    # print(f"thrust_indi: {thrust_indi}")

    u_indi = (thrust_indi/quad.k_1000 + 150) / 1000

    # print(u)
    # print(u_indi)

    # exit(1)
    # print("##############################################################################")

    
    return u_indi
    












    
# def calc_alpha(omega_history, thrust): #calculate torque from u using dynamics
#     quad = Quadrotor3D()
#     thrust*=quad.max_thrust

#     h = quad.length
#     k = quad.c
#     quantity = np.zeros(3)
#     I_v = np.array([[quad.J[0], 0, 0], 
#                 [0, quad.J[1], 0], 
#                 [0, 0, quad.J[2]]])
    
#     omega = omega_history[-1]
    
#     quantity[0] = (-h*thrust[0] + h*thrust[1] + h*thrust[2] - h*thrust[3] + (quad.J[1] - quad.J[2]) * omega[1] * omega[2])
#     quantity[1] = (h*thrust[0] - h*thrust[1] + h*thrust[2] - h*thrust[3] + (quad.J[2] - quad.J[0]) * omega[0] * omega[2])
#     quantity[2] = (k*thrust[0] + k*thrust[1] - k*thrust[2] - k*thrust[3] + (quad.J[0] - quad.J[1]) * omega[0] * omega[1])
#     return calc_Td(quantity, omega_history, thrust)


# def calc_Td(quantity, omega_history, thrust): 
#     quad = Quadrotor3D()
#     h = np.cos(np.pi / 4) * quad.length
#     k = quad.c
#     tau = np.zeros(3)
#     I_v = np.array([[quad.J[0], 0, 0], 
#                 [0, quad.J[1], 0], 
#                 [0, 0, quad.J[2]]])

#     omega_dot = (- omega_history[-2] + omega_history[-1])*PERIODIC_FREQUENCY
#     tourque_asli = I_v @ omega_dot

#     tourqe_ext = 0.3*(tourque_asli - quantity.reshape((3,1)))

#     tourque_want_to_gen = quantity.reshape((3,1)) - tourqe_ext

#     gu_3 = tourque_want_to_gen + np.cross(omega_history[-1].reshape(3,), np.dot(I_v, omega_history[-1]).reshape(3,))#3,1
#     Td = np.sum(thrust)
#     gu = np.array([Td, gu_3[0][0],gu_3[1][0],gu_3[2][0]]).reshape(4,1)
#     print(gu.shape)
#     G = np.array([[1, 1, 0, 1], 
#             [-h, h, 0, -h], 
#             [h, -h, 0, -h], 
#             [k, k, 0, -k]])
#     G_inv = np.linalg.pinv(G)

#     u = G_inv @ gu
#     return (u/quad.max_thrust).reshape((4,))








































    # tau[0] = -h*thrust[0] + h*thrust[1] + h*thrust[2] - h*thrust[3]
    # tau[1] = +h*thrust[0] - h*thrust[1] + h*thrust[2] - h*thrust[3]
    # tau[2] = k*thrust[0] + k*thrust[1] - k*thrust[2] - k*thrust[3]  

    # w_dot_f, tau_f = update_filter(omega_dot, tau)
    # w_dot_f = update_low_pass_fw(omega_dot)
    # tau_f = update_low_pass_ftau(tau)

#     tau_d = np.zeros(3)
#     tau_d[0] = tau_f[0] + quantity[0]-np.dot(I_v, w_dot_f)[0]
#     tau_d[1] = tau_f[1] + quantity[1]-np.dot(I_v, w_dot_f)[1]
#     tau_d[2] = tau_f[2] + quantity[2]-np.dot(I_v, w_dot_f)[2]

#     T = thrust[0]+thrust[1]+thrust[2]+thrust[3]
#     return indi_thrust(tau_d, T)


# def indi_thrust(tau_d, T):
#     quad = Quadrotor3D()
#     h = np.cos(np.pi / 4) * quad.length
#     k = quad.c
#     G = np.array([[1, 1, 0, 1], 
#               [-h, h, 0, -h], 
#               [h, -h, 0, -h], 
#               [k, k, 0, -k]])

#     G_inv = np.linalg.pinv(G)
#     thrust_matrix = np.zeros(4)
#     thrust_matrix[0] = T
#     thrust_matrix[1] = tau_d[0]
#     thrust_matrix[2] = tau_d[1]
#     thrust_matrix[3] = tau_d[2]
#     u = np.dot(G_inv, thrust_matrix)

#     return u

# def indi_init_filters():
#     global filters     # w_df, tau_df

#     w_tau = 1.0/(2.0*np.pi*STABILIZATION_INDI_FILT_CUTOFF_w)
#     tau_tau = 1.0/(2.0*np.pi*STABILIZATION_INDI_FILT_CUTOFF_tau)
#     tau = np.vstack((w_tau, tau_tau))

#     for i in range(2):
#         filters[i, 0].init(tau[i, 0], sample_time, 0.0)
#         filters[i, 1].init(tau[i, 0], sample_time, 0.0)
#         filters[i, 2].init(tau[i, 0], sample_time, 0.0)
#     return

# def update_filter(w, tau):
#     global filters

#     w_new = np.zeros(3)
#     tau_new = np.zeros(3)
#     est = np.vstack((w, tau))
#     est_new = np.vstack((w_new, tau_new))

#     for i in range(2):
#         est_new[i, 0] = filters[i, 0].update(est[i, 0])
#         est_new[i, 1] = filters[i, 1].update(est[i, 1])
#         est_new[i, 2] = filters[i, 2].update(est[i, 2])

#     return est_new[0], est_new[1]

# def finite_difference(new1, old):
#     output = np.zeros(3)
#     for i in range(3):
#         output[i] = (new1[i] - old[i])*PERIODIC_FREQUENCY # output[i] = (new1[i] - old[i]) / dt;
#     return output

# ####################################################
# def init_low_pass_fw(value):
#   global filter_w_last_in
#   global filter_w_last_out 
#   global filter_time_const
#   global tau

#   filter_w_last_in = value
#   filter_w_last_out = value
#   filter_time_const = 2 * tau / 250
  

# def update_low_pass_fw(value):
#   global filter_w_last_in
#   global filter_w_last_out 
#   global filter_time_const

#   out = (value + filter_w_last_in + (filter_time_const - 1) * filter_w_last_out) / (1 + filter_time_const)
#   filter_w_last_in = value
#   filter_w_last_out = out
#   return out


# def init_low_pass_ftau(value):
#   global filter_tau_last_in
#   global filter_tau_last_out 
#   global filter_time_const
#   global tau

#   filter_tau_last_in = value
#   filter_tau_last_out = value
#   filter_time_const = 2 * tau / 250
  

# def update_low_pass_ftau(value):
#   global filter_tau_last_in
#   global filter_tau_last_out 
#   global filter_time_const

#   out = (value + filter_tau_last_in + (filter_time_const - 1) * filter_tau_last_out) / (1 + filter_time_const)
#   filter_tau_last_in = value
#   filter_tau_last_out = out
#   return out
