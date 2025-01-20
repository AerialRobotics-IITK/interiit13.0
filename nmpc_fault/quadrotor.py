from math import sqrt
import numpy as np
from utils import quaternion_to_euler, skew_symmetric, v_dot_q, unit_quat, quaternion_inverse

    #   <jointName>rotor_1_joint</jointName>
    #   <linkName>rotor_1</linkName>
    #   <turningDirection>ccw</turningDirection>
    #   <timeConstantUp>1e-06</timeConstantUp>
    #   <timeConstantDown>1e-06</timeConstantDown>
    #   <maxRotVelocity>1000.0</maxRotVelocity>
    #   <motorConstant>8.54858e-06</motorConstant>
    #   <momentConstant>0.016</momentConstant>
    #   <commandSubTopic>command/motor_speed</commandSubTopic>
    #   <motorNumber>1</motorNumber>
    #   <rotorDragCoefficient>8.06428e-05</rotorDragCoefficient>
    #   <rollingMomentCoefficient>1e-06</rollingMomentCoefficient>
    #   <rotorVelocitySlowdownSim>1e-06</rotorVelocitySlowdownSim>
    #   <motorType>velocity</motorType>

class Quadrotor3D:
    def __init__(self, drag=False, motor_noise=False):
        """Initialize the Quadrotor3D class with options for drag and motor noise."""

        # Parameters in SI Units

        self.magic_const = 1.0125437762618865
        self.max_rot_vel = 1000.0
        self.motor_const = 17.091776e-06

        self.max_thrust = 14.528009599999999  # maximum thrust
        self.k_1000= 17.091776e-03
        # self.J = np.array([0.033555106642101415, 0.03365803145244886, 0.06387979708741934])  # moment of inertia of quadrotor
        # self.J = np.array([0.028, 0.028, 0.06387979708741934])  # moment of inertia of quadrotor

        self.J = np.array([0.02361518496,
                           0.02371810977,
                           0.04399995371])  # moment of inertia of quadrotor according to sdf file (including motors)
        
        self.mass = 2.064307692 # Mass of Quadrotor
        self.length = 0.174  # half of the full arm length
        self.c = 0.016  # z-torque constant for motors

        # Input constraints
        self.max_input_value = 1
        self.min_input_value = 0


        self.pos = np.array([0, 0, -0.5])
        self.vel = np.array([0, 0, 0])
        self.angle = np.array([1., 0., 0., 0.])
        self.a_rate = np.zeros(3)

        # Initialize state
        # self.pos = np.array([0, 0, 0])
        # self.vel = np.array([0, 0, 0])
        # self.angle = np.array([1., 0., 0., 0.])
        # self.a_rate = np.zeros(3)

        # Motor configurations for thrust torques

        # h = np.cos(np.pi / 4) * self.length
        h = self.length
        self.y_f  = np.array([-h, h, h, -h])
        self.x_f = np.array([h, -h, h, -h])
        self.z_l_tau = np.array([-self.c, self.c, -self.c, self.c])

        # Gravity and disturbances
        self.g = np.array([[0], [0], [9.81]])
        self.u = np.array([0, 0, 0, 0])
        self.drag = drag
        self.motor_noise = motor_noise

        # Drag coefficients
        self.rotor_drag_xy = 0.3
        self.rotor_drag_z = 0.0
        self.rotor_drag = np.array([self.rotor_drag_xy, self.rotor_drag_xy, self.rotor_drag_z])[:, np.newaxis]
        self.aero_drag = 0.08

    def set_state(self, *args, **kwargs):
        """Set the state of the quadrotor based on positional, angular, velocity, and rate inputs."""
        if args:
            assert len(args) == 1 and len(args[0]) == 13
            self.pos[:], self.angle[:], self.vel[:], self.a_rate[:] = np.split(args[0], [3, 7, 10])
        else:
            self.pos, self.angle, self.vel, self.a_rate = kwargs["pos"], kwargs["angle"], kwargs["vel"], kwargs["rate"]

    def get_state(self, quaternion=True, stacked=False):
        """Retrieve the current state of the quadrotor in quaternion or Euler angle format, as specified."""
        angle = quaternion_to_euler(self.angle) if not quaternion else self.angle
        state = [self.pos, angle, self.vel, self.a_rate]
        if stacked:
            return np.concatenate([self.pos, angle, self.vel, self.a_rate])
        return state

    def get_control(self):
        """Get the current control inputs."""
        return self.u

    def update(self, u, dt):
        """Update the state variables based on thrust input and a timestep `dt`."""
        self.u[:] = np.clip(u, self.min_input_value, self.max_input_value) * self.max_thrust

        # Generate disturbances if motor noise is enabled
        f_d = np.random.normal(0, 10 * dt, (3, 1)) if self.motor_noise else np.zeros((3, 1))
        t_d = np.random.normal(0, 10 * dt, (3, 1)) if self.motor_noise else np.zeros((3, 1))

        # Runge-Kutta method for state update
        x = self.get_state(quaternion=True, stacked=False)
        k1 = [self.f_pos(x), self.f_att(x), self.f_vel(x, self.u, f_d), self.f_rate(x, self.u, t_d)]

        # print("################################")
        # print(f"pos_d: {k1[0]}")
        # print(f"att_d: {k1[1]}")
        # print(f"vel_d: {k1[2]}")
        # print(f"rate_d: {k1[3]}")
        # print("################################")
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]
        k3 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]
        k4 = [self.f_pos(x_aux), self.f_att(x_aux), self.f_vel(x_aux, self.u, f_d), self.f_rate(x_aux, self.u, t_d)]

        self.pos, self.angle, self.vel, self.a_rate = [x[i] + dt * (1/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])) for i in range(4)]
        self.angle = unit_quat(self.angle)

    def f_pos(self, x):
        """Calculate positional change rate based on current velocity."""
        return x[2]

    def f_att(self, x):
        """Calculate rate of change of attitude based on angular rates."""
        return 0.5 * skew_symmetric(x[3]).dot(x[1])

    def f_vel(self, x, u, f_d):
        """Calculate the acceleration based on thrust, gravity, and drag (if enabled)."""
        # print("################################")
        # print(self.mass)
        # print("################################")
        a_thrust = (np.array([[0], [0], [np.sum(u)]]) / self.mass) 

        # print(f"a_thrust: {a_thrust}")
        # print(f"g: {self.g}")

        v_b = v_dot_q(x[2], quaternion_inverse(x[1]))[:, np.newaxis] if self.drag else np.zeros((3, 1))
        a_drag = v_dot_q(-self.aero_drag * v_b ** 2 * np.sign(v_b) / self.mass - self.rotor_drag * v_b / self.mass, x[1]) if self.drag else np.zeros((3, 1))
        
        return np.squeeze(self.g + a_drag - v_dot_q(a_thrust + f_d / self.mass, x[1]))

    def f_rate(self, x, u, t_d):
        """Calculate angular acceleration based on torques generated by thrust and any disturbances."""
        rate = x[3]
        return np.array([
            (u.dot(self.y_f) + t_d[0] + (self.J[1] - self.J[2]) * rate[1] * rate[2]) / self.J[0],
            (u.dot(self.x_f) + t_d[1] + (self.J[2] - self.J[0]) * rate[2] * rate[0]) / self.J[1],
            (u.dot(self.z_l_tau) + t_d[2] + (self.J[0] - self.J[1]) * rate[0] * rate[1]) / self.J[2]
        ]).squeeze()
