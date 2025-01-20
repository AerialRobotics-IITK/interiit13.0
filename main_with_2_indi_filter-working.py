#! /usr/bin/env python3

import rclpy
import pdb
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import csv

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from actuator_msgs.msg import Actuators
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, OffboardControlMode, VehicleControlMode, TrajectorySetpoint, VehicleStatus, VehicleCommand, VehicleAttitudeSetpoint, VehicleLocalPositionSetpoint, VehicleGlobalPosition, ActuatorMotors, VehicleThrustSetpoint, VehicleOdometry

import os
import numpy as np
import math
import timeit
import matplotlib
import quaternion
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from quadrotor_filter import Quadrotor3D
from controller_with_2_filter import Controller
from indi_utils_filter import INDI
from indi import Butterworth2LowPass as LPF

iter = 0

class Offboard(Node):
	def __init__(self):
		super().__init__("controller")

		# QOS Profiles
		qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
		qos_profile_gt = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
		qos_profile_mavros = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

		# Publishers
		self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile)
		self.vehicle_command_publisher = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_profile)
		self.actuator_motors_publisher = self.create_publisher(ActuatorMotors, "/fmu/in/actuator_motors", qos_profile)
		self.vehicle_thrust_publisher = self.create_publisher(VehicleThrustSetpoint, "/fmu/in/vehicle_thrust_setpoint", qos_profile)
		self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile)

		# Subscribers
		## PX4
		self.vehicle_status_subscriber = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_profile)
		self.status_subscriber = self.create_subscription(VehicleControlMode, "/fmu/out/vehicle_control_mode", self.state_callback, qos_profile)
		self.attitude_subscirber = self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_callback, qos_profile)
		self.local_position_subscriber = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_position_callback, qos_profile)
		self.global_position_subscriber = self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_position_callback, qos_profile)
		self.vehicle_odometry_subscriber = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.vehicle_odometry_callback, qos_profile)
		self.vehicle_actuator_sub = self.create_subscription(ActuatorMotors, "/fmu/out/actuator_motors", self.actuator_callback, qos_profile)
		## MAVROS
		self.imu_subscriber = self.create_subscription(Imu, "/mavros/imu/data", self.imu_callback, qos_profile_mavros)
		self.local_position_mavros_subscriber = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.local_position_mavros_callback, qos_profile_mavros)
		self.velocity_subscriber = self.create_subscription(TwistStamped, "/mavros/local_position/velocity_body", self.velocity_callback, qos_profile_mavros)
		self.actuator_vel_subscriber = self.create_subscription(Actuators, "/x500_0/command/motor_speed", self.actuator_vel_callback, qos_profile_mavros)

		# Subscriber Messages Received
		## PX4
		self.vehicle_status = None
		self.state = None
		self.attitude = None
		self.local_position = None
		self.global_position = None
		self.vehicle_odometry = None
		self.actuator_thrusts = None
		self.actuator_vels = None
		self.prev_actuator_vels = None
		## MAVROS
		self.imu_data = None
		self.local_position_mavros = None
		self.velocity = None


		self.controller = None
		self.controller1 = None
		self.controller2 = None

		self.prev_actuator_vel = None
		self.actuator_vel =  None
		# Controller

		self.N = 25
		self.dt = 1/self.N 

		# tau = 1/(2*pi*Fc)

		self.omega_x_filter = LPF()
		self.omega_y_filter = LPF()
		self.omega_z_filter = LPF()

		self.actuator_filter1 = LPF()
		self.actuator_filter2 = LPF()
		self.actuator_filter3 = LPF()
		self.actuator_filter4 = LPF()

		self.omega_Fc = 8
		self.omega_tau = 1.0/(2.0*np.pi*self.omega_Fc)

		self.actuator_Fc = 8
		self.actuator_tau = 1.0/(2.0*np.pi*self.omega_Fc)
		
		self.omega_x_filter.init(self.omega_tau, sample_time=self.dt, value=0.)
		self.omega_y_filter.init(self.omega_tau, sample_time=self.dt, value=0.)
		self.omega_z_filter.init(self.omega_tau, sample_time=self.dt, value=0.)

		self.actuator_filter1.init(self.actuator_tau, sample_time=self.dt, value=150.)
		self.actuator_filter2.init(self.actuator_tau, sample_time=self.dt, value=150.)
		self.actuator_filter3.init(self.actuator_tau, sample_time=self.dt, value=150.)
		self.actuator_filter4.init(self.actuator_tau, sample_time=self.dt, value=150.)
 

		self.quad = Quadrotor3D()
		self.final_goal = np.array([0, 0, 1])
		self.path, self.thrust_history = [], []
		self.hover = False

		# Timer
		self.time_period_drone = self.dt
		self.timer = self.create_timer(self.time_period_drone, self.command_loop)
		self.counter = 0

		with open("dta.csv", mode="w+", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(["pos_x", "pos_y", "pos_z", "q_w", "q_x", "q_y", "q_z", "vel_x", "vel_y", "vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z", "t1", "t2", "t3", "t4"] + ["x"]*13*21)

		with open("check_dta.csv", mode="w+", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(["ideal_pos_x", "ideal_pos_y", "ideal_pos_z", "actual_pos_x", "actual_pos_y", "actual_pos_z"])

		with open("check_indi.csv", mode="w+", newline="") as file:
			writer = csv.writer(file)
			writer.writerow(["indi_thrust1", "indi_thrust2", "indi_thrust3", "indi_thrust4", "actual_thrust1", "actual_thrust2", "actual_thrust3", "actual_thrust4"])
	# Subscriber Callback Functions
	## PX4
	def vehicle_status_callback(self, vehicle_status_msg):
		self.vehicle_status = vehicle_status_msg
	def state_callback(self, status_msg):
		self.status = status_msg
	def attitude_callback(self, attitude_msg):
		self.attitude = attitude_msg
	def local_position_callback(self, local_position_msg):
		self.local_position = local_position_msg
	def global_position_callback(self, global_position_msg):
		self.global_position = global_position_msg
	def vehicle_odometry_callback(self, vehicle_odometry_msg):
		self.vehicle_odometry = vehicle_odometry_msg
	## MAVROS
	def imu_callback(self, imu_msg):
		self.imu_data = imu_msg
	def local_position_mavros_callback(self, local_position_mavros_msg):
		self.local_position_mavros = local_position_mavros_msg
	def velocity_callback(self, velocity_msg):
		self.velocity = velocity_msg
	def actuator_callback(self, msg):
		self.actuator_thrusts = msg.control[:4]

	def actuator_vel_callback(self, msg):
		self.prev_actuator_vels = self.actuator_vels
		self.actuator_vels = msg.velocity[:4]


	def offboard_control_heartbeat_signal_publisher(self, what_control):
		msg = OffboardControlMode()
		msg.position = False
		msg.velocity = False
		msg.acceleration = False
		msg.attitude = False
		msg.body_rate = False
		msg.thrust_and_torque = False
		msg.direct_actuator = False
		match what_control:
			case 'position':
				msg.position = True
			case 'velocity':
				msg.velocity = True
			case 'acceleration':
				msg.acceleration = True
			case 'attitude':
				msg.attitude = True
			case 'body_rate':
				msg.body_rate = True
			case 'thrust_and_torque':
				msg.thrust_and_torque = True
			case 'direct_actuator':
				msg.direct_actuator = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.offboard_mode_publisher.publish(msg)
	def engage_offboard_mode(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
		msg.param1 = 1.0
		msg.param2 = 6.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = 0.0
		msg.param6 = 0.0
		msg.param7 = 0.0
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)
	def arm(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
		msg.param1 = 1.0
		msg.param2 = 0.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = 0.0
		msg.param6 = 0.0
		msg.param7 = 0.0
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)
	def disarm(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
		msg.param1 = 0.0
		msg.param2 = 0.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = 0.0
		msg.param6 = 0.0
		msg.param7 = 0.0
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)
	def takeoff(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF
		msg.param1 = 0.0
		msg.param2 = 0.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = 0.0
		msg.param6 = 0.0
		msg.param7 = 0.0
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)
	def set_origin(self, latitude, longitude, altitude):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_SET_GPS_GLOBAL_ORIGIN
		msg.param1 = 0.0
		msg.param2 = 0.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = latitude
		msg.param6 = longitude
		msg.param7 = altitude
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)
	def individual_motor(self, motor_1, motor_2, motor_3, motor_4):
		msg = ActuatorMotors()
		msg.control[0] = motor_1
		msg.control[1] = motor_2
		msg.control[2] = motor_3
		msg.control[3] = motor_4
		msg.control[4] = float('nan')
		msg.control[5] = float('nan')
		msg.control[6] = float('nan')
		msg.control[7] = float('nan')
		msg.control[8] = float('nan')
		msg.control[9] = float('nan')
		msg.control[10] = float('nan')
		msg.control[11] = float('nan')
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.actuator_motors_publisher.publish(msg)
	def set_thrust(self, x, y, z):
		msg = VehicleThrustSetpoint()
		msg.xyz[0] = x
		msg.xyz[1] = y
		msg.xyz[2] = z
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_thrust_publisher.publish(msg)
	def publish_trajectory(self, x, y, z):
		msg = TrajectorySetpoint()
		msg.position[0] = x
		msg.position[1] = y
		msg.position[2] = z
		msg.velocity[0] = float('nan')
		msg.velocity[1] = float('nan')
		msg.velocity[2] = float('nan')
		msg.acceleration[0] = float('nan')
		msg.acceleration[1] = float('nan')
		msg.acceleration[2] = float('nan')
		msg.jerk[0] = float('nan')
		msg.jerk[1] = float('nan')
		msg.jerk[2] = float('nan')
		msg.yaw = 1.57079
		msg.yawspeed = float('nan')
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.trajectory_publisher.publish(msg)
	def land(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
		msg.param1 = 0.0
		msg.param2 = 0.0
		msg.param3 = 0.0
		msg.param4 = 0.0
		msg.param5 = 0.0
		msg.param6 = 0.0
		msg.param7 = 0.0
		msg.target_system = instance_num
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		msg.timestamp = self.get_clock().now().nanoseconds//1000
		self.vehicle_command_publisher.publish(msg)

	def command_loop(self):
		# print("running......")
		self.actuator_vel = [None, None, None, None]


		if self.vehicle_odometry == None:
			print("No odom!!")
			return
		if self.actuator_vels == None:
			print("No vel!!")
			return
		
		elif self.local_position_mavros != None and self.imu_data != None and self.velocity != None and (self.counter == 0 or self.counter == 400):

			position = [self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.vehicle_odometry.position[2]]
			orientation = [self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]]
			velocity = [self.vehicle_odometry.velocity[0], self.vehicle_odometry.velocity[1], self.vehicle_odometry.velocity[2]]
			angular_velocity = [self.vehicle_odometry.angular_velocity[0], self.vehicle_odometry.angular_velocity[1], self.vehicle_odometry.angular_velocity[2]]
			
			
			if self.counter == 400:
				self.final_goal = [position[0], position[1], position[2]+2] 
				self.counter += 1
				return
			else:
				self.final_goal = [position[0], position[1], position[2]-2] 
				

			#self.final_goal = [1, 2, 3]

			print("starting....")
			print(orientation)
			initial_state = []
			initial_state.extend(position)
			initial_state.extend(orientation)
			initial_state.extend(velocity)
			initial_state.extend(angular_velocity)

			initial_state_2 = initial_state[:]
			initial_state_2[2] = initial_state_2[2] - 5

			self.quad.set_state(pos=np.array(position), angle=np.array(orientation), vel=np.array(velocity), rate=np.array(angular_velocity))
			self.controller = Controller(self.quad, t_horizon=1, n_nodes=self.N, initial_state=initial_state, model_name='quad')

			# self.controller1 = Controller(self.quad, t_horizon=1, n_nodes=self.N, initial_state=initial_state_2, model_name='unbroken')
			# self.controller2 = Controller(self.quad, t_horizon=1, n_nodes=self.N, initial_state=initial_state_2, broken=True, model_name='broken')


			self.counter += 1
			return
		elif self.counter == 0:
			return
		if self.counter < 200:
			self.individual_motor(0.0, 0.0, 0.0, 0.0)
		
		self.offboard_control_heartbeat_signal_publisher("direct_actuator")
		solver_x = None

		if self.counter == 100:
			self.engage_offboard_mode()
			self.arm()

		elif self.counter > 200:
			print("running!!!")
			

			position = [self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.vehicle_odometry.position[2]]
			orientation = [self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]]
			velocity = [self.vehicle_odometry.velocity[0], self.vehicle_odometry.velocity[1], self.vehicle_odometry.velocity[2]]
			angular_velocity = [self.vehicle_odometry.angular_velocity[0], self.vehicle_odometry.angular_velocity[1], self.vehicle_odometry.angular_velocity[2]]
	

			for ii in range(4):
				if self.actuator_vels[ii] != 0.:
					self.actuator_vel[ii] = self.actuator_vels[ii]
				else:
					self.actuator_vel[ii] = self.prev_actuator_vel[ii]
			
			self.prev_actuator_vel = self.actuator_vel

			actuator_vel = np.array(self.actuator_vel)

			current = np.concatenate([position, orientation, velocity, angular_velocity])
			start1 = time.process_time()
			start = Node.get_clock(self).now()

			


			if(self.counter<400):
				thrust_arr, x_opt = self.controller.run_optimization(initial_state=current, goal=self.final_goal, use_model=0, broken=False)
				thrust = thrust_arr[0:4]

				# state = self.quad.get_state()
				# print(x_opt)

				# print(np.shape(x_opt[0, :]))
				ideal_p = x_opt[0, :3]
				ideal_w = x_opt[0, -3:]
				
				print(f"w according to dynamics: {ideal_w}")
				# exit(1)
				
				thrust_0_1 = thrust
				print(f"u_solver: {thrust_0_1}")
				print(f"thrust according to solver: {thrust_0_1*self.quad.max_thrust}")

				actual_thrust=((thrust*self.quad.max_thrust)/self.quad.k_1000 + 150)/1000
				# thrust = actual_thrust.reshape(4,1)

				k = 2

				prev_omega_x = self.omega_x_filter.get()
				prev_omega_y = self.omega_y_filter.get()
				prev_omega_z = self.omega_z_filter.get()
				prev_omega = [prev_omega_x, prev_omega_y, prev_omega_z]

				curr_actuator_vel = np.array([
								self.actuator_filter1.get(),
								self.actuator_filter2.get(),
								self.actuator_filter3.get(),
								self.actuator_filter4.get()
							]) 
				omega_x = self.omega_x_filter.get()
				omega_y = self.omega_y_filter.get()
				omega_z = self.omega_z_filter.get()
				omega = [omega_x, omega_y, omega_z]

				indi_thrust = INDI(omega, prev_omega, curr_actuator_vel, np.array(ideal_w), thrust_arr[0:4])

				thrust = []
				for i in range(4):
					curr = 1*actual_thrust[i] - 0.0*indi_thrust[i]
					thrust.append(curr)
				thrust = np.array(thrust).reshape(4,)
				# print(np.shape(thrust))
				# print(thrust)
				# exit(1)

				# print(f"thrust_shape: {np.shape(thrust[0, 0])}")
				
				self.individual_motor(thrust[0], thrust[1], thrust[2], thrust[3])
				print(f"pre_switch: {thrust}")

				# self.quad.update(thrust_0_1, self.dt)

				

			else:
				thrust_arr, x_opt = self.controller.run_optimization(initial_state=current, goal=[self.final_goal[0],self.final_goal[1],self.final_goal[2]], use_model=0, broken=True)
				# #LAND 

				# thrust_arr, x_opt = self.controller.run_optimization(initial_state=current, goal=[self.final_goal[0],self.final_goal[1],self.final_goal[2]], use_model=0, broken=True)
				#GOTO

				thrust = thrust_arr[0:4]
				
				ideal_p = x_opt[0, :3]
				ideal_w = x_opt[0, -3:]
				
				print(f"w according to dynamics: {ideal_w}")
				# exit(1)
				
				thrust_0_1 = thrust
				print(f"u_solver: {thrust_0_1}")
				print(f"thrust according to solver: {thrust_0_1*self.quad.max_thrust}")

				actual_thrust=((thrust*self.quad.max_thrust)/self.quad.k_1000 + 150)/1000
				# thrust = actual_thrust.reshape(4,1)

				k = 2

				prev_omega_x = self.omega_x_filter.get()
				prev_omega_y = self.omega_y_filter.get()
				prev_omega_z = self.omega_z_filter.get()
				prev_omega = [prev_omega_x, prev_omega_y, prev_omega_z]

				curr_actuator_vel = np.array([
								self.actuator_filter1.get(),
								self.actuator_filter2.get(),
								self.actuator_filter3.get(),
								self.actuator_filter4.get()
							]) 
				omega_x = self.omega_x_filter.get()
				omega_y = self.omega_y_filter.get()
				omega_z = self.omega_z_filter.get()
				omega = [omega_x, omega_y, omega_z]

				indi_thrust = INDI(omega, prev_omega, curr_actuator_vel, np.array(ideal_w), thrust_arr[0:4])

				thrust = []
				print("Running INDI:")
				for i in range(4):
					curr = 1.0*actual_thrust[i] - 0.0*indi_thrust[i]
					thrust.append(curr)
				thrust = np.array(thrust).reshape(4,)
				# print(np.shape(thrust))
				# print(thrust)
				# exit(1)

				# print(f"thrust_shape: {np.shape(thrust[0, 0])}")
				
				# if self.counter < 450:
				self.individual_motor(thrust[0], thrust[1], thrust[2], thrust[3])
				print(f"pre_switch: {thrust}")

				# self.quad.update(thrust_0_1, self.dt)

				# enable_indi = False

				# if enable_indi:
				# 	thrust = INDI(self.omega_x, self.omega_y, self.omega_z, self.actuator_vels, thrust)
				# 	self.individual_motor(thrust[0], thrust[1], thrust[2], thrust[3])
				# 	print(f"post_switch_after_indi: {thrust}")
				# else:
				# 	thrust=((thrust*self.quad.max_thrust)/self.quad.k_1000 + 150)/1000
				# 	self.individual_motor(thrust[0], thrust[1], thrust[2], thrust[3])
				# 	print(f"post_switch: {thrust}")
				
				# self.quad.update(thrust_0_1, self.dt) #check by moving below as well

				if self.counter >= 425:
					self.individual_motor(0.0, 0.01, 0.0, 0.0)
					# self.land()

					self.disarm()
					exit(1)
			
			
			# print(self.actuator_vels_buffer1)
			
			self.omega_x_filter.update(angular_velocity[0])
			self.omega_y_filter.update(angular_velocity[1])
			self.omega_z_filter.update(angular_velocity[2])
			
			print(actuator_vel)
			# exit(1)
			self.actuator_filter1.update(actuator_vel[0])
			self.actuator_filter2.update(actuator_vel[1])
			self.actuator_filter3.update(actuator_vel[2])
			self.actuator_filter4.update(actuator_vel[3])

			with open("dta.csv", mode="a", newline="") as file:
				writer = csv.writer(file)
				writer.writerow(np.concatenate((current, thrust.reshape(4,), x_opt.reshape((13*self.N+13,)))))

			with open("check_dynamics.csv", mode="a", newline="") as file:
				writer = csv.writer(file)
				print(f"\nPosition according to Gazebo: {position}")
				print(f"w according to Gazebo: {angular_velocity}")
				# print(f"\nPosition according to Dynamics: {ideal_p}")
				# writer.writerow(np.concatenate((np.array(ideal_p).reshape(3,), position)))
			
			with open("check_indi.csv", mode="a", newline="") as file:
				writer = csv.writer(file)
				# print(thrust.reshape(4,))
				writer.writerow(np.concatenate((thrust.reshape(4,), actual_thrust.reshape(4,))))

		print(self.counter)
		self.counter += 1
	
	def BasicNavigationTrajectory(sim_time, dt, current_state):
		# Generate 8-shape trajectory
		xref, yref, zref = [], [], []

		for i in range(int(sim_time / dt)):
			t = dt * i
			radius = 8 * np.exp(-0.07 * t)
			x = radius * np.sin(2 * t)
			y = radius * np.sin(t)
			z = 3

			xref.append(x)
			yref.append(y)
			zref.append(z)

		return np.array(xref), np.array(yref), np.array(zref)
	

def main(args=None):
	rclpy.init(args=args)
	offboard = Offboard()
	rclpy.spin(offboard)
	offboard.destroy_node()
	rclpy.shutdown()

if __name__ == "__main__":
    #move2Goal()
    #StableHover()
    #BasicNavigation()
    # ControlledLanding()
    main()
