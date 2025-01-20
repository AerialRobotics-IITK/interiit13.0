#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition, OffboardControlMode, VehicleControlMode, TrajectorySetpoint, VehicleStatus, VehicleCommand, VehicleAttitudeSetpoint, VehicleLocalPositionSetpoint, VehicleGlobalPosition, ActuatorMotors, VehicleThrustSetpoint, VehicleOdometry

import numpy as np
import math
import timeit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D 

from quadrotor import Quadrotor3D
from controller import Controller

class Offboard(Node):
	def __init__(self):
		super().__init__("controller")

		# QOS Profiles
		qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
		qos_profile_gt = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)

		# Publishers
		self.offboard_mode_publisher = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile)
		self.vehicle_command_publisher = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_profile)
		self.actuator_motors_publisher = self.create_publisher(ActuatorMotors, "/fmu/in/actuator_motors", qos_profile)
		self.vehicle_thrust_publisher = self.create_publisher(VehicleThrustSetpoint, "/fmu/in/vehicle_thrust_setpoint", qos_profile)
		self.trajectory_publisher = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile)

		# Subscribers
		self.vehicle_status_subscriber = self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.vehicle_status_callback, qos_profile)
		self.status_subscriber = self.create_subscription(VehicleControlMode, "/fmu/out/vehicle_control_mode", self.state_callback, qos_profile)
		self.attitude_subscirber = self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_callback, qos_profile)
		self.local_position_subscriber = self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_position_callback, qos_profile)
		self.global_position_subscriber = self.create_subscription(VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.global_position_callback, qos_profile)
		self.vehicle_odometry_subscriber = self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.vehicle_odometry_callback, qos_profile)

		# Subscriber Messages Received
		self.vehicle_status = None
		self.state = None
		self.attitude = None
		self.local_position = None
		self.global_position = None
		self.vehicle_odometry = None

		# Controller
		self.dt = 0.01
		self.N = 100
		self.quad = Quadrotor3D()
		self.controller = Controller(self.quad, t_horizon=2*self.N*self.dt, n_nodes=self.N)
		self.final_goal = np.array([0, 0, 5])
		self.path, self.thrust_history = [], []
		self.iteration = 0
		self.hover = False

		# Timer
		self.time_period_drone = 0.01
		self.timer = self.create_timer(self.time_period_drone, self.command_loop)
		self.counter = 0

	# Subscriber Callback Functions
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
	def disengage_offboard_mode(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
		msg.param1 = 1.0
		msg.param2 = 3.0
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
	def kill(self):
		instance_num = 1
		msg = VehicleCommand()
		msg.command = VehicleCommand.VEHICLE_CMD_DO_FLIGHTTERMINATION
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

	def command_loop(self):
		if self.counter < 10000:
			self.publish_trajectory(0.0, 0.0, -5.0)
			self.offboard_control_heartbeat_signal_publisher("position")
		else:
			self.offboard_control_heartbeat_signal_publisher("direct_actuator")
		if self.counter == 1000:
			self.engage_offboard_mode()
			self.arm()
		elif self.counter > 10000:
			position = [-self.vehicle_odometry.position[0], -self.vehicle_odometry.position[1], -self.vehicle_odometry.position[2]]
			angle = [self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]]
			velocity = [self.vehicle_odometry.velocity[0], self.vehicle_odometry.velocity[1], self.vehicle_odometry.velocity[2]]
			angular_velocity = [self.vehicle_odometry.angular_velocity[0], self.vehicle_odometry.angular_velocity[1], self.vehicle_odometry.angular_velocity[2]]
			current = np.concatenate([position, angle, velocity, angular_velocity])
			thrust = self.controller.run_optimization(initial_state=current, goal=self.final_goal, use_model=0)[:4]
			self.quad.update(thrust, self.dt)
			self.set_thrust(thrust[0], thrust[1], thrust[2], thrust[3])
		self.counter += 1

# Print statement formatting
def print_quad_state(iteration = '', current=None, thrust=None, hover_status=None):
    hover_line = f"Hover Status:                 {('True' if hover_status else 'False')}" if hover_status is not None else ""
    print(f"""
    ------------------------ Quad State Iteration {iteration} ------------------------
    Position (x, y, z):           ({current[0]:.4f}, {current[1]:.4f}, {current[2]:.4f})
    Quaternion (q0, q1, q2, q3):  ({current[3]:.4f}, {current[4]:.4f}, {current[5]:.4f}, {current[6]:.4f})
    Velocity (vx, vy, vz):        ({current[7]:.4f}, {current[8]:.4f}, {current[9]:.4f})
    Angular Rate (wx, wy, wz):    ({current[10]:.4f}, {current[11]:.4f}, {current[12]:.4f})
    Thrust (t0, t1, t2, t3):      ({thrust[0]:.4f}, {thrust[1]:.4f}, {thrust[2]:.4f}, {thrust[3]:.4f})
    {hover_line}
    --------------------------------------------------------------------------
    """)


def ControlledLandingTrajectory(sim_time, dt):

    # Controlled landing trajectory that follows a converging helix
    vertical_speed = -0.5
    initial_radius = 3
    convergence_rate = 0.15
    initial_z = 10

    xref, yref, zref = [], [], []

    for i in range(int(sim_time / dt)):
        t = dt * i
        radius = initial_radius * np.exp(-convergence_rate * t)
        x = radius * np.sin(t)
        y = radius * (1- np.cos(t))
        z = max(initial_z + vertical_speed * t, 0.2)  # Ensure z doesn't go below 0.2

        xref.append(x)
        yref.append(y)
        zref.append(z)
    
    return np.array(xref), np.array(yref), np.array(zref)

def ControlledLanding():
    dt = 0.01
    sim_time = 24   
    N = 200
    quad = Quadrotor3D()
    controller = Controller(quad, t_horizon=2*N*dt, n_nodes=N)

    xref, yref, zref = ControlledLandingTrajectory(sim_time, dt)
    path, time_record, velocity_record = [], [], []

    # Declare initial state for controlled landing
    quad.pos = np.array([0, 0, 10])
    quad.vel = np.array([1, 3, 2])
    quad.angle = np.array([1., 0., 0., 0.])
    quad.a_rate = np.array([1, 2, 2])
    thrust = np.array([2, 2, 3, 4])
    current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
    print("This is the current state from which we are trying to land")
    print_quad_state(None, current, thrust)
    time.sleep(2)

    # Controlled descent loop
    for i in range(int(sim_time/dt)):
        x, y, z = xref[i:i+N+1], yref[i:i+N+1], zref[i:i+N+1]
        if len(x) < N+1:
            x = np.pad(x, (0, N+1-len(x)), constant_values=xref[-1])
            y = np.pad(y, (0, N+1-len(y)), constant_values=yref[-1])
            z = np.pad(z, (0, N+1-len(z)), constant_values=zref[-1])
        goal = np.array([x, y, z]).T

        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
        start = timeit.default_timer()
        thrust = controller.run_optimization(initial_state=current, goal=goal, mode='traj', use_model=0)[:4]
        time_record.append(timeit.default_timer() - start)
        print_quad_state(i, current, thrust)
        quad.update(thrust, dt)
        path.append(quad.pos)
        velocity_record.append(quad.vel)

    # Timing info
    print("Average estimation time: {:.5f}".format(np.mean(time_record)))
    print("Max estimation time: {:.5f}".format(np.max(time_record)))
    print("Min estimation time: {:.5f}".format(np.min(time_record)))

    # Visualization
    path = np.array(path)
    velocity_record = np.array(velocity_record)
    time_axis = np.linspace(0, sim_time, len(velocity_record))

    # 3D path plot
    plt.figure()
    ax = plt.axes(projection='3d')
    #ax.plot(xref, yref, zref, c=[1,0,0], label='goal')
    ax.plot(path[:,0], path[:,1], path[:,2], label='trajectory')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()

    # CPU time plot
    plt.figure()
    plt.plot(time_record)
    plt.ylabel('CPU Time [s]')
    plt.xlabel('Iteration')
    plt.title('Computation Time per Iteration')

    # Velocity plots
    plt.figure()
    plt.plot(time_axis, velocity_record[:, 0], label='x-velocity')
    plt.plot(time_axis, velocity_record[:, 1], label='y-velocity')
    plt.plot(time_axis, velocity_record[:, 2], label='z-velocity')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity vs Time')
    plt.legend()

    plt.show()


def move2Goal():
    dt = 0.001 
    N = 100
    quad = Quadrotor3D()
    controller = Controller(quad, t_horizon=2 * N * dt, n_nodes=N)

    # Final Destination Position Coordinates x, y, z
    final_goal = np.array([0, 0, 5])

    # Setting waypoints for optimal computation
    num_waypoints = max(int(max(final_goal) / 10) + 1, 1)
    waypoints = [quad.pos + i * (final_goal - quad.pos) / num_waypoints for i in range(1, num_waypoints + 1)]
    for i in range(100):
        waypoints.append(final_goal)
    print('#######################')
    print(len(waypoints))
    print('#######################')
    vicinity_radius = 0.01

    path, thrust_history = [], []
    iteration = 0  # Add iteration counter

    # Navigation loop through waypoints
    for goal in waypoints:
        while np.linalg.norm(goal - quad.pos) > vicinity_radius:
            current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
            thrust = controller.run_optimization(initial_state=current, goal=goal, use_model=0)[:4]
            thrust_history.append(thrust)
            quad.update(thrust, dt)
            path.append(quad.pos)
            print_quad_state(iteration, current, thrust)
            iteration += 1
    with open("thrust_move2goal.csv", 'w') as file:
        file.write('\n'.join(f"{thrust[0]},{thrust[1]},{thrust[2]},{thrust[3]}" for thrust in (thrust_history)))
    

    print("Goal tracked successfully!")

    # Visualization
    path = np.array(path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='path')
    ax.scatter(final_goal[0], final_goal[1], final_goal[2], c='red', label='final goal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    plt.show()


def StableHover():
    dt, N = 0.01, 100
    quad = Quadrotor3D()
    controller = Controller(quad, t_horizon=2 * N * dt, n_nodes=N)
    final_goal = np.array([0, 0, 5])

    path, thrust_history, counter = [], [], 0
    iteration = 0  # Add iteration counter
    hover = False
    # Hover at goal until stability is reached
    while counter < 500:
        if np.linalg.norm(final_goal - quad.pos) < 0.2:
            hover = True
            counter += 1
            if counter == 1:
                print("Drone is now hovering!")
                time.sleep(2)
        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
        thrust = controller.run_optimization(initial_state=current, goal=final_goal, use_model=0)[:4]
        if(counter < 1):
            thrust_history.append(thrust)
        quad.update(thrust, dt)
        path.append(quad.pos)
        print_quad_state(iteration, current, thrust, hover)  # Use iteration counter
        iteration += 1  # Increment counter
    for i in range(500):
        thrust_history.append([0.5922287228270210217076988224945, 0.5922287228270210217076988224945, 0.5922287228270210217076988224945, 0.5922287228270210217076988224945])
    with open("thrust_hover.csv", 'w') as file:
        file.write('\n'.join(f"{thrust[0]},{thrust[1]},{thrust[2]},{thrust[3]}" for thrust in (thrust_history)))

    # Visualization
    path = np.array(path)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='path')
    ax.scatter(final_goal[0], final_goal[1], final_goal[2], c='red', label='final goal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()
    plt.show()


def BasicNavigationTrajectory(sim_time, dt):
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

def BasicNavigation():
    dt, sim_time, N = 0.01, 15, 200
    quad = Quadrotor3D()
    controller = Controller(quad, t_horizon=2*N*dt, n_nodes=N)

    xref, yref, zref = BasicNavigationTrajectory(sim_time, dt)
    path, time_record = [], []

    # Navigation loop following 8-shape trajectory
    for i in range(int(sim_time/dt)):
        x, y, z = xref[i:i+N+1], yref[i:i+N+1], zref[i:i+N+1]
        if len(x) < N+1:
            x = np.pad(x, (0, N+1-len(x)), constant_values=xref[-1])
            y = np.pad(y, (0, N+1-len(y)), constant_values=yref[-1])
            z = np.pad(z, (0, N+1-len(z)), constant_values=zref[-1])
        goal = np.array([x, y, z]).T

        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
        start = timeit.default_timer()
        thrust = controller.run_optimization(initial_state=current, goal=goal, mode='traj', use_model=0)[:4]
        time_record.append(timeit.default_timer() - start)
        print_quad_state(i, current, thrust)
        quad.update(thrust, dt)
        path.append(quad.pos)

    # Timing info
    print("Average estimation time: {:.5f}".format(np.mean(time_record)))
    print("Max estimation time: {:.5f}".format(np.max(time_record)))
    print("Min estimation time: {:.5f}".format(np.min(time_record)))

    # Visualization
    path = np.array(path)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(xref, yref, zref, c=[1,0,0], label='goal')
    ax.plot(path[:,0], path[:,1], path[:,2])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend()

    plt.figure()
    plt.plot(time_record)
    plt.ylabel('CPU Time [s]')
    plt.show()

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
