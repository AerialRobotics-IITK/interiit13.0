import os
import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from quadrotor import Quadrotor3D
from utils import skew_symmetric, v_dot_q, quaternion_inverse, decompose_quaternion, quaternion_error
from casadi import SX, vertcat


class Controller:
    def __init__(self, quad: Quadrotor3D, t_horizon=1, n_nodes=2000,
                 q_cost=None, r_cost=None, q_mask=None, rdrv_d_mat=None,
                 model_name="quad", solver_options=None, initial_state=None):
        """
        Initialize MPC controller for quadrotor control
        
        Args:
            quad: Quadrotor3D model instance
            t_horizon: Time horizon for MPC
            n_nodes: Number of control nodes within horizon
            q_cost: State cost weights
            r_cost: Control input cost weights
            q_mask: Mask for state costs
            rdrv_d_mat: Drag coefficient matrix
            model_name: Name for the acados model
            solver_options: Additional solver configuration
        """
        # Q-Matrix px, py, pz, qxy, qz, vx, vy, vz, wx, wy, wz
        q_cost = np.array([80, 80, 800, 600, 0, 1, 1, 1, 50, 50, 1])
        # R-Matrix u1, u2, u3, u4
        r_cost = np.array([10, 10, 10, 10])

        # Define Parameters
        self.T = t_horizon
        self.N = n_nodes
        self.quad = quad
        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Define state variables
        self.p = cs.MX.sym('p', 3)  # position
        self.q = cs.MX.sym('a', 4)  # quaternion orientation
        self.v = cs.MX.sym('v', 3)  # velocity
        self.r = cs.MX.sym('r', 3)  # angular velocity

        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13
        
        

        # Control inputs
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(u1, u2, u3, u4)

        # Quaternion norm constraint
        f_thrust = self.u * self.quad.max_thrust
        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)

        # h = cs.vertcat(cs.sqrt(cs.sumsqr(self.q)), cs.mtimes(f_thrust.T, y_f), cs.mtimes(f_thrust.T, x_f), cs.mtimes(f_thrust.T, c_f))
        h = cs.vertcat(cs.sqrt(cs.sumsqr(self.q)), self.r)
        

        # Reference states
        self.p_ref = cs.MX.sym('p_ref', 3)
        self.q_ref = cs.MX.sym('q_ref', 4)
        self.v_ref = cs.MX.sym('v_ref', 3)
        self.r_ref = cs.MX.sym('r_ref', 3)
        self.u_ref = cs.MX.sym('u_ref', 4)

        state_refs = cs.vertcat(self.p_ref, self.q_ref, self.v_ref, self.r_ref)
        self.all_refs = cs.vertcat(state_refs, self.u_ref)

        # Create nominal dynamics model
        self.quad_xdot_nominal = self.quad_dynamics(rdrv_d_mat)

        # Initialize objective function components
        self.L = None
        self.target = None

        # Setup acados model and solver
        acados_models, nominal_with_gp = self.acados_setup_model(
            self.quad_xdot_nominal(x=self.x, u=self.u)['x_dot'], model_name)

        # Create dynamics functions
        self.quad_xdot = {
            0:nominal_with_gp
        }

        self.acados_ocp_solver = {}
        
        # Setup cost weights
        # q_diagonal = np.concatenate((q_cost[:3], q_cost[3:]))
        # if q_mask is not None:
        #     q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
        #     q_diagonal *= q_mask

        # Configure and create acados solver

        key_model = acados_models[0]
        
        ocp = AcadosOcp()
        ocp.model = key_model
        
        # Extract model dimensions and parameters
        nx = key_model.x.size()[0]
        nu = key_model.u.size()[0]
        x = key_model.x
        u = key_model.u
        p = key_model.p
        n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

        # Configure horizon and solver settings
        ocp.solver_options.N_horizon = self.N
        ocp.dims.N = self.N
        ocp.solver_options.tf = t_horizon
        ocp.dims.np = n_param
        ocp.parameter_values = np.concatenate([initial_state, [0, 0, 0, 0]])

        # Setup cost functions
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'

        Q = np.diag(np.concatenate((q_cost, r_cost)))
        Q_e = np.diag(q_cost)
        terminal_cost = 5 if solver_options is None or not solver_options["terminal_cost"] else 1
        Q_e *= terminal_cost
        R = np.diag(r_cost)

        # Define stage and terminal costs
        ocp.model.cost_expr_ext_cost = self.cost_vector(x, p).T @ Q @ self.cost_vector(x, p) + u.T @ R @ u 
        ocp.model.cost_expr_ext_cost_e = self.cost_vector_terminal(x, p).T @ Q_e @ self.cost_vector_terminal(x, p) 
        ocp.model.con_h_expr = h

        # Set initial state and constraints
        #x_ref = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        x_ref = np.array(initial_state)

        print("##########################################################")
        print(initial_state)
        print("##########################################################")
        ocp.constraints.x0 = x_ref

        # Input constraints
        ocp.constraints.lbu = np.array([self.min_u, self.min_u, self.min_u, self.min_u])
        ocp.constraints.ubu = np.array([self.max_u, self.max_u, self.max_u, self.max_u])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.lh = np.array([0.98, -2, -2, -2])
        ocp.constraints.uh = np.array([1.02, 2, 2, 2])

        # Solver settings
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.qp_tol = 1e-2
        ocp.solver_options.nlp_tol = 1e-2
        ocp.solver_options.nlp_tol_eq = 1e-2
        ocp.solver_options.nlp_tol_ineq = 1e-2

        # Generate and compile solver
        json_file = os.path.join('./', key_model.name + 'hello_acados_ocp.json')
        try:
            AcadosOcpSolver.generate(ocp, json_file=json_file)
        except Exception as e:
             print(f"Error generating JSON file: {e}")

        self.acados_ocp_solver[0] = AcadosOcpSolver(ocp, json_file=json_file)
        

    def cost_vector(self, x, y_ref):
        """
        Compute the stage cost vector for optimization
        """
        p_ref = y_ref[0:3]
        q_ref = y_ref[3:7]
        v_ref = y_ref[7:10]
        w_ref = y_ref[10:13]
        u_ref = y_ref[13:17]
        
        p_err = x[0:3] - p_ref
        q_err = quaternion_error(x[3:7], q_ref)
        q_xy, q_z = decompose_quaternion(q_err)
        q_xy_x, q_xy_y = q_xy[1], q_xy[2]
        q_z_z = q_z[3]
        
        v_err = x[7:10] - v_ref
        w_err = x[10:13] - w_ref
        u_err = self.u - u_ref
        q_xy_error = q_xy_x**2 + q_xy_y**2

        return vertcat(p_err, q_xy_error, q_z_z, v_err, w_err, u_err)
    
    def cost_vector_terminal(self, x, yref):
        """
        Compute the terminal cost vector for optimization
        """
        p_ref = yref[0:3]
        q_ref = yref[3:7]
        v_ref = yref[7:10]
        w_ref = yref[10:13]
        
        p_err = x[0:3] - p_ref
        q_err = quaternion_error(x[3:7], q_ref)
        q_xy, q_z = decompose_quaternion(q_err)
        q_xy_x, q_xy_y = q_xy[1], q_xy[2]
        q_z_z = q_z[3]
        
        v_err = x[7:10] - v_ref
        w_err = x[10:13] - w_ref
        q_xy_error = q_xy_x**2 + q_xy_y**2

        return vertcat(p_err, q_xy_error, q_z_z, v_err, w_err)
    
    def acados_setup_model(self, nominal, model_name):
        """
        Build acados symbolic models using CasADi expressions
        """
        def fill_in_acados_model(x, u, p, dynamics, name):
            x_dot = cs.MX.sym('x_dot', dynamics.shape)
            f_impl = x_dot - dynamics
            
            model = AcadosModel()
            model.f_expl_expr = dynamics
            # model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {0: nominal}
        acados_models[0] = fill_in_acados_model(
            x=self.x, u=self.u, p=self.all_refs, dynamics=nominal, name=model_name)

        return acados_models, dynamics_equations

    def quad_dynamics(self, rdrv_d):
        """
        Compute symbolic dynamics of the 3D quadrotor model
        """
        x_dot = cs.vertcat(self.p_dynamics(), self.q_dynamics(), 
                          self.v_dynamics(rdrv_d), self.w_dynamics())
        return cs.Function('x_dot', [self.x[:13], self.u], [x_dot], ['x', 'u'], ['x_dot'])

    def p_dynamics(self):
        """Position dynamics"""
        return self.v

    def q_dynamics(self):
        """Quaternion dynamics"""
        return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self, rdrv_d):
        """
        Velocity dynamics with optional drag compensation
        """
        f_thrust = self.u * self.quad.max_thrust
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, - f_thrust[0] - f_thrust[1] - 
                            f_thrust[2] - f_thrust[3]) / self.quad.mass

        v_dynamics = v_dot_q(a_thrust, self.q) + g

        if rdrv_d is not None:
            v_b = v_dot_q(self.v, quaternion_inverse(self.q))
            rdrv_drag = v_dot_q(cs.mtimes(rdrv_d, v_b), self.q)
            v_dynamics += rdrv_drag

        return v_dynamics

    def w_dynamics(self):
        """Angular velocity dynamics"""
        f_thrust = self.u * self.quad.max_thrust
        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)
        
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * 
             self.r[1] * self.r[2]) / self.quad.J[0],
            (cs.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * 
             self.r[2] * self.r[0]) / self.quad.J[1],
            (cs.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * 
             self.r[0] * self.r[1]) / self.quad.J[2])

    def run_optimization(self, initial_state=None, goal=None, use_model=0, return_x=False, mode='pose'):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param goal: 3 element [x,y,z] for moving to goal mode, 3*(N+1) for trajectory tracking mode
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :param mode: string, whether to use moving to pose mode or tracking mode
        :return: optimized control input sequence (flattened)
        """

        # Set initial state
        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]
            # print("Exiting....")
            exit(1)
        x_init = initial_state

        # print("---------------")
        # print(initial_state)
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[use_model].set(0, 'lbx', x_init)
        self.acados_ocp_solver[use_model].set(0, 'ubx', x_init)

        # Set final condition

        for j in range(self.N):
            y_ref = np.array([goal[0], goal[1], goal[2], 1,0,0,0, 0,0,0, 0,0,0, 0.68,0.68,0.68,0.68])
            self.acados_ocp_solver[use_model].set(j, 'p', y_ref)
            # print(y_ref)
        y_refN = np.array([goal[0], goal[1], goal[2], 1,0,0,0, 0,0,0, 0,0,0, 0.68,0.68,0.68,0.68])
        self.acados_ocp_solver[use_model].set(self.N, 'p', y_refN)
    

        # Solve OCP
        self.acados_ocp_solver[use_model].solve()

        # Get optimal u
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[use_model].get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver[use_model].get(i, "u")
            
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[use_model].get(i + 1, "x")


        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return (w_opt_acados)
