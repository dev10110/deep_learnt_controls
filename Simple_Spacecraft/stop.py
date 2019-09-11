import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics

class STOP:
    # simulated trajectory optimization program
    # name inspiried by NASA's POST algorithm.

    def __init__(self, rocket):

        self.rocket = rocket

        rocket.lunar_guidance_live() #calculate apollo guidance

        #create the problem instance
        time_init = [0.0, rocket.t[-1]]
        n = 12
        num_states    = self.num_states   = 5
        num_controls  = self.num_controls = 2
        max_iteration = 10

        prob = self.prob = Problem(time_init, [n], [num_states], [num_controls], max_iteration)
        prob.dynamics    = [self._og_dynamics]
        prob.cost        = self._og_cost
        prob.runing_cost = self._og_running_cost
        prob.equality    = self._og_equality
        prob.inequality  = self._og_inequality
        #prob.knot_states_smooth = [True]

        # generate intial guess of trajectory
        self._initial_guess_apollo() #assigns the initial guess trajectory - one with no thrust

        return


    def solve(self, mesh_refine=1, disp_plots = False):
        """ Performs the trajectory optimization

        Parameters
        ----------
        mesh_refine : int
            Number of times to refine (defaults to doubling) mesh
        disp_plots : boolean
            plot intermediate plots?

        Returns
        -------
        tuple
            Tuple containing (time, (state), (controls))

        """
        print('********* Starting Solve ***********')
        self.mesh_refine_count = 0
        prob = self.prob

        #solve it
        self.disp_plots = disp_plots

        prob.solve(self.rocket, self._display_func, ftol=1e-8)
        print(f'********* Solved Mesh iteration: {self.mesh_refine_count} ***********')


        #mesh refine

        while self.mesh_refine_count < mesh_refine:
            prob = self.prob = self.global_mesh_refine(self.prob)
            prob.solve(self.rocket, self._display_func,ftol=1e-10)
            self.mesh_refine_count += 1
            print(f'********* Solved Mesh iteration: {self.mesh_refine_count} ***********')


        print('********* Solved! ***********')
        ## TODO: perform sanity check on the control scheme.

        #returns a (t, state, action) tuple
        t = self.t_sol = prob.time_update()
        s = self.s_sol = list((prob.states_all_section(i) for i in range(self.num_states)))
        c = self.c_sol = list((prob.controls_all_section(i) for i in range(self.num_controls)))

        return (t, s, c)

    def _initial_guess_apollo(self):

        # perform apollo guidance, and then optimize

        prob = self.prob
        r = self.rocket

        #r.lunar_guidance_live() #stores the solution into the parameters

        t_vec = prob.time_all_section

        x_guess = np.interp(t_vec, r.t, r.x)
        z_guess = np.interp(t_vec, r.t, r.z)
        vx_guess = np.interp(t_vec, r.t, r.vx)
        vz_guess = np.interp(t_vec, r.t, r.vz)
        m_guess = np.interp(t_vec, r.t, r.m)

        u1_guess = np.interp(t_vec, r.t, r.u1_mag)
        u2_guess = np.interp(t_vec, r.t, r.u2_angle)

        prob.set_states_all_section(0, x_guess)
        prob.set_states_all_section(1, z_guess)
        prob.set_states_all_section(2, vx_guess)
        prob.set_states_all_section(3, vz_guess)
        prob.set_states_all_section(4, m_guess)

        prob.set_controls_all_section(0, u1_guess)
        prob.set_controls_all_section(1, u2_guess)

        return



    def _initial_guess_no_thrust(self):
        """Short summary.

        Returns
        -------
        type
            Description of returned object.

        """
        ''' Determine and store an initial guess for the parameters. Initial guess based on solution to no thrusting'''

        prob = self.prob
        r = self.rocket

        #set guess based on no thrusting
        x_guess  = np.array([r.x_0 + r.vx_0*t for t in prob.time_all_section])
        z_guess  = np.array([r.z_0 + r.vz_0*t - 0.5*r.g*t**2 for t in prob.time_all_section])
        vx_guess = np.array([r.vx_0           for t in prob.time_all_section])
        vz_guess = np.array([r.vz_0 - r.g*t   for t in prob.time_all_section])
        m_guess  = Guess.linear(prob.time_all_section, r.m_0 , 0.95*r.m_0)

        u1_guess = Guess.constant(prob.time_all_section, 0.01)
        u2_guess = Guess.constant(prob.time_all_section, 0)

        prob.set_states_all_section(0, x_guess)
        prob.set_states_all_section(1, z_guess)
        prob.set_states_all_section(2, vx_guess)
        prob.set_states_all_section(3, vz_guess)
        prob.set_states_all_section(4, m_guess)

        prob.set_controls_all_section(0, u1_guess)
        prob.set_controls_all_section(1, u2_guess)

        return




    def _og_dynamics(self, prob, obj, section):
        """ Wrapper function to pass the rocket's dynamics into openGoddard """


        #extract states and controls
        s = tuple([prob.states(i, section) for i in range(self.num_states)])
        u = tuple([prob.controls(i, section) for i in range(self.num_controls)])

        #create Dynamics block
        dx = Dynamics(prob, section)

        #get dynamics from the rocket
        ds = obj.dynamics(s, u)

        for i in range(self.num_states):
            dx[i] = ds[i]

        return dx()


    def _og_equality(self, prob, obj):
        """ define the equality conditions to be met. in theory the user should be able to provide/modify this"""

        result = Condition()

        for i in range(5):
            result.equal(prob.states_all_section(i)[0], obj.s_0[i]), #r, v, m = [r,v,m](0)

        for i in range(4):
            result.equal(prob.states_all_section(i)[-1], 0.) #r, v = 0

        return result()



    def _og_inequality(self, prob, obj):

        """ define inequality conditions """

        s = [prob.states_all_section(i) for i in range(5)]



        result = Condition()

        u_m = prob.controls(0, 0)
        u_a = prob.controls(1, 0)

        result.lower_bound(u_m, 0)
        result.upper_bound(u_m, 1)
        result.lower_bound(u_a, (-np.pi/2)) #thrust force to the left
        result.upper_bound(u_a, (+np.pi/2)) #thrust force to the right

        #bound states
        for i in range(5):
            result.lower_bound(s[i], obj.ranges[i][0])
            result.upper_bound(s[i], obj.ranges[i][1])

        #cone constraint
        result.lower_bound(s[1], 1.*s[0]) # z > (1/tan(th)) * x

        return result()


    def _og_cost(self, prob, obj):

        mf = prob.states_all_section(4)[-1]

        return (obj.alpha/2)*(obj.m_0 - mf)


    def _og_running_cost(self, prob, obj):

        u1 = prob.controls_all_section(0)

        J = (1/obj.c2)*((1-obj.alpha)*(obj.gamma1*obj.c1**2*u1**2) + (obj.alpha/2)*(obj.c1*u1))

        return J


    def _display_func(self):
        """Display function during solve.
        """

        prob = self.prob
        r = self.rocket
        print(f'Mesh Refine Count: {self.mesh_refine_count}')
        print(f'mf/m0   : {prob.states_all_section(4)[-1]/r.m_0}')
        print(f'tf/tchar: {prob.time_final(-1)/r.t_char}')

        if self.disp_plots:
            f, (ax1, ax2, ax3)= plt.subplots(1, 3)

            ax1.plot(prob.states_all_section(0), prob.states_all_section(1))
            ax1.set_aspect(aspect=1)
            ax2.plot(prob.time_update(), prob.controls_all_section(0))
            ax3.plot(prob.time_update(), prob.controls_all_section(1))

            ax1.set_title('path')
            ax2.set_title('thrust fraction')
            ax3.set_title('thrust angle')

            ax3.set_ylim([-np.pi/2, np.pi/2])

            ax1.grid()
            ax2.grid()
            ax3.grid()
            plt.show()

        return




    def global_mesh_refine(self, prob = None, new_nodes = None, maxIterator = None, **kwargs):
        """ Refines the mesh. Does not modify current prob

        Parameters
        ----------
        new_nodes : type
            Description of parameter `new_nodes`.
        maxIterator : type
            Description of parameter `maxIterator`.
        **kwargs : type
            Description of parameter `**kwargs`.

        Returns
        -------
        type
            Description of returned object.

        """

        if prob is None:
            prob = self.prob
        #else refine the provided problem

        #get parameters from previous prob
        time_init = prob.time_knots()

        if new_nodes is None:
            new_nodes = [2*n for n in prob.nodes]

        num_of_states = prob.number_of_states
        num_of_controls = prob.number_of_controls

        if maxIterator is None:
            maxIterator = prob.maxIterator

        # construct new problem
        new_prob = Problem(time_init, new_nodes, num_of_states, num_of_controls, maxIterator, **kwargs)


        #assume the equations are the same as before
        new_prob.dynamics           = prob.dynamics
        new_prob.knot_states_smooth = prob.knot_states_smooth
        new_prob.cost               = prob.cost
        new_prob.cost_derivative    = prob.cost_derivative
        new_prob.equality           = prob.equality
        new_prob.inequality         = prob.inequality


        #add in the initial guesses - linear interp on previous solution
        t_previous = prob.time_update() #update total
        t_previous_section = prob.time

        t_new = new_prob.time_all_section
        t_new_section = new_prob.time

        for sec in range(len(new_nodes)):
            for i in range(num_of_states[sec]):
                new_prob.set_states(i, sec, np.interp(t_new_section[sec], t_previous_section[sec], prob.states(i, sec)))

            for i in range(num_of_controls[sec]):
                new_prob.set_controls(i, sec, np.interp(t_new_section[sec], t_previous_section[sec], prob.controls(i, sec)))

        return new_prob


    def plot_solution(self):

        prob = self.prob
        r = self.rocket

        t = prob.time_all_section

        #plot the path
        plt.figure()

        plt.plot(prob.states_all_section(0), prob.states_all_section(1),'x-')
        plt.plot(r.x, r.z, 'r--')
        ax = plt.gca()
        ax.set_aspect('equal', 'datalim')
        plt.grid()
        plt.title('path')
        plt.show()

        t_vec = r.t
        states = (r.x, r.z, r.vx, r.vz, r.m)
        controls = (r.u1_mag, r.u2_angle)

        #plot the breakdown in states
        for i in range(self.num_states):
            plt.figure()
            plt.plot(t, prob.states_all_section(i),'x-')
            plt.plot(t_vec, states[i],'r--')
            plt.ylabel(f'state {i}')
            plt.grid()
            plt.show()

        #plot the breakdown in controls
        for i in range(self.num_controls):
            plt.figure()
            plt.plot(t, prob.controls_all_section(i), 'x-')
            plt.plot(t_vec, controls[i],'r--')
            plt.ylabel(f'control {i}')
            plt.grid()
            plt.show()
