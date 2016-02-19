"""
A task in which a bug-looking robot chases a ball.

In this task, the robot's sensors inform it about the relative position
of the ball, which changes often, but not about the absolute position 
of the ball or about the robot's own absolute position.
"""
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os

import core.tools as tools
from worlds.base_world import World as BaseWorld
import becca_world_chase_ball.clock_step as cs

class World(BaseWorld):
    """ 
    A ball-chasing dog robot world.
    
    In this two-dimensional world the robot can spin and move 
    forward and backward. It can sense both how far away the 
    ball is and in which direction. In order to succeed in this world,
    the robot has to learn what to do for a given combination of
    ball heading and ball range values.

    The physics in this world are intended to be those of the physical
    world, at least to the depth of an introductory mechanics class.

    Attributes
    ----------

    """
    def __init__(self, lifespan=None):
        """ 
        Set up the world.

        Parameters
        ----------
        lifespan : int
            The number of time steps during which the robot will try to
            catch the ball. If None, it will be set to a default determined
            by the ``BaseWorld`` class.
        """
        BaseWorld.__init__(self, lifespan)
        timesteps_per_second = 4.
        self.clockticks_per_timestep = int(1000. / timesteps_per_second)
        self.plot_feature_set = False
        filming = False
        if filming:
            #self.timesteps_per_frame = 1
            # Render the world for creating a 30 frame-per-second video
            self.timesteps_per_frame = timesteps_per_second / 30. 
            self.lifespan = 250
            # Don't plot features while filming
            self.plot_feature_set = False
        else:
            self.timesteps_per_frame = 1000 
        self.clockticks_per_frame = int(self.clockticks_per_timestep * 
                                        self.timesteps_per_frame)
        self.clockticks_until_render = self.clockticks_per_frame
        self.world_visualize_period = self.timesteps_per_frame
        self.brain_visualize_period = 1e3
        self.name = 'chase_26' 
        self.name_long = 'ball chasing world'
        print "Entering", self.name_long
        self.n_bump_heading = 1
        self.n_bump_mag = 3
        self.n_ball_heading = 17
        self.n_prox_heading = 2
        self.n_ball_range = 11
        self.n_prox_range = 1
        self.n_prox = self.n_prox_heading * self.n_prox_range
        self.n_bump = self.n_bump_heading * self.n_bump_mag
        # Must be odd
        self.n_vel_per_axis = 3 
        # Must be odd
        self.n_acc_per_axis = 1 
        self.n_vel = 3 * self.n_vel_per_axis
        self.n_acc = 3 * self.n_acc_per_axis

        self._initialize_world()
        self.num_sensors = (self.n_ball_range + self.n_ball_heading + 
                            self.n_prox + self.n_bump + 
                            self.n_vel + self.n_acc)
        self.num_actions = 20
        self.action = np.zeros((self.num_actions, 1))
        self.catch_reward = 1.
        self.touch_reward = 1e-8
        self.bump_penalty = 1e-2
        self.effort_penalty = -1e-8
        self.world_directory = 'becca_world_chase_ball'
        self.log_directory = os.path.join(self.world_directory, 'log')
        self.frames_directory = os.path.join(self.world_directory, 'frames') 
        self.frame_counter = 10000

    def _initialize_world(self):
        """
        Set up the physics of the simulation.
        """
        self.clock_tick = 0.
        self.tick_counter = 0
        self.clock_time = 0.

        #-----------------
        # This section of constants defines the dimensions and physics of 
        # the ball chase world. These are also present in cs.clock_step()   
        self.dt = .001 # seconds per clock tick

        # wall parameters
        self.width = 5.#16 # meters
        self.depth = 5.#9. # meters
        self.k_wall =  3000. # Newtons / meter

        # ball parameters
        self.r_ball = .4 # meters
        self.k_ball = 3000. # Newtons / meter
        self.c_ball = 1. # Newton-seconds / meter
        self.cc_ball = 1. # Newton
        self.m_ball = 1. # kilogram
        self.mouth_width_bot = np.pi / 3 # angle over which the bot can catch the ball

        # robot parameters
        self.r_bot = .8 # meters
        self.k_bot =  3000. # Newtons / meter
        self.c_bot = 10. # Newton-seconds / meter
        self.cc_bot = 1. # Newton
        self.d_bot = 10. # Newton-meter-seconds / radian
        self.dd_bot = 1. # Newton-meter
        self.m_bot = 5. # kilogram
        self.I_bot = 1. # kilogram-meters**2
        #-------------------------

        # state variables
        # These continue to evolve. The are initialized here.
        # ball state
        self.x_ball = 4. # meters
        self.y_ball = 4. # meters
        self.vx_ball = 0. # meters / second
        self.vy_ball = 0. # meters / second
        self.ax_ball = 0. # meters**2 / second
        self.ay_ball = 0. # meters**2 / second
    
        # bot state
        self.x_bot = 2. # meters
        self.y_bot = 2. # meters
        self.th_bot = np.pi / 4. # radians
        self.vx_bot = 0. # meters / second
        self.vy_bot = 0. # meters / second
        self.omega_bot = 0. # radians / second
        self.ax_bot = 0. # meters**2 / second
        self.ay_bot = 0. # meters**2 / second
        self.alpha_bot = 0. # radians / second**2
        
        # detector parameters
        min_vision_range = -1
        min_prox_range = -1.

        max_vision_range = self.width * 1.2
        max_prox_range = self.width * 1.2
        max_bump_mag = .2
        max_v_fwd = 3.8
        max_v_lat = 3.5
        max_omega = 8.
        max_a_fwd = 50.
        max_a_lat = 35.
        max_alpha = 125.
        """
        Calculate radial bins that get wider as they move outward
        in a geometric sequence.
        """
        def build_bins_one_sided(n, max_edge, min_edge=-.01):
            """
            Build an array of bin edges from (near) zero to some positive value
            
            The array of bin edges produced will be of the form
                [min_edge, 0, a_2, a_3, ..., a_n-1]
            where a_i form a geometric sequence to max_edge
                a_i = max_edge * 2 ** i/n - 1 
            """
            bins = [min_edge]
            for i in range(n - 1):
                fraction = float(i) / (n - 1)
                bin_edge = max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)
            return np.array(bins)

        self.ball_range_bins = build_bins_one_sided(self.n_ball_range, 
                                                    max_vision_range, 
                                                    min_vision_range)
        self.prox_range_bins = build_bins_one_sided(self.n_prox_range, 
                                                    max_prox_range, 
                                                    min_prox_range)
        self.bump_mag_bins = build_bins_one_sided(self.n_bump_mag, 
                                                    max_bump_mag)

        def build_bins_two_sided(n, max_edge):
            """
            Build an array of bin edges that is symmetric about zero
            
            The array of bin edges produced will be of the form
                [-BIG, -a_n, ..., -a_2, -a_1, a_1, a_2, ..., a_n]
            where a_i form a geometric sequence to max_edge
                a_i = max_edge * 2 ** i/(n+1) - 1 
            """
            #bins = [-tools.big]
            bins = [-1e6]
            for i in range(n):
                j = n - (i + 1)
                fraction = float(j + 1) / (n + 1)
                bin_edge = -max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)
            for i in range(n):
                fraction = float(i + 1) / (n + 1)
                bin_edge = max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)
            return np.array(bins)

        n_vel_half = (self.n_vel_per_axis - 1) / 2
        n_acc_half = (self.n_acc_per_axis - 1) / 2
        self.v_fwd_bins = build_bins_two_sided(n_vel_half, max_v_fwd)
        self.v_lat_bins = build_bins_two_sided(n_vel_half, max_v_lat)
        self.omega_bins = build_bins_two_sided(n_vel_half, max_omega)
        self.a_fwd_bins = build_bins_two_sided(n_acc_half, max_a_fwd)
        self.a_lat_bins = build_bins_two_sided(n_acc_half, max_a_lat)
        self.alpha_bins = build_bins_two_sided(n_acc_half, max_alpha)

        self.v_heading = np.zeros(self.n_ball_heading)
        self.v_range = np.zeros(self.n_ball_range)
        self.prox = np.zeros((self.n_prox_heading, self.n_prox_range))
        self.bump = np.zeros((self.n_bump_heading, self.n_bump_mag))
        self.v_fwd_bot_sensor = np.zeros(self.n_vel_per_axis)
        self.v_lat_bot_sensor = np.zeros(self.n_vel_per_axis)
        self.omega_bot_sensor = np.zeros(self.n_vel_per_axis)
        self.a_fwd_bot_sensor = np.zeros(self.n_acc_per_axis)
        self.a_lat_bot_sensor = np.zeros(self.n_acc_per_axis)
        self.alpha_bot_sensor = np.zeros(self.n_acc_per_axis)

        self.n_catch = 0.

        # Create a prototype force profile.
        # It is a triangular profile, ramping linearly from 0 to peak over its
        # first half, than ramping linearly back down to 0.
        duration = .25
        peak = 1. # Newtons
        length = int(duration / self.dt)
        self.proto_force = np.ones(length)
        self.proto_force[:length/2] = (np.cumsum(self.proto_force)[:length/2] * 
                                       peak / float(length/2))
        self.proto_force[length:length-length/2-1:-1] = (
                self.proto_force[:length/2])
        self.proto_force[length/2] = peak
        
        # create a ring buffer for future force information
        buffer_duration = 1. # seconds
        buffer_length = int(buffer_duration / self.dt)
        self.f_x_buffer = np.zeros(buffer_length)
        self.f_y_buffer = np.zeros(buffer_length)
        self.tau_buffer = np.zeros(buffer_length)

    def zero_sensors(self):
        self.bump = np.zeros(self.bump.shape)
        self.v_heading = np.zeros(self.v_heading.shape)
        self.v_range = np.zeros(self.v_range.shape)
        self.prox = np.zeros(self.prox.shape)
        self.v_fwd_bot_sensor = np.zeros(self.v_fwd_bot_sensor.shape)
        self.v_lat_bot_sensor = np.zeros(self.v_lat_bot_sensor.shape)
        self.omega_bot_sensor = np.zeros(self.omega_bot_sensor.shape)
        self.a_fwd_bot_sensor = np.zeros(self.a_fwd_bot_sensor.shape)
        self.a_lat_bot_sensor = np.zeros(self.a_lat_bot_sensor.shape)
        self.alpha_bot_sensor = np.zeros(self.alpha_bot_sensor.shape)

    def convert_detectors_to_sensors(self):
        """
        Construct a sensor vector from the detector values
        """
        self.sensors = np.zeros(self.num_sensors)
        last = 0

        first = last
        last = first + self.n_bump
        self.sensors[first:last] = self.bump.ravel()
        
        first = last
        last = first + self.n_ball_heading
        self.sensors[first:last] = self.v_heading.ravel()

        first = last
        last = first + self.n_ball_range
        self.sensors[first:last] = self.v_range.ravel()

        first = last
        last = first + self.n_prox
        self.sensors[first:last] = self.prox.ravel()

        first = last
        last = first + self.n_vel_per_axis
        self.sensors[first:last] = self.v_fwd_bot_sensor.ravel()

        first = last
        last = first + self.n_vel_per_axis
        self.sensors[first:last] = self.v_lat_bot_sensor.ravel()

        first = last
        last = first + self.n_vel_per_axis
        self.sensors[first:last] = self.omega_bot_sensor.ravel()

        first = last
        last = first + self.n_acc_per_axis
        self.sensors[first:last] = self.a_fwd_bot_sensor.ravel()

        first = last
        last = first + self.n_acc_per_axis
        self.sensors[first:last] = self.a_lat_bot_sensor.ravel()

        first = last
        last = first + self.n_acc_per_axis
        self.sensors[first:last] = self.alpha_bot_sensor.ravel()

        self.zero_sensors()
    
    def convert_sensors_to_detectors(self, sensors, show=False):
        """
        Construct a sensor vector from the detector values
        """
        self.zero_sensors()

        last = 0
        first = last
        last = first + self.n_bump
        self.bump = sensors[first:last].reshape(self.bump.shape)
        
        first = last
        last = first + self.n_ball_heading
        self.v_heading = sensors[first:last].reshape(self.v_heading.shape)

        first = last
        last = first + self.n_ball_range
        self.v_range = sensors[first:last].reshape(self.v_range.shape)

        first = last
        last = first + self.n_prox
        self.prox = sensors[first:last].reshape(self.prox.shape)

        first = last
        last = first + self.n_vel_per_axis
        self.v_fwd_bot_sensor = sensors[first:last].reshape(
                self.v_fwd_bot_sensor.shape)

        first = last
        last = first + self.n_vel_per_axis
        self.v_lat_bot_sensor = sensors[first:last].reshape(
                self.v_lat_bot_sensor.shape)

        first = last
        last = first + self.n_vel_per_axis
        self.v_fwd_bot_sensor = sensors[first:last].reshape(
                self.omega_bot_sensor.shape)

        first = last
        last = first + self.n_acc_per_axis
        self.a_fwd_bot_sensor = sensors[first:last].reshape(
                self.a_fwd_bot_sensor.shape)

        first = last
        last = first + self.n_acc_per_axis
        self.a_lat_bot_sensor = sensors[first:last].reshape(
                self.a_lat_bot_sensor.shape)

        first = last
        last = first + self.n_acc_per_axis
        self.alpha_bot_sensor = sensors[first:last].reshape(
                self.alpha_bot_sensor.shape)

    def list_detectors(self):
        print
        print '___________________________________'
        print 'bump'
        tools.format(self.bump)
        print 'v_heading'
        tools.format(self.v_heading)
        print 'v_range'
        tools.format(self.v_range)
        print 'prox'
        tools.format(self.prox)
        print 'v_fwd_bot_sensor'
        tools.format(self.v_fwd_bot_sensor)
        print 'v_lat_bot_sensor'
        tools.format(self.v_lat_bot_sensor)
        print 'omega_bot_sensor'
        tools.format(self.omega_bot_sensor)
        print 'a_fwd_bot_sensor'
        tools.format(self.a_fwd_bot_sensor)
        print 'a_lat_bot_sensor'
        tools.format(self.a_lat_bot_sensor)
        print 'alpha_bot_sensor'
        tools.format(self.alpha_bot_sensor)

    def step(self, action): 
        """ 
        Take one time step through the world 
        """

        def convert_actions_to_drives(action):
            self.action = action
            # Find the drive magnitude
            self.drive = (  self.action[0] + 
                        2 * self.action[1] + 
                        4 * self.action[2] + 
                        8 * self.action[3] + 
                        16 * self.action[4] - 
                            self.action[5] - 
                        2 * self.action[6] - 
                        4 * self.action[7] -
                        8 * self.action[8] -
                        16 * self.action[9])
            # Find the spin magnitude
            self.spin = (   self.action[10] + 
                        2 * self.action[11] + 
                        4 * self.action[12] + 
                        8 * self.action[13] + 
                        16 * self.action[14] - 
                            self.action[15] - 
                        2 * self.action[16] - 
                        4 * self.action[17] -
                        8 * self.action[18] -
                        16 * self.action[19])
            # Find the total effort
            self.effort = ( self.action[0] + 
                        2 * self.action[1] + 
                        4 * self.action[2] + 
                        8 * self.action[3] + 
                        16 * self.action[4] + 
                            self.action[5] +
                        2 * self.action[6] + 
                        4 * self.action[7] +
                        8 * self.action[8] +
                        16 * self.action[9] +
                            self.action[10] + 
                        2 * self.action[11] + 
                        4 * self.action[12] + 
                        8 * self.action[13] + 
                        16 * self.action[14] + 
                            self.action[15] + 
                        2 * self.action[16] + 
                        4 * self.action[17] +
                        8 * self.action[18] +
                        16 * self.action[19])


        def calculate_reward():
            """ 
            Assign reward based on accumulated target events over the 
            previous time step
            """
            self.reward = 0.
            self.reward += self.n_catch * self.catch_reward
            self.reward += (2 * np.sum(self.bump[:,0]) + 
                            np.sum(self.bump[:,1])) * self.touch_reward
            self.reward -= (2 * np.sum(self.bump[:,-1]) + 
                            np.sum(self.bump[:,-2])) * self.bump_penalty
            self.reward -= self.effort * self.effort_penalty

            self.n_catch = 0.
        
        # Use the convenient internal methods defined above to 
        # step the world forward at a high leve of abstraction.
        self.timestep += 1 
        convert_actions_to_drives(action)

        # Call the high speed simulation code.
        done = False
        clockticks_remaining = self.clockticks_per_timestep
        while not done:
            # Check whether it's time to render the world.
            if self.clockticks_until_render <= 0.: 
                self.render()
                self.clockticks_until_render = self.clockticks_per_frame

            # The number of iterations limited by rendering.
            if clockticks_remaining > self.clockticks_until_render:
                n_clockticks = self.clockticks_until_render
            # The number of iterations limited by the time step.
            else:
                n_clockticks = clockticks_remaining
                done = True

            self.clockticks_until_render -= n_clockticks
            clockticks_remaining -= n_clockticks

            (self.clock_tick, 
            self.clock_time, 
            self.n_catch,
            self.x_ball,
            self.y_ball,
            self.vx_ball,
            self.vy_ball,
            self.ax_ball,
            self.ay_ball,
            self.x_bot,
            self.y_bot,
            self.th_bot,
            self.vx_bot,
            self.vy_bot,
            self.omega_bot,
            self.ax_bot,
            self.ay_bot,
            self.alpha_bot) = cs.clock_step(
                    self.clock_tick, 
                    self.clock_time, 
                    n_clockticks,
                    self.clockticks_per_timestep,
                    self.ball_range_bins,
                    self.prox_range_bins,
                    self.bump_mag_bins,
                    self.v_fwd_bins,
                    self.v_lat_bins,
                    self.omega_bins,
                    self.a_fwd_bins,
                    self.a_lat_bins,
                    self.alpha_bins,
                    self.v_heading,
                    self.v_range,
                    self.prox,
                    self.bump,
                    self.v_fwd_bot_sensor,
                    self.v_lat_bot_sensor,
                    self.omega_bot_sensor,
                    self.a_fwd_bot_sensor,
                    self.a_lat_bot_sensor,
                    self.alpha_bot_sensor,
                    self.n_catch,
                    self.proto_force,
                    self.f_x_buffer,
                    self.f_y_buffer,
                    self.tau_buffer,
                    self.drive,
                    self.spin,
                    self.x_ball,
                    self.y_ball,
                    self.vx_ball,
                    self.vy_ball,
                    self.ax_ball,
                    self.ay_ball,
                    self.x_bot,
                    self.y_bot,
                    self.th_bot,
                    self.vx_bot,
                    self.vy_bot,
                    self.omega_bot,
                    self.ax_bot,
                    self.ay_bot,
                    self.alpha_bot)

        calculate_reward()
        self.convert_detectors_to_sensors()
        return self.sensors, self.reward

    def plot_robot(self, ax, x_bot, y_bot, th_bot, alpha=1., dzorder=0): 
        """
        Plot the robot and sensors in the current figure and axes
        """
        # Rixel color is gray (3b3b3b)
        # eye color is light blue (c1e0ec)
        robot_color = (59./255.,59./255.,59./255.)
        eye_color = (193./255.,224./255.,236./255.)
        ax.add_patch(patches.Circle((x_bot, y_bot), 
                                     self.r_bot, color=robot_color, 
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_bot, y_bot), 
                                     self.r_bot, 
                                     color=tools.copper_shadow, 
                                     linewidth=2., fill=False,
                                     alpha=alpha, zorder=-dzorder))
        # robot eyes
        x_left = (x_bot + self.r_bot * .7 * np.cos(th_bot) + 
                  self.r_bot * .25 * np.cos(th_bot + np.pi/2.))
        y_left = (y_bot + self.r_bot * .7 * np.sin(th_bot) + 
                  self.r_bot * .25 * np.sin(th_bot + np.pi/2.))
        x_right = (x_bot + self.r_bot * .7 * np.cos(th_bot) + 
                   self.r_bot * .25 * np.cos(th_bot - np.pi/2.))
        y_right = (y_bot + self.r_bot * .7 * np.sin(th_bot) + 
                  self.r_bot * .25 * np.sin(th_bot - np.pi/2.))
        # pupil locations
        xp_left = (x_bot + self.r_bot * .725 * np.cos(th_bot) + 
                  self.r_bot * .248 * np.cos(th_bot + np.pi/2.))
        yp_left = (y_bot + self.r_bot * .725 * np.sin(th_bot) + 
                  self.r_bot * .248 * np.sin(th_bot + np.pi/2.))
        xp_right = (x_bot + self.r_bot * .725 * np.cos(th_bot) + 
                   self.r_bot * .248 * np.cos(th_bot - np.pi/2.))
        yp_right = (y_bot + self.r_bot * .725 * np.sin(th_bot) + 
                  self.r_bot * .248 * np.sin(th_bot - np.pi/2.))
        ax.add_patch(patches.Circle((x_left, y_left), 
                                     self.r_bot * .1, 
                                     color=eye_color,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((xp_left, yp_left), 
                                     self.r_bot * .06, 
                                     color=tools.copper_shadow,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_left, y_left), 
                                     self.r_bot * .1, 
                                     color=tools.copper_shadow, 
                                     linewidth=1., fill=False,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=eye_color,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((xp_right, yp_right), 
                                     self.r_bot * .06, 
                                     color=tools.copper_shadow,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=tools.copper_shadow, 
                                     linewidth=1., fill=False,
                                     alpha=alpha, zorder=-dzorder))

    def plot_sensors(self, ax, x_bot, y_bot, th_bot):
        """
        Visually represent what the sensors are detecting around the robot
        """
        # Show sensors visually
        # ball range sensor
        max_alpha = .3
        for i_vision_range in np.nonzero(self.v_range)[0]:
            magnitude = self.v_range[i_vision_range]
            i_range = np.minimum(i_vision_range + 1, self.n_ball_range - 1)  
            range_radius = self.r_bot + self.ball_range_bins[i_range]
            alpha = np.minimum(1., magnitude * max_alpha) 
            ax.add_patch(patches.Circle((x_bot, y_bot), range_radius, 
                                         color=tools.oxide, 
                                         alpha=alpha, 
                                         linewidth=10., fill=False))

        # ball heading sensors
        for i_vision_heading in np.nonzero(self.v_heading)[0]:
            magnitude = self.v_heading[i_vision_heading]
            heading_sensor_radius = self.width + self.depth
            d_heading = 2. * np.pi / self.n_ball_heading
            heading_sensor_angle_1 = th_bot - i_vision_heading * d_heading
            heading_sensor_angle_2 = th_bot - (i_vision_heading + 1) * d_heading
            x = x_bot + np.array([
                    0., 
                    np.cos(heading_sensor_angle_1) * heading_sensor_radius,
                    np.cos(heading_sensor_angle_2) * heading_sensor_radius,
                    0.])
            y = y_bot + np.array([
                    0., 
                    np.sin(heading_sensor_angle_1) * heading_sensor_radius,
                    np.sin(heading_sensor_angle_2) * heading_sensor_radius,
                    0.])
            ax.fill(x, y, color=tools.oxide, 
                    alpha=np.minimum(1., magnitude*max_alpha),
                    zorder=-1)
                                         

        # proximity sensors
        for (i_prox, prox_theta) in enumerate(
                    np.arange(0., 2 * np.pi, 
                              2 * np.pi / self.n_prox_heading)):
            for i_range in np.where(self.prox[i_prox,:] > 0)[0]:
                magnitude = self.prox[i_prox, i_range]
                i_prox_range = np.minimum(i_range, self.n_prox_range - 1)
                prox_range = self.r_bot + self.prox_range_bins[i_prox_range]
                prox_angle = th_bot - prox_theta
                x = x_bot + np.cos(prox_angle) * prox_range
                y = y_bot + np.sin(prox_angle) * prox_range
                prox_sensor_radius = self.r_bot / 10. 
                alpha = np.minimum(1., magnitude * max_alpha) 
                ax.add_patch(patches.Circle((x, y), prox_sensor_radius, 
                                            color=tools.copper, 
                                            alpha=alpha,
                                            linewidth=0., fill=True))
                plt.plot([x_bot, x], [y_bot, y],
                         color=tools.copper, linewidth=.5, 
                         alpha=alpha,
                         zorder=-10)
        
        # bump sensors
        max_alpha = .8
        for (i_bump, bump_theta) in enumerate(
                    np.arange(0., 2 * np.pi, 2 * np.pi / self.n_bump_heading)):
            bump_angle = th_bot - bump_theta
            x = x_bot + np.cos(bump_angle) * self.r_bot
            y = y_bot + np.sin(bump_angle) * self.r_bot
            for i_mag in np.where(self.bump[i_bump,:] > 0)[0]:
                magnitude = np.minimum(1., self.bump[i_bump, i_mag])

                bump_sensor_radius = ((self.r_bot * i_mag) / 
                                      (2. * self.n_bump_mag))
                bump_sensor_radius = np.maximum(0., bump_sensor_radius)
                alpha = np.minimum(1., magnitude * max_alpha) 
                ax.add_patch(patches.Circle(
                        (x, y), bump_sensor_radius, color=tools.copper_shadow, 
                        alpha=alpha,
                        linewidth=0., fill=True))
        
       
        # speed and acceleration sensors
        # TODO:
        scale = .1
        dx = self.vx_bot * scale
        dy = self.vy_bot * scale
        dth = self.omega_bot * scale
        self.plot_robot(ax, x_bot+dx, y_bot+dy, th_bot+dth, 
                        alpha=.3, dzorder=13)
        ddx = dx + self.ax_bot * scale ** 2
        ddy = dy + self.ay_bot * scale ** 2
        ddth = dth + self.alpha_bot * scale ** 2
        self.plot_robot(ax, x_bot+ddx, y_bot+ddy, th_bot+ddth, 
                        alpha=.15, dzorder=16)


    def render(self): 
        """ 
        Make a pretty picture of what's going on in the world 
        """ 
        fig = plt.figure(num=83)
        fig.clf()
        fig, ax = plt.subplots(num=83, figsize=(self.width, self.depth))

        # the walls
        plt.plot(np.array([0., self.width, self.width, 0., 0.]), 
                 np.array([0., 0., self.depth, self.depth, 0.]), 
                 linewidth=10, color=tools.copper_shadow)
        # the floor
        ax.fill([0., self.width, self.width, 0., 0.], 
                [0., 0., self.depth, self.depth, 0.], 
                color=tools.light_copper, zorder=-100)
        for x in np.arange(1., self.width):
            plt.plot(np.array([x, x]), np.array([0., self.depth]),
                     linewidth=2, color=tools.copper_highlight,
                     zorder=-99)
        for y in np.arange(1., self.depth):
            plt.plot(np.array([0., self.width]), np.array([y, y]),
                     linewidth=2, color=tools.copper_highlight,
                     zorder=-99)
        # the ball
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, color=tools.oxide))
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, 
                                     color=tools.copper_shadow, 
                                     linewidth=2., fill=False))

        self.plot_robot(ax, self.x_bot, self.y_bot, self.th_bot) 
        self.plot_sensors(ax, self.x_bot, self.y_bot, self.th_bot)
                
        plt.axis('equal')
        plt.axis('off')
        # Make sure the walls don't get clipped
        plt.ylim((-.1, self.depth + .1))
        plt.xlim((-.1, self.width + .1))
        fig.canvas.draw()
        # Save the image
        filename = ''.join([self.name, '_', str(self.frame_counter),'.png'])
        full_filename = os.path.join(self.frames_directory, filename)
        self.frame_counter += 1
        dpi = 80 # for a resolution of 720 lines
        #dpi = 120 # for a resolution of 1080 lines
        facecolor = fig.get_facecolor()
        plt.savefig(full_filename, format='png', dpi=dpi, 
                    facecolor=facecolor, edgecolor='none') 

    def visualize_world(self, brain):
        """ 
        Show what's going on in the world 
        """
        if (self.timestep % self.timesteps_per_frame) != 0:
            return
        timestr = tools.timestr(self.clock_time, s_per_step=1.)
        print ' '.join(['world running for', timestr])

        # Periodcally show the entire feature set 
        if self.plot_feature_set:
            (feature_set, 
             feature_activities) = brain.cortex.get_index_projections()
            num_blocks = len(feature_set)
            for block_index in range(num_blocks):
                for feature_index in range(len(feature_set[block_index])):
                    projection = feature_set[block_index][feature_index] 
                    # Convert projection to sensor activities
                    #print(' '.join(['block_index', str(block_index), 
                    #                'feature_index', str(feature_index)])) 
                    self.convert_sensors_to_detectors(projection)
                                        
                    fig = plt.figure(num=99)
                    fig.clf()
                    fig, ax = plt.subplots(num=99, figsize=(3 * self.width, 
                                                            3 * self.depth))
                    self.plot_robot(ax, 0., 0., np.pi/2)
                    self.plot_sensors(ax, 0., 0., np.pi/2)
                    
                    plt.axis('equal')
                    plt.axis('off')
                    plt.ylim((-self.depth, self.depth))
                    plt.xlim((-self.width, self.width))
                    fig.canvas.draw()

                    filename = '_'.join(('block', str(block_index).zfill(2),
                                         'feature',str(feature_index).zfill(4),
                                         self.name, 'world.png'))
                    full_filename = os.path.join(self.world_directory, 
                            'features', filename)
                    plt.title(filename)
                    plt.savefig(full_filename, format='png') 


