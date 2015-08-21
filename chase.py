"""
a task in which a bug-looking robot chases a ball

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

class World(BaseWorld):
    """ 
    ball-chasing bug robot world
    
    In this two-dimensional world the robot can spin and move 
    forward and backward. It can sense both how far away the 
    ball is and in which direction. It gets a small reward for being
    pointed toward the ball, a slightly larger reward for being nearer
    the ball, and a large reward for 'catching' it--touching it with its
    nose.

    The physics in this world are intended to be those of the physical
    world, at least to the depth of an introductoy mechanics class.
    """
    def __init__(self, lifespan=None):
        """ 
        Set up the world 
        """
        BaseWorld.__init__(self, lifespan)
        timesteps_per_second = 10.
        self.clockticks_per_timestep = int(1000. / timesteps_per_second)
        self.plot_feature_set = False
        filming = False
        if filming:
            # Render the world for creating a 30 frame-per-second video
            self.timesteps_per_frame = timesteps_per_second / 30. 
            # Don't plot features while filming
            self.plot_feature_set = False
        else:
            self.timesteps_per_frame = 10000 
        self.clockticks_per_frame = int(self.clockticks_per_timestep * 
                                        self.timesteps_per_frame)
        #self.name = 'chase'
        self.name = 'chase_explore_25'
        self.name_long = 'ball chasing world'
        print "Entering", self.name_long
        self.n_bump_heading = 1#9
        self.n_bump_mag = 2#5
        self.n_ball_heading = 19
        self.n_prox_heading = 2#11
        self.n_ball_range = 15
        self.n_prox_range = 1#7
        self.n_prox = self.n_prox_heading * self.n_prox_range
        self.n_bump = self.n_bump_heading * self.n_bump_mag
        # Must be odd
        self.n_vel_per_axis = 9 
        # Must be odd
        self.n_acc_per_axis = 5 
        self.n_vel = 3 * self.n_vel_per_axis
        self.n_acc = 3 * self.n_acc_per_axis

        self._initialize_world()
        self.num_sensors = (self.n_ball_range + self.n_ball_heading + 
                            self.n_prox + self.n_bump + 
                            self.n_vel + self.n_acc)
        self.num_actions = 20
        self.action = np.zeros((self.num_actions, 1))
        self.CATCH_REWARD = .8
        self.TOUCH_REWARD = 0.#.01
        self.BUMP_PENALTY = 0.#.1
        self.EFFORT_PENALTY = 0.#1e-4
        #self.state_history = []
        self.world_directory = 'becca_world_chase_ball'
        self.log_directory = os.path.join(self.world_directory, 'log')
        self.frames_directory = os.path.join(self.world_directory, 'frames') 
        self.frame_counter = 10000

    def _initialize_world(self):
        """
        Set up the phyics of the simulation
        """
        self.clock_tick = 0.
        self.clock_time = 0.
        self.dt = .001 # seconds per clock tick

        # wall parameters
        self.width = 5.#9. # meters
        self.depth = 5.#16. # meters
        self.k_wall =  3000. # Newtons / meter

        # ball parameters
        self.r_ball = .4 # meters
        self.k_ball = 3000. # Newtons / meter
        self.c_ball = 1. # Newton-seconds / meter
        self.cc_ball = 1. # Newton
        self.m_ball = 1. # kilogram
        self.x_ball = 4. # meters
        self.y_ball = 4. # meters
        self.vx_ball = 0. # meters / second
        self.vy_ball = 0. # meters / second
        self.ax_ball = 0. # meters**2 / second
        self.ay_ball = 0. # meters**2 / second

        # robot parameters
        self.r_bot = .8 # meters
        self.k_bot =  3000. # Newtons / meter
        self.c_bot = 10. # Newton-seconds / meter
        self.cc_bot = 1. # Newton
        self.d_bot = 10. # Newton-meter-seconds / radian
        self.dd_bot = 1. # Newton-meter
        self.m_bot = 5. # kilogram
        self.I_bot = 1. # kilogram-meters**2
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
        def build_bins_one_sided(n, max_edge, min_edge=0.):
            """
            Build an array of bin edges from (near) zero to some positive value
            
            The array of bin edges produced will be of the form
                [min_edge, a_1, a_2, a_3, ..., a_n-1]
            where a_i form a geometric sequence to max_edge
                a_i = max_edge * 2 ** i/n - 1 
            """
            bins = [min_edge]
            for i in range(n - 1):
                fraction = float(i + 1) / n
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
            bins = [-tools.BIG]
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

        self.drive_scale = 8.
        self.spin_scale = 8.

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
        print 'self.bump'
        print self.bump
        print 'self.v_heading'
        print self.v_heading
        print 'self.v_range'
        print self.v_range
        print 'self.prox'
        print self.prox
        print 'self.v_fwd_bot_sensor'
        print self.v_fwd_bot_sensor
        print 'self.v_lat_bot_sensor'
        print self.v_lat_bot_sensor
        print 'self.omega_bot_sensor'
        print self.omega_bot_sensor
        print 'self.a_fwd_bot_sensor'
        print self.a_fwd_bot_sensor
        print 'self.a_lat_bot_sensor'
        print self.a_lat_bot_sensor
        print 'self.alpha_bot_sensor'
        print self.alpha_bot_sensor

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
                        16 * self.action[19])[0]

        def calculate_reward():
            """ 
            Assign reward based on accumulated target events over the 
            previous time step
            """
            self.reward = 0.
            self.reward += self.n_catch * self.CATCH_REWARD
            self.reward += (2 * np.sum(self.bump[:,0]) + 
                            np.sum(self.bump[:,1])) * self.TOUCH_REWARD
            self.reward -= (2 * np.sum(self.bump[:,-1]) + 
                            np.sum(self.bump[:,-2])) * self.BUMP_PENALTY
            self.reward -= self.effort * self.EFFORT_PENALTY

            self.n_catch = 0.
        
        # Use the convenient internal methods defined above to 
        # step the world forward at a high leve of abstraction.
        self.timestep += 1 
        convert_actions_to_drives(action)
        for _ in range(self.clockticks_per_timestep):
            self.clock_step()
        calculate_reward()
        self.convert_detectors_to_sensors()
        return self.sensors, self.reward

    def clock_step(self):
        """
        Advance the phyisical simulation of the world by one clock tick.
        This is at a much finer temporal granularity. 
        """
        self.clock_tick += 1
        self.clock_time = self.clock_tick * self.dt
        
        def sector_index(n_sectors, theta):
            """ 
            For sector-based detectors, find the sector based on angle 
            """
            theta = np.mod(theta, 2 * np.pi)
            return int(np.floor(n_sectors * theta / (2 * np.pi)))

        def find_wall_range(theta):
            """
            Calculate the range to the nearest wall
            """
            wall_range = 1e10
            # Find distance to West wall
            range_west = self.x_bot / (np.cos(np.pi - theta) + 1e-6)
            if range_west < 0.:
                range_west = 1e10
            range_west -= self.r_bot
            wall_range = np.minimum(wall_range, range_west)
            # Find distance to East wall
            range_east = (self.width - self.x_bot) / (np.cos(theta) + 1e-6) 
            if range_east < 0.:
                range_east = 1e10
            range_east -= self.r_bot
            wall_range = np.minimum(wall_range, range_east)
            # Find distance to South wall
            range_south = self.y_bot / (np.sin(-theta) + 1e-6)
            if range_south < 0.:
                range_south = 1e10
            range_south -= self.r_bot
            wall_range = np.minimum(wall_range, range_south)
            # Find distance to North wall
            range_north = (self.depth - self.y_bot) / (np.sin(theta) + 1e-6)
            if range_north < 0.:
                range_north = 1e10
            range_north -= self.r_bot
            wall_range = np.minimum(wall_range, range_north)
            return wall_range

        # Add new force profiles to the buffers
        if np.abs(self.drive) > 0.:
            drive_x = self.drive * np.cos(self.th_bot)
            drive_y = self.drive * np.sin(self.th_bot)
            for proto_index in np.arange(self.proto_force.size):
                tick  = self.clock_tick + proto_index
                buffer_index = np.mod(tick, self.f_x_buffer.size)
                self.f_x_buffer[buffer_index] += (
                    self.proto_force[proto_index] * drive_x * self.drive_scale)
                self.f_y_buffer[buffer_index] += (
                    self.proto_force[proto_index] * drive_y * self.drive_scale)
            self.drive = 0.

        if np.abs(self.spin) > 0.:
            for proto_index in np.arange(self.proto_force.size):
                tick  = self.clock_tick + proto_index
                buffer_index = np.mod(tick, self.tau_buffer.size)
                self.tau_buffer[buffer_index] += (
                    self.proto_force[proto_index] * self.spin * self.spin_scale)
            self.spin = 0.

        # grab next value from force buffers
        buffer_index = np.mod(self.clock_tick, self.f_x_buffer.size)
        f_x_drive = self.f_x_buffer[buffer_index]
        f_y_drive = self.f_y_buffer[buffer_index]
        tau_drive = self.tau_buffer[buffer_index]
        self.f_x_buffer[buffer_index] = 0.
        self.f_y_buffer[buffer_index] = 0.
        self.tau_buffer[buffer_index] = 0.

        # contact between the robot and the ball
        delta_bot_ball = (self.r_ball + self.r_bot -
                          ((self.x_ball - self.x_bot) ** 2 + 
                           (self.y_ball - self.y_bot) ** 2) ** .5) 
        dist_bot_ball = -delta_bot_ball
        delta_bot_ball = np.maximum(delta_bot_ball, 0.)
        th_bot_ball = np.arctan2(self.y_ball - self.y_bot,
                                 self.x_ball - self.x_bot)
        k_bot_ball = ((self.k_bot * self.k_ball) / 
                      (self.k_bot + self.k_ball))
        # ball range detection
        self.i_vision_range = np.where(
                dist_bot_ball > self.ball_range_bins)[0][-1] 
        #self.range[self.i_vision_range] += 1.
        # ball heading detection
        # the angle of contact relative to the robot's angle
        th_bot_ball_rel = np.mod(self.th_bot - th_bot_ball, 2. * np.pi)
        self.i_vision_heading = sector_index(
                self.n_ball_heading, th_bot_ball_rel)
        #self.heading[self.i_vision_heading] += 1. 
        #self.vision = np.zeros(self.vision.shape)
        #self.vision[self.i_vision_heading, self.i_vision_range] = 1.
        self.v_heading[self.i_vision_heading] += (1. /
                                                  self.clockticks_per_timestep)
        self.v_range[self.i_vision_range] += 1. / self.clockticks_per_timestep
        
        # ball bump detection
        if delta_bot_ball > 0.:
            i_bump_heading = sector_index(self.n_bump_heading, th_bot_ball_rel)
            i_bump_mag  = np.where(delta_bot_ball > self.bump_mag_bins)[0][-1] 
            self.bump[i_bump_heading, i_bump_mag] += (
                    1. / self.clockticks_per_timestep)

        # calculate the forces on the ball
        # wall contact
        delta_ball_N = self.y_ball + self.r_ball - self.depth
        delta_ball_N = np.maximum(delta_ball_N, 0.)
        delta_ball_S = self.r_ball - self.y_ball
        delta_ball_S = np.maximum(delta_ball_S, 0.)
        delta_ball_E = self.x_ball + self.r_ball - self.width
        delta_ball_E = np.maximum(delta_ball_E, 0.)
        delta_ball_W = self.r_ball - self.x_ball
        delta_ball_W = np.maximum(delta_ball_W, 0.)
        k_wall_ball = ((self.k_wall * self.k_ball) / 
                       (self.k_wall + self.k_ball))
        f_ball_N_y = -delta_ball_N * k_wall_ball 
        f_ball_S_y =  delta_ball_S * k_wall_ball 
        f_ball_E_x = -delta_ball_E * k_wall_ball 
        f_ball_W_x =  delta_ball_W * k_wall_ball 
        # contact with the robot
        f_ball_bot_x = delta_bot_ball * k_bot_ball * np.cos(th_bot_ball)
        f_ball_bot_y = delta_bot_ball * k_bot_ball * np.sin(th_bot_ball)
        # friction and damping
        f_ball_vx = (-self.vx_ball * self.c_ball
                     -np.sign(self.vx_ball) * self.cc_ball)
        f_ball_vy = (-self.vy_ball * self.c_ball
                     -np.sign(self.vy_ball) * self.cc_ball)

        # wall bump detection
        # Calculate the angle of contact relative to the robot's angle
        # for each of the walls. Assign the contact to the appropriate
        # bump sensor.
        delta_bot_N = self.y_bot + self.r_bot - self.depth
        delta_bot_N = np.maximum(delta_bot_N, 0.)
        if delta_bot_N > 0.:
            th_bot_N_rel = np.mod(self.th_bot - np.pi / 2., 2. * np.pi)
            #i_bump = sector_index(self.n_bump_heading, th_bot_N_rel)
            #self.bump[i_bump] += delta_bot_N 

            i_bump_heading = sector_index(self.n_bump_heading, th_bot_N_rel)
            i_bump_mag  = np.where(delta_bot_N > self.bump_mag_bins)[0][-1] 
            self.bump[i_bump_heading, i_bump_mag] += (
                    1. / self.clockticks_per_timestep)

        delta_bot_S = self.r_bot - self.y_bot
        delta_bot_S = np.maximum(delta_bot_S, 0.)
        if delta_bot_S > 0.:
            th_bot_S_rel = np.mod(self.th_bot + np.pi / 2., 2. * np.pi)
            #i_bump = sector_index(self.n_bump_heading, th_bot_S_rel)
            #self.bump[i_bump] += delta_bot_S
            i_bump_heading = sector_index(self.n_bump_heading, th_bot_S_rel)
            i_bump_mag  = np.where(delta_bot_S > self.bump_mag_bins)[0][-1] 
            self.bump[i_bump_heading, i_bump_mag] += (
                    1. / self.clockticks_per_timestep)

        delta_bot_E = self.x_bot + self.r_bot - self.width
        delta_bot_E = np.maximum(delta_bot_E, 0.)
        if delta_bot_E > 0.:
            th_bot_E_rel = np.mod(self.th_bot, 2. * np.pi)
            #i_bump = sector_index(self.n_bump_heading, th_bot_E_rel)
            #self.bump[i_bump] += delta_bot_E
            i_bump_heading = sector_index(self.n_bump_heading, th_bot_E_rel)
            i_bump_mag  = np.where(delta_bot_E > self.bump_mag_bins)[0][-1] 
            self.bump[i_bump_heading, i_bump_mag] += (
                    1. / self.clockticks_per_timestep)

        delta_bot_W = self.r_bot - self.x_bot
        delta_bot_W = np.maximum(delta_bot_W, 0.)
        if delta_bot_W > 0.:
            th_bot_W_rel = np.mod(self.th_bot + np.pi, 2. * np.pi)
            #i_bump = sector_index(self.n_bump_heading, th_bot_W_rel)
            #self.bump[i_bump] += delta_bot_W
            i_bump_heading = sector_index(self.n_bump_heading, th_bot_W_rel)
            i_bump_mag  = np.where(delta_bot_W > self.bump_mag_bins)[0][-1] 
            self.bump[i_bump_heading, i_bump_mag] += (
                    1. / self.clockticks_per_timestep)

        # wall proximity detection
        # calculate the range detected by each proximity sensor
        #if self.include_prox:
        for (i_prox, prox_theta) in enumerate(
                    np.arange(0., 2 * np.pi, 
                              2 * np.pi / self.n_prox_heading)):

            wall_range = find_wall_range(self.th_bot - prox_theta)
            i_prox_range = np.where(wall_range > self.prox_range_bins)[0][-1]
            self.prox[i_prox, i_prox_range] += (
                    1. / self.clockticks_per_timestep)


        # calculated the forces on the bot
        # wall contact
        k_wall_bot = ((self.k_wall * self.k_bot) / 
                       (self.k_wall + self.k_bot))
        f_bot_N_y = -delta_bot_N * k_wall_bot 
        f_bot_S_y =  delta_bot_S * k_wall_bot 
        f_bot_E_x = -delta_bot_E * k_wall_bot 
        f_bot_W_x =  delta_bot_W * k_wall_bot 
        # contact with the robot
        f_bot_ball_x = -delta_bot_ball * k_bot_ball * np.cos(th_bot_ball)
        f_bot_ball_y = -delta_bot_ball * k_bot_ball * np.sin(th_bot_ball)
        # friction and damping, both proportional and Coulomb
        #f_bot_vx = -self.vx_bot * self.c_bot
        #f_bot_vy = -self.vy_bot * self.c_bot
        #tau_bot_omega = -self.omega_bot * self.d_bot
        f_bot_vx = (-self.vx_bot * self.c_bot 
                    -np.sign(self.vx_bot) * self.cc_bot)
        f_bot_vy = (-self.vy_bot * self.c_bot
                    -np.sign(self.vx_bot) * self.cc_bot)
        tau_bot_omega = (-self.omega_bot * self.d_bot
                         -np.sign(self.omega_bot) * self.dd_bot)

        # calculate total external forces
        f_ball_x = f_ball_E_x + f_ball_W_x + f_ball_bot_x + f_ball_vx
        f_ball_y = f_ball_N_y + f_ball_S_y + f_ball_bot_y + f_ball_vy
        f_bot_x = f_bot_E_x + f_bot_W_x + f_bot_ball_x + f_bot_vx + f_x_drive
        f_bot_y = f_bot_N_y + f_bot_S_y + f_bot_ball_y + f_bot_vy + f_y_drive
        tau_bot = tau_bot_omega + tau_drive

        # use forces to update accelerations, velocities, and positions
        # ball
        self.ax_ball = f_ball_x / self.m_ball
        self.ay_ball = f_ball_y / self.m_ball
        self.vx_ball += self.ax_ball * self.dt
        self.vy_ball += self.ay_ball * self.dt
        self.x_ball += self.vx_ball * self.dt
        self.y_ball += self.vy_ball * self.dt
        # robot
        self.ax_bot = f_bot_x / self.m_bot
        self.ay_bot = f_bot_y / self.m_bot
        self.alpha_bot = tau_bot / self.I_bot
        self.vx_bot += self.ax_bot * self.dt
        self.vy_bot += self.ay_bot * self.dt
        self.omega_bot+= self.alpha_bot * self.dt
        self.x_bot += self.vx_bot * self.dt
        self.y_bot += self.vy_bot * self.dt
        self.th_bot += self.omega_bot * self.dt
        self.th_bot = np.mod(self.th_bot, 2 * np.pi)
        
        # The robot's speed in the direction of its nose
        self.v_fwd_bot = (self.vx_bot * np.cos(self.th_bot) +
                          self.vy_bot * np.sin(self.th_bot) )
        # The robot's sideways speed (right is positive)
        self.v_lat_bot = (self.vx_bot * np.sin(self.th_bot) -
                          self.vy_bot * np.cos(self.th_bot))
        # The robot's speed in the direction of its nose
        self.a_fwd_bot = (self.ax_bot * np.cos(self.th_bot) +
                          self.ay_bot * np.sin(self.th_bot) )
        # The robot's sideways speed (right is positive)
        self.a_lat_bot = (self.ax_bot * np.sin(self.th_bot) -
                          self.ay_bot * np.cos(self.th_bot))

        i_v_fwd = np.where(self.v_fwd_bot > self.v_fwd_bins)[0][-1]
        self.v_fwd_bot_sensor[i_v_fwd] += 1. / self.clockticks_per_timestep
        i_v_lat = np.where(self.v_lat_bot > self.v_lat_bins)[0][-1]
        self.v_lat_bot_sensor[i_v_lat] += 1. / self.clockticks_per_timestep
        i_omega = np.where(self.omega_bot > self.omega_bins)[0][-1]
        self.omega_bot_sensor[i_omega] += 1. / self.clockticks_per_timestep
        i_a_fwd = np.where(self.a_fwd_bot > self.a_fwd_bins)[0][-1]
        self.a_fwd_bot_sensor[i_a_fwd] += 1. / self.clockticks_per_timestep
        i_a_lat = np.where(self.a_lat_bot > self.a_lat_bins)[0][-1]
        self.a_lat_bot_sensor[i_a_lat] += 1. / self.clockticks_per_timestep
        i_alpha = np.where(self.alpha_bot > self.alpha_bins)[0][-1]
        self.alpha_bot_sensor[i_alpha] += 1. / self.clockticks_per_timestep

        # check whether the bot caught the ball
        caught = False
        if ((self.i_vision_heading == 0) | 
            (self.i_vision_heading == self.n_ball_heading - 1)):
            #self.n_see += 1.
            if self.i_vision_range == 0:
                caught = True

        #self.n_reach += float(self.n_range - 1 - self.i_vision_range)

        if caught:
            self.n_catch += 1.
            # when caught, the ball jumps to a new location
            good_location = False
            while not good_location:
                self.x_ball = self.r_ball + np.random.random_sample() * (
                        self.width - 2 * self.r_ball)
                self.y_ball = self.r_ball + np.random.random_sample() * (
                        self.depth - 2 * self.r_ball)
                self.vx_ball = np.random.normal()
                self.vy_ball = np.random.normal()
                # check that the ball doesn't splinch the robot
                delta_bot_ball = (self.r_ball + self.r_bot -
                                  ((self.x_ball - self.x_bot) ** 2 + 
                                   (self.y_ball - self.y_bot) ** 2) ** .5) 
                if delta_bot_ball < 0.:
                    good_location = True
                    
        if (self.clock_tick % self.clockticks_per_frame) == 0:
            self.render()

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
                                     color=tools.COPPER_SHADOW, 
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
                                     color=tools.COPPER_SHADOW,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_left, y_left), 
                                     self.r_bot * .1, 
                                     color=tools.COPPER_SHADOW, 
                                     linewidth=1., fill=False,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=eye_color,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((xp_right, yp_right), 
                                     self.r_bot * .06, 
                                     color=tools.COPPER_SHADOW,
                                     alpha=alpha, zorder=-dzorder))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=tools.COPPER_SHADOW, 
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
            ax.add_patch(patches.Circle((x_bot, y_bot), range_radius, 
                                         color=tools.OXIDE, 
                                         alpha=magnitude*max_alpha, 
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
            ax.fill(x, y, color=tools.OXIDE, 
                    alpha=magnitude*max_alpha, 
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
                ax.add_patch(patches.Circle((x, y), prox_sensor_radius, 
                                            color=tools.COPPER, 
                                            alpha=magnitude*max_alpha,
                                            linewidth=0., fill=True))
                plt.plot([x_bot, x], [y_bot, y],
                         color=tools.COPPER, linewidth=.5, 
                         alpha=magnitude*max_alpha,
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
                ax.add_patch(patches.Circle(
                        (x, y), bump_sensor_radius, color=tools.COPPER_SHADOW, 
                        alpha=max_alpha*magnitude,
                        linewidth=0., fill=True))
       
        # speed and acceleration sensors
        # TODO:
        dx = .3
        dy = .2
        dth = .5
        self.plot_robot(ax, x_bot+dx, y_bot+dy, th_bot+dth, 
                        alpha=.3, dzorder=13)
        ddx = .5
        ddy = .4
        ddth = .8
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
                 linewidth=10, color=tools.COPPER_SHADOW)
        # the floor
        ax.fill([0., self.width, self.width, 0., 0.], 
                [0., 0., self.depth, self.depth, 0.], 
                color=tools.LIGHT_COPPER, zorder=-100)
        for x in np.arange(1., self.width):
            plt.plot(np.array([x, x]), np.array([0., self.depth]),
                     linewidth=2, color=tools.COPPER_HIGHLIGHT,
                     zorder=-99)
        for y in np.arange(1., self.depth):
            plt.plot(np.array([0., self.width]), np.array([y, y]),
                     linewidth=2, color=tools.COPPER_HIGHLIGHT,
                     zorder=-99)
        # the ball
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, color=tools.OXIDE))
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, 
                                     color=tools.COPPER_SHADOW, 
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
        plt.savefig(full_filename, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none') 

    def visualize(self, agent=None):
        """ 
        Show what's going on in the world 
        """
        if (self.timestep % self.timesteps_per_frame) != 0:
            return
        timestr = tools.timestr(self.clock_time, s_per_step=1.)
        print ' '.join(['world running for', timestr])

        # Periodcally show the entire feature set 
        if self.plot_feature_set:
            (feature_set, feature_activities) = agent.get_index_projections()
            num_blocks = len(feature_set)
            for block_index in range(num_blocks):
                for feature_index in range(len(feature_set[block_index])):
                    projection = feature_set[block_index][feature_index] 
                    # Convert projection to sensor activities
                    #print 'block_index', block_index, 'feature_index', feature_index 
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


