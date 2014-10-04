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
    
    """
    def __init__(self, lifespan=None):
        """ Set up the world """
        BaseWorld.__init__(self, lifespan)
        self.CLOCKS_PER_LOOP = int(1000. / 4.)
        self.TIMESTEPS_PER_FRAME = 100
        self.CLOCKS_PER_FRAME = 33.
        self.name = 'chase'
        self.name_long = 'ball chasing world'
        print "Entering", self.name_long
        self._initialize_world()
        self.num_sensors = self.n_bump + self.n_range + self.n_heading
        self.num_actions = 18
        self.action = np.zeros((self.num_actions,1))
        self.CATCH_REWARD = 100.
        self.bump_penalty = True
        self.name = 'chase_world'
        print "Entering", self.name
        self.world_directory = 'becca_world_chase_ball'
        self.log_directory = os.path.join(self.world_directory, 'log')
        self.frames_directory = os.path.join(self.world_directory, 'frames') 
        self.frame_counter = 10000
        self.frames_per_step = 1
        self.frames_per_sec = 30.

    def _initialize_world(self):
        self.clock_tick = 0.
        self.clock_time = 0.
        self.dt = .001 # seconds per clock tick

        # wall parameters
        self.width = 5. # meters
        self.depth = 5. # meters
        self.k_wall =  3000. # Newtons / meter

        # ball parameters
        self.r_ball = .4 # meters
        self.k_ball = 3000. # Newtons / meter
        self.c_ball = 1. # Newton-seconds / meter
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
        self.d_bot = 10. # Newton-meter-seconds / radian
        self.m_bot = 5. # kilogram
        self.I_bot = 1. # kilogram-meters**2
        self.x_bot = 2. # meters
        self.y_bot = 2. # meters
        self.th_bot = np.pi/4. # radians
        self.vx_bot = 0. # meters / second
        self.vy_bot = 0. # meters / second
        self.omega_bot = 0. # radians / second
        self.ax_bot = 0. # meters**2 / second
        self.ay_bot = 0. # meters**2 / second
        self.alpha_bot = 0. # radians / second**2

        # detector parameters
        self.n_bump = 12
        self.bump = np.zeros(self.n_bump)
        self.n_heading = 12
        self.heading = np.zeros(self.n_heading)
        self.n_range = 7 
        self.range = np.zeros(self.n_range)
        self.range_bins = self.r_bot * np.array([-1., .1, .2, .4, .8, 1.6, 3.2])
        self.n_catch = 0.

        # create a prototype force profile
        # It is a triangular profile, ramping linearly from 0 to peak over its
        # first half, than ramping linearly back down to 0
        duration = .25 # seconds
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

        self.drive_scale = 20.
        self.spin_scale = 20.
        #self.visualize()
        return

    def step(self, action): 
        """ Take one time step through the world """

        def convert_actions_to_drives(action):
            self.action = action
            #for index in np.arange(self.action.size):
            #    if np.random.random_sample() < .05:
            #        self.action[index] = 1.
                    
            #print 'action', self.action.ravel()
            self.timestep += 1 
            # Find the drive magnitude
            self.drive = (
                     self.action[0] + 
                 2 * self.action[1] + 
                 3 * self.action[2] + 
                 4 * self.action[3] - 
                     self.action[4] - 
                 2 * self.action[5] - 
                 3 * self.action[6] - 
                 4 * self.action[7])
            # Find the spin magnitude
            self.spin = (
                    self.action[8] + 
                2 * self.action[9] + 
                3 * self.action[10] + 
                4 * self.action[11] - 
                    self.action[12] - 
                2 * self.action[13] - 
                3 * self.action[14] - 
                4 * self.action[15])
            return

        def calculate_reward():
            """ Assign reward based on the current state """
            self.reward = self.n_catch * self.CATCH_REWARD
            if self.bump_penalty:
                self.reward -= np.sum(self.bump)
            self.n_catch = 0.
            return 
        
        def convert_detectors_to_sensors():
            self.sensors = np.zeros(self.num_sensors)
            self.sensors[:self.n_bump] = self.bump
            self.sensors[self.n_bump:self.n_bump + self.n_range] = self.range
            self.sensors[self.n_bump + self.n_range:
                    self.n_bump + self.n_range + self.n_heading] = self.heading
            self.bump = np.zeros(self.n_bump)
            self.range = np.zeros(self.n_range)
            self.heading = np.zeros(self.n_heading)
            return
            
        convert_actions_to_drives(action)
        for _ in range(self.CLOCKS_PER_LOOP):
            self.clock_step()
        calculate_reward()
        convert_detectors_to_sensors()
        return self.sensors, self.reward

    def clock_step(self):
        self.clock_tick += 1
        self.clock_time = self.clock_tick * self.dt
        
        def sector_index(n_sectors, theta):
            """ For sector-based detectors, find the sector based on angle """
            theta = np.mod(theta, 2 * np.pi)
            return int(np.floor(n_sectors * theta / (2 * np.pi)))

        # add new force profiles to the buffers
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

        #print 'x', f_x_drive, 'y', f_y_drive, 'th', tau_drive 
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
        # TODO: limit vision to field of view
        # ball range detection
        i_range = np.where(dist_bot_ball > self.range_bins)[0][-1] 
        self.range[i_range] += 1.
        # ball heading detection
        # the angle of contact relative to the robot's angle
        th_bot_ball_rel = np.mod(self.th_bot - th_bot_ball, 2. * np.pi)
        i_heading = sector_index(self.n_heading, th_bot_ball_rel)
        self.heading[i_heading] += 1. 
        # ball bump detection
        if delta_bot_ball > 0.:
            i_bump = sector_index(self.n_bump, th_bot_ball_rel)
            self.bump[i_bump] += delta_bot_ball 

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
        f_ball_vx = -self.vx_ball * self.c_ball
        f_ball_vy = -self.vy_ball * self.c_ball

        # wall bump detection
        # calculate the angle of contact relative to the robot's angle
        delta_bot_N = self.y_bot + self.r_bot - self.depth
        delta_bot_N = np.maximum(delta_bot_N, 0.)
        if delta_bot_N > 0.:
            th_bot_N_rel = np.mod(self.th_bot - np.pi / 2., 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_N_rel)
            self.bump[i_bump] += delta_bot_N 
        delta_bot_S = self.r_bot - self.y_bot
        delta_bot_S = np.maximum(delta_bot_S, 0.)
        if delta_bot_S > 0.:
            th_bot_S_rel = np.mod(self.th_bot + np.pi / 2., 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_S_rel)
            self.bump[i_bump] += delta_bot_S
        delta_bot_E = self.x_bot + self.r_bot - self.width
        delta_bot_E = np.maximum(delta_bot_E, 0.)
        if delta_bot_E > 0.:
            th_bot_E_rel = np.mod(self.th_bot, 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_E_rel)
            self.bump[i_bump] += delta_bot_E
        delta_bot_W = self.r_bot - self.x_bot
        delta_bot_W = np.maximum(delta_bot_W, 0.)
        if delta_bot_W > 0.:
            th_bot_W_rel = np.mod(self.th_bot + np.pi, 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_W_rel)
            self.bump[i_bump] += delta_bot_W

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
        # friction and damping
        f_bot_vx = -self.vx_bot * self.c_bot
        f_bot_vy = -self.vy_bot * self.c_bot
        tau_bot_omega = -self.omega_bot * self.d_bot

        # calculate total external forces
        f_ball_x = f_ball_E_x + f_ball_W_x + f_ball_bot_x + f_ball_vx
        f_ball_y = f_ball_N_y + f_ball_S_y + f_ball_bot_y + f_ball_vy
        f_bot_x = f_bot_E_x + f_bot_W_x + f_bot_ball_x + f_bot_vx + f_x_drive
        f_bot_y = f_bot_N_y + f_bot_S_y + f_bot_ball_y + f_bot_vy + f_y_drive
        tau_bot = tau_bot_omega + tau_drive

        # use forces to update accerations, velocities, and positions
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

        # check whether the bot caught the ball
        caught = False
        if i_range == 0:
            if ((i_heading == 0) | (i_heading == self.n_heading - 1)):
                caught = True
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
                    
        if (self.clock_tick % self.CLOCKS_PER_FRAME) == 0:
            self.render()
        #print
        #print 'bump', self.bump.ravel()
        #print 'range', self.range.ravel()
        #print 'heading', self.heading.ravel()
        return
    
    def render(self): 
    
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
                color=tools.LIGHT_COPPER, zorder=-10)
        for x in np.arange(1., self.width):
            plt.plot(np.array([x, x]), np.array([0., self.depth]),
                     linewidth=2, color=tools.COPPER_HIGHLIGHT,
                     zorder=-9)
        for y in np.arange(1., self.depth):
            plt.plot(np.array([0., self.width]), np.array([y, y]),
                     linewidth=2, color=tools.COPPER_HIGHLIGHT,
                     zorder=-9)
        # the ball
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, color=tools.OXIDE))
        ax.add_patch(patches.Circle((self.x_ball, self. y_ball), 
                                     self.r_ball, 
                                     color=tools.COPPER_SHADOW, 
                                     linewidth=2., fill=False))

        # the robot
        ax.add_patch(patches.Circle((self.x_bot, self. y_bot), 
                                     self.r_bot, color=tools.COPPER))
        ax.add_patch(patches.Circle((self.x_bot, self. y_bot), 
                                     self.r_bot, 
                                     color=tools.COPPER_SHADOW, 
                                     linewidth=2., fill=False))
        # robot eyes
        x_left = (self.x_bot + self.r_bot * .7 * np.cos(self.th_bot) + 
                  self.r_bot * .25 * np.cos(self.th_bot + np.pi/2.))
        y_left = (self.y_bot + self.r_bot * .7 * np.sin(self.th_bot) + 
                  self.r_bot * .25 * np.sin(self.th_bot + np.pi/2.))
        x_right = (self.x_bot + self.r_bot * .7 * np.cos(self.th_bot) + 
                   self.r_bot * .25 * np.cos(self.th_bot - np.pi/2.))
        y_right = (self.y_bot + self.r_bot * .7 * np.sin(self.th_bot) + 
                  self.r_bot * .25 * np.sin(self.th_bot - np.pi/2.))
        ax.add_patch(patches.Circle((x_left, y_left), 
                                     self.r_bot * .1, 
                                     color=tools.DARK_COPPER))
        ax.add_patch(patches.Circle((x_left, y_left), 
                                     self.r_bot * .1, 
                                     color=tools.COPPER_SHADOW, 
                                     linewidth=1., fill=False))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=tools.DARK_COPPER))
        ax.add_patch(patches.Circle((x_right, y_right), 
                                     self.r_bot * .1, 
                                     color=tools.COPPER_SHADOW, 
                                     linewidth=1., fill=False))

        plt.axis('equal')
        plt.axis('off')
        # make sure the walls don't get clipped
        plt.ylim((-.1, self.depth + .1))
        plt.xlim((-.1, self.width + .1))
        #fig.show()
        fig.canvas.draw()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '.png'
        full_filename = os.path.join(self.frames_directory, filename)
        self.frame_counter += 1
        #dpi = 80 # for a resolution of 720 lines
        dpi = 120 # for a resolution of 1080 lines
        plt.savefig(full_filename, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none') 
        #plt.show()
        return

    def visualize(self, agent=None):
        """ Show what's going on in the world """
        if (self.timestep % self.TIMESTEPS_PER_FRAME) != 0:
            return
        print("world is %s seconds old " % self.clock_time)
        
        self.render()
        return
