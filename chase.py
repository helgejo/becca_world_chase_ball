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
        self.CLOCKTICKS_PER_TIMESTEP = int(1000. / 4.)
        self.TIMESTEPS_PER_FRAME = 100
        self.CLOCKTICKS_PER_FRAME = (self.CLOCKTICKS_PER_TIMESTEP * 
                                     self.TIMESTEPS_PER_FRAME)
        self.name = 'chase'
        #self.name = 'chase_no_bump'
        #self.name = 'chase_no_bump_large'
        #self.name = 'chase_small'
        #self.bump_penalty = True
        self.small = False
        self.name_long = 'ball chasing world'
        print "Entering", self.name_long
        self._initialize_world()
        #self.num_sensors = (self.n_bump + self.n_range + self.n_heading +
        #                    self.n_bump * self.n_range)
        self.num_sensors = (self.n_range * self.n_heading)
        #self.num_sensors = (self.n_bump + self.n_range * self.n_heading +
        #                    self.n_bump * self.n_range)
        self.num_actions = 16
        self.action = np.zeros((self.num_actions,1))
        self.CATCH_REWARD = .8
        self.SEE_REWARD = .01
        self.RANGE_REWARD = .002
        self.BUMP_PENALTY = 0.#.01
        self.state_history = []
        self.world_directory = 'becca_world_chase_ball'
        self.log_directory = os.path.join(self.world_directory, 'log')
        self.frames_directory = os.path.join(self.world_directory, 'frames') 
        self.frame_counter = 10000

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
        if self.small:
            self.n_bump = 6
        else:
            self.n_bump = 12
        self.bump = np.zeros(self.n_bump)
        self.n_heading = 12
        #self.heading = np.zeros(self.n_heading)
        if self.small:
            self.n_range = 3 
            self.range_bins = self.r_bot * np.array([-1., .2, .8])
        else:
            self.n_range = 7 
            self.range_bins = self.r_bot * np.array([
                    -1., .1, .2, .4, .8, 1.6, 3.2])
        #self.range = np.zeros(self.n_range)
        self.vision = np.zeros((self.n_heading, self.n_range))
        self.vision_bins = np.copy(self.range_bins)
        self.prox = np.zeros((self.n_bump, self.n_range))
        self.prox_bins = np.copy(self.range_bins)
        self.n_catch = 0.
        self.n_see = 0.
        self.n_reach = 0.

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
            # Find the drive magnitude
            self.drive = (  self.action[0] + 
                        2 * self.action[1] + 
                        3 * self.action[2] + 
                        4 * self.action[3] - 
                            self.action[4] - 
                        2 * self.action[5] - 
                        3 * self.action[6] - 
                        4 * self.action[7])
            # Find the spin magnitude
            self.spin = (   self.action[8]  + 
                        2 * self.action[9]  + 
                        3 * self.action[10] + 
                        4 * self.action[11] - 
                            self.action[12] - 
                        2 * self.action[13] - 
                        3 * self.action[14] - 
                        4 * self.action[15])

        def calculate_reward():
            """ Assign reward based on the current state """
            self.reward = 0.
            #if self.n_catch: print 'catch', self.n_catch
            #if self.n_see: print 'see', self.n_see
            #if self.n_reach: print 'reach', self.n_reach
            if self.n_catch:
                self.reward += self.CATCH_REWARD
                print 'caught one!========================================'
            if self.n_see:
                self.reward += self.SEE_REWARD
            if self.n_reach:
                self.reward += self.n_reach * self.RANGE_REWARD
            if np.sum(self.bump) > 0.:
                #bump_penalty:
                self.reward -= self.BUMP_PENALTY
                #self.reward -= np.sum(self.bump)
            self.n_catch = 0.
            self.n_see = 0.
            self.n_reach = 0.
        
        def convert_detectors_to_sensors():
            self.sensors = np.zeros(self.num_sensors)
            last = 0
            #first = last
            #last = first + self.n_bump
            #self.sensors[first:last] = self.bump
            '''
            self.sensors[self.n_bump:self.n_bump + self.n_range] = self.range
            self.sensors[self.n_bump + self.n_range:
                    self.n_bump + self.n_range + self.n_heading] = self.heading
            self.sensors[self.n_bump + self.n_range + self.n_heading:
                    self.n_bump + self.n_range + self.n_heading + 
                    self.n_bump * self.n_range] = self.prox.ravel()
            '''
            first = last
            last = first + self.n_heading * self.n_range
            self.sensors[first:last] = self.vision.ravel()

            #first = last
            #last = first + self.n_bump * self.n_range
            #self.sensors[first:last] = self.prox.ravel()

            #print 'bump', self.bump.ravel()
            #print 'vision', self.vision.ravel()
            #print 'prox', self.prox.ravel()

            self.bump = np.zeros(self.n_bump)
            #self.range = np.zeros(self.n_range)
            #self.heading = np.zeros(self.n_heading)
            self.vision = np.zeros(self.vision.shape)
            self.prox = np.zeros(self.prox.shape)
            
        self.timestep += 1 
        convert_actions_to_drives(action)
        for _ in range(self.CLOCKTICKS_PER_TIMESTEP):
            self.clock_step()
        calculate_reward()
        convert_detectors_to_sensors()
        full_state = np.concatenate((action.ravel(), 
                                     self.sensors.ravel(), 
                                     np.array([self.reward]) ))
        self.state_history.append(full_state)
        return self.sensors, self.reward

    def clock_step(self):
        self.clock_tick += 1
        self.clock_time = self.clock_tick * self.dt
        
        def sector_index(n_sectors, theta):
            """ For sector-based detectors, find the sector based on angle """
            theta = np.mod(theta, 2 * np.pi)
            return int(np.floor(n_sectors * theta / (2 * np.pi)))

        def wall_range(theta):
            range = 1e10
            # find distance to West wall
            range_west = self.x_bot / (np.cos(np.pi - theta) + 1e-6)
            if range_west < 0.:
                range_west = 1e10
                range_west -= self.r_bot
            range = np.minimum(range, range_west)
            # find distance to East wall
            range_east = (self.width - self.x_bot) / (np.cos(theta) + 1e-6) 
            if range_east < 0.:
                range_east = 1e10
            range_east -= self.r_bot
            range = np.minimum(range, range_east)
            # find distance to South wall
            range_south = self.y_bot / (np.sin(-theta) + 1e-6)
            if range_south < 0.:
                range_south = 1e10
            range_south -= self.r_bot
            range = np.minimum(range, range_south)
            # find distance to North wall
            range_north = (self.depth - self.y_bot) / (np.sin(theta) + 1e-6)
            if range_north < 0.:
                range_north = 1e10
            range_north -= self.r_bot
            range = np.minimum(range, range_north)
            return range

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
        i_vision_range = np.where(dist_bot_ball > self.range_bins)[0][-1] 
        #self.range[i_vision_range] += 1.
        # ball heading detection
        # the angle of contact relative to the robot's angle
        th_bot_ball_rel = np.mod(self.th_bot - th_bot_ball, 2. * np.pi)
        i_vision_heading = sector_index(self.n_heading, th_bot_ball_rel)
        #self.heading[i_vision_heading] += 1. 
        self.vision = np.zeros(self.vision.shape)
        self.vision[i_vision_heading, i_vision_range] = 1.
        
        # ball bump detection
        # debug: don't penalize ball bumps
        if delta_bot_ball > 0.:
            i_bump = sector_index(self.n_bump, th_bot_ball_rel)
            #self.bump[i_bump] += delta_bot_ball 
            self.bump[i_bump] += 0.
            #print 'ball bump', i_bump, 

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
        # Calculate the angle of contact relative to the robot's angle
        # for each of the walls. Assign the contact to the appropriate
        # bump sensor.
        delta_bot_N = self.y_bot + self.r_bot - self.depth
        delta_bot_N = np.maximum(delta_bot_N, 0.)
        if delta_bot_N > 0.:
            th_bot_N_rel = np.mod(self.th_bot - np.pi / 2., 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_N_rel)
            #self.bump[i_bump] += delta_bot_N 
            self.bump[i_bump]  = 1.
        delta_bot_S = self.r_bot - self.y_bot
        delta_bot_S = np.maximum(delta_bot_S, 0.)
        if delta_bot_S > 0.:
            th_bot_S_rel = np.mod(self.th_bot + np.pi / 2., 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_S_rel)
            #self.bump[i_bump] += delta_bot_S
            self.bump[i_bump]  = 1.
        delta_bot_E = self.x_bot + self.r_bot - self.width
        delta_bot_E = np.maximum(delta_bot_E, 0.)
        if delta_bot_E > 0.:
            th_bot_E_rel = np.mod(self.th_bot, 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_E_rel)
            #self.bump[i_bump] += delta_bot_E
            self.bump[i_bump]  = 1.
        delta_bot_W = self.r_bot - self.x_bot
        delta_bot_W = np.maximum(delta_bot_W, 0.)
        if delta_bot_W > 0.:
            th_bot_W_rel = np.mod(self.th_bot + np.pi, 2. * np.pi)
            i_bump = sector_index(self.n_bump, th_bot_W_rel)
            #self.bump[i_bump] += delta_bot_W
            self.bump[i_bump]  = 1.

        # wall proximity detection
        # calculate the range detected by each proximity sensor
        for (i_prox, prox_theta) in enumerate(
                    np.arange(0., 2 * np.pi, 2 * np.pi / self.n_bump)):
            range = wall_range(prox_theta + self.th_bot)
            i_prox_range = np.where(range > self.range_bins)[0][-1]
            self.prox[i_prox, i_prox_range] = 1.


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

        # check whether the bot caught the ball
        caught = False
        if ((i_vision_heading == 0) | (i_vision_heading == self.n_heading - 1)):
            self.n_see = 1.
            if i_vision_range == 0:
                caught = True

        self.n_reach = np.maximum(self.n_reach, float(6 - i_vision_range))

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
                    
        if (self.clock_tick % self.CLOCKTICKS_PER_FRAME) == 0:
            #print
            #print 'bump', self.bump.ravel()
            #print 'vision', self.vision.ravel()
            #print 'prox', self.prox.ravel()
            self.render()
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
        dpi = 80 # for a resolution of 720 lines
        #dpi = 120 # for a resolution of 1080 lines
        plt.savefig(full_filename, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none') 
        #plt.show()
        return

    def visualize(self, agent=None):
        """ Show what's going on in the world """
        if (self.timestep % self.TIMESTEPS_PER_FRAME) != 0:
            return
        print("world is %s seconds old " % self.clock_time)
        summary = np.sum(np.array(self.state_history), axis=0)
        print '==='
        first = 0
        last = self.num_actions
        print 'action', summary[first:last]
        #first = last
        #last = first + self.n_bump
        #print 'bumps', summary[first:last]
        first = last
        last = first + self.n_heading * self.n_range
        print 'vision', np.reshape(summary[first:last], 
                                   (self.n_heading, self.n_range))
        #first = last
        #last = first + self.n_heading * self.n_range
        #print 'proximity', np.reshape(summary[first:last], 
        #                              (self.n_heading, self.n_range))

        print 'reward', summary[-1]

        self.render()
        return
