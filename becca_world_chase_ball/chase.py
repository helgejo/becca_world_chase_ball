"""
A task in which a circular dog robot chases a ball.

In this task, the robot's sensors inform it about the relative position
of the ball, which changes often, but not about the absolute position
of the ball or about the robot's own absolute position. It is meant to
be similar to robot tasks, where global information is often lacking.
This task also requires a small amount of sequential planning, since
catching the ball typically takes multiple actions.

Usage
-----
From the command line:
    python -m chase

To generate plots of all features:
    python -m chase --plot

To generate images for making a video:
    python -m chase --film
"""

from __future__ import print_function

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import becca.connector
import becca.tools as tools
from becca_test.base_world import World as BaseWorld
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
    """
    def __init__(self, lifespan=None, plot_feature_set=False, filming=False):
        """
        Set up the world.

        Parameters
        ----------
        lifespan : int
            The number of time steps during which the robot will try to
            catch the ball. If None, it will be set to a default determined
            by the BaseWorld class.
        plot_feature_set : boolean
            Indicate whether to create visualizations of all the features.
        filming : boolean
            Indicate whether to create visualizations of the state of the
            world at a video frame rate.
        """
        BaseWorld.__init__(self, lifespan)
        # How often time steps occur in simulated time. Four per second,
        # or 250 milliseconds per timestep, is consistent with some
        # observations of how often humans can make repetitive voluntary
        # movements.
        timesteps_per_second = 4.
        # Clockticks are the finer-grained time step of the case world's
        # physics simulation.
        clockticks_per_second = 1000.
        # clockticks_per_timestep : int
        #     The number of phyics simulation time steps that occur
        #     during one Becca time step.
        self.clockticks_per_timestep = int(clockticks_per_second /
                                           timesteps_per_second)
        # plot_feature_set : boolean
        #     Indicate whether to create a set of images, one for each
        #     of the features that have been created.
        self.plot_feature_set = plot_feature_set
        # During filming, create a series of still images. These can later
        # be put together into a video.
        # timesteps_per_frame : int
        #     How often, in time steps, to render one frame.
        if filming:
            # Render the world for creating a 30 frame-per-second video
            self.timesteps_per_frame = timesteps_per_second / 30.
            # Shorten the lifespan so as not to fill the disk with images.
            self.lifespan = 250
            # Don't plot features while filming
            self.plot_feature_set = False
        else:
            self.timesteps_per_frame = 1000
        # clockticks_per_frame : int
        #     How often, in physics simulation time steps, to render one frame.
        self.clockticks_per_frame = int(self.clockticks_per_timestep *
                                        self.timesteps_per_frame)
        # clockticks_until_render : int
        #     A counter for how many clockticks are left until creating
        #     the next frame.
        self.clockticks_until_render = self.clockticks_per_frame
        # brain_visualize_period : int
        #     How often to put a picture of Becca's internal state
        #     on the screen.
        self.brain_visualize_period = 1e3
        # name : str
        #     A short descriptor of this world.
        self.name = 'chase'
        self.name = 'chase_1' # Low-res vision
        # name_long : str
        #     A more verbose descriptor of this world.
        self.name_long = 'ball chasing world'
        print("Entering", self.name_long)

        # The robot has several types of sensors: bump sensors, proximity
        # detectors, primitive vision, velocity and acceleration.
        # These parameters define how many of them there are and
        # how sensitive they are.
        # n_bump_heading : int
        #     The number of bump sensors arrayed around the perimeter of the
        #     robot. Each one covers a fraction of the robot's circumference.
        #     For instance, if n_bump_heading = 4 each bump sensor registers
        #     contact from an 90 degrees (pi/2 radians) angular bin.
        #     The first bin starts right between the robot's eyes,
        #     at 0 degrees, and extends clockwise.
        #     The rest are evenly spaced around the robot.
        self.n_bump_heading = 1
        # n_bump_mag : int
        #     How many distinct levels of bump pressure each sensor can detect.
        self.n_bump_mag = 3
        # In the robot's primitive vision, it can only distinguish
        # angular position (heading) and distance (range).
        # n_ball_heading : int
        #     The number of bins for detecting the heading of the ball.
        #     These angular bins are defined using the same scheme as
        #     the bump sensor bins: equal sized divisions of 360 degrees,
        #     starting between the robots eyes and extending clockwise
        #     to surround the robot.
        self.n_ball_heading = 9
        # n_ball_range : int
        #     The number of distinct bins that the ball distance can fall
        #     into.
        self.n_ball_range = 4
        # Proximity sensors feel how far away the wall is in several different
        # directions. They don't sense the ball.
        # n_prox_heading : int
        #     Similar to bump and vision sensors, proximity sensors are
        #     evenly arrayed around the robot. Unlike bump and vision,
        #     proximity is sensed a distinct points, rather than in bins.
        #     The first sensor is at 0 degrees, the second at
        #     360/n_prox_heading to the clockwise, etc.
        self.n_prox_heading = 2
        # n_prox_heading : int
        #     The number of discrete bins for measuring distance to
        #     the wall.
        self.n_prox_range = 1
        # n_prox : int
        #     The total number of proximity sensors, one per bin.
        #     There is one set of range bins for each heading.
        self.n_prox = self.n_prox_heading * self.n_prox_range
        # n_bump : int
        #     The total number of bump sensors, one per bin.
        #     There is one set of bump magnitude bins for each heading.
        self.n_bump = self.n_bump_heading * self.n_bump_mag
        # n_vel_per_axis : int
        #     The number of discrete bins for sensing velocity in
        #     translation (x and y) and rotation (theta).
        #     This works best if it's odd, to leave one bin centered on
        #     zero velocity.
        self.n_vel_per_axis = 3
        if self.n_vel_per_axis % 2 == 0:
            self.n_vel_per_axis += 1
        # n_acc_per_axis
        #     As with velocity, the number of bins for sensing acceleration
        #     in x, y, and theta. This one also should be odd.
        self.n_acc_per_axis = 1
        if self.n_acc_per_axis % 2 == 0:
            self.n_acc_per_axis += 1
        # n_vel : int
        #     The total number of velocity sensors across all axes.
        self.n_vel = 3 * self.n_vel_per_axis
        # n_acc : int
        #     The total number of acceleration sensors across all axes.
        self.n_acc = 3 * self.n_acc_per_axis

        # Set the physical constants and initial state of the world.
        self._initialize_world()
        # num_sensors : int
        #     The total number of sensors that the chase world will
        #     be passing back to Becca.
        self.num_sensors = (self.n_ball_range + self.n_ball_heading +
                            self.n_prox + self.n_bump +
                            self.n_vel + self.n_acc)
        # num_actions : int
        #     The total number of actions that can be taken in
        #     the chase world.
        self.num_actions = 20
        # action : array of floats
        #     The set of actions taken in the current time steo.
        self.action = np.zeros(self.num_actions)
        # drive, spin, effort : float
        #     The effect of actions on the robot. Drive is a forward/backward
        #     force. Spin is a rotational force. Effort is the sum of the
        #     magnitudes of all the drive and spin commands. It's used to
        #     incur a small penalty for making the robot work.
        self.drive = 0.
        self.spin = 0.
        self.effort = 0.
        # catch_reward : float
        #     The reward for catching the ball.
        self.catch_reward = 1.
        # touch_reward : float
        #     The reward for gentle bumps. Rewarding this may encourage
        #     wall following behavior.
        self.touch_reward = 1e-8
        # bump_penalty : float
        #     The magnitude of punishment (negative reward) for violent
        #     bumps. Penalizing this may discourage wall-ramming.
        self.bump_penalty = 1e-2
        # effort_penalty : float
        #     The magnitude of punishment (negative reward) for taking
        #     action. A small penalty for this will encourage efficiency.
        self.effort_penalty = -1e-8
        # module_path : string
        #     The fully qualified path name to the module.
        #     This is used to store and retrieve logs and images
        #     related to this world.
        self.module_path = os.path.dirname(os.path.abspath(__file__))
        # log_directory : string
        #     The fully qualified path name to a log directory.
        #     There's no need to initialize this one, since the brain
        #     takes care of that.
        self.log_directory = os.path.join(self.module_path, 'log')
        # features_directory : string
        #     The fully qualified path name to a features directory.
        #     This is for storing visual representations fo features.
        self.features_directory = os.path.join(self.module_path, 'features')
        # Check whether the features directory is already there.
        # If not, create it.
        if not os.path.isdir(self.features_directory):
            os.makedirs(self.features_directory)
        # frames_directory : string
        #     The fully qualified path name to a frames directory.
        #     This is for storing still images that can be stitched
        #     together into videos.
        self.frames_directory = os.path.join(self.module_path, 'frames')
        # Check whether the frames directory is already there.
        # If not, create it.
        if not os.path.isdir(self.frames_directory):
            os.makedirs(self.frames_directory)
        # frame_counter : int
        #     Intiailize an identifier to append to still frames,
        #     indicating their order in the sequence.
        self.frame_counter = 10000

    def _initialize_world(self):
        """
        Set up the physics of the simulation.
        """
        # clock_tick : int
        #     A counter for the total number of clock ticks that have
        #     elapsed since the world began.
        self.clock_tick = 0
        # clock_time : float
        #     The total number of seconds that have elapsed since the
        #     world began.
        self.clock_time = 0.
        #self.tick_counter = 0

        #-----------------
        # This section of constants defines the dimensions and physics of
        # the ball chase world. These are also present in clock_step()
        # in clock_step.py.
        # Any changes made in this section need to also be made there.
        # dt : float
        #     Seconds per clock tick. This also needs to be consistent with
        #     clockticks_per_second in __init__(), above.
        self.dt = .001

        # wall parameters
        # width, depth : float
        #     The dimensions of the room in meters in x and y, respectively.
        self.width = 5.
        self.depth = 5.
        # k_wall : float
        #     The spring stiffness of the walls in Newtons / meter
        self.k_wall = 3000.

        # ball parameters
        # r_ball : float
        #     The radius of the ball, in meters.
        self.r_ball = .4
        # k_ball : float
        #     The stiffness of the ball in Newtons / meter.
        self.k_ball = 3000.
        # c_ball : float
        #     The coeficient of dynamic friction between the ball
        #     and the floor, in Newton-seconds / meter.
        self.c_ball = 1.
        # cc_ball : float
        #     The coefficient of Coulomb friction between the ball
        #     and the floor, in Newtons.
        self.cc_ball = 1.
        # m_ball : float
        #     The mass of the ball, in kilograms.
        self.m_ball = 1.

        # robot parameters
        # r_bot : float
        #     The radius of the bot, in meters.
        self.r_bot = .8
        # k_bot : float
        #     The spring stiffness of the bot, in Newtons / meter.
        self.k_bot = 3000.
        # c_bot : float
        #     The coefficient of dynamic froction between the bot and floor,
        #     for translational (scooting) motion in Newton-seconds / meter.
        self.c_bot = 10.
        # cc_bot : float
        #     The coefficient of Coulomb friction between the bot
        #     and the floor, for translational (scooting) motion in Newtons.
        self.cc_bot = 1.
        # d_bot : float
        #     The coefficient of dynamic froction between the bot and floor,
        #     for rotational (spinning) motion
        #     in Newton-meter-seconds / radian.
        self.d_bot = 10.
        # dd_bot : float
        #     The coefficient of Coulomb friction between the bot
        #     and the floor, for rotational (spinning) motion in Newtons.
        self.dd_bot = 1.
        # m_bot : float
        #     The mass of the bot, in kilograms.
        self.m_bot = 5.
        # I_bot : float
        #     The rotational inertia of the bot, in kilogram-meters**2.
        self.I_bot = 1.
        # mouth_width_bot : float
        #     The angle over which the bot can catch the ball, in radians.
        #     The bot catches the ball by touching it with its mouth.
        #     A wider mouth makes it easier to catch the ball.
        self.mouth_width_bot = np.pi / 3
        #-------------------------
        # This is the end of the section that is duplicated in clock_step().


        # State variables
        # These continue to evolve. They are initialized here.
        #
        # The coordinate axes of the room
        # are at the lower left hand corner of the room.
        # The positive x direction is to the right, along the width, and
        # the positive y direction is up, along the depth.
        # For absolute angles, 0 radians is in the positive x direction
        # and pi/2 radians is in the positive y direction.

        # ball state
        # x_ball, y_ball : float
        #     The position of the center of the ball in meters.
        #     Start out near the upper right corner of the room.
        self.x_ball = 4.
        self.y_ball = 4.
        # vx_ball, vy_ball : float
        #     The velocity of the center of the ball in meters / second.
        #     Start out at rest.
        self.vx_ball = 0.
        self.vy_ball = 0.
        # ax_ball, ay_ball : float
        #     The acceleration of the center of the ball in meters**2 / second.
        #     Start out at rest.
        self.ax_ball = 0.
        self.ay_ball = 0.

        # Bot state
        # x_bot, y_bot : float
        #     The position of the center of the bot in meters.
        #     Start out near the middle of the room.
        self.x_bot = 2.
        self.y_bot = 2.
        # th_bot : float
        #     The angular orientation of the bot's mouth, in radians.
        #     Start out pointed toward the upper rigth corner of the room.
        self.th_bot = np.pi / 4.
        # vx_bot, vy_bot : float
        #     The velocity of the center of the bot in meters / second.
        #     Start out at rest.
        self.vx_bot = 0.
        self.vy_bot = 0.
        # omega_bot : float
        #     The angular velocity of the bot in radians / second.
        #     Start out at rest.
        self.omega_bot = 0.
        # ax_bot, ay_bot : float
        #     The acceleration of the center of the bot in meters**2 / second.
        #     Start out at rest.
        self.ax_bot = 0.
        self.ay_bot = 0.
        # alpha_bot : float
        #     The angular acceleration of the center of the bot
        #     in radians / second**2.
        #     Start out at rest.
        self.alpha_bot = 0.

        # Detector parameters
        # Set minimum and maximum values for the bin edges, where necessary.
        # Useful values for these were determined empirically.
        min_prox_range = -1.
        min_vision_range = -1.
        max_vision_range = self.width * 1.2
        max_prox_range = self.width * 1.2
        max_bump_mag = .2
        max_v_fwd = 3.8
        max_v_lat = 3.5
        max_omega = 8.
        max_a_fwd = 50.
        max_a_lat = 35.
        max_alpha = 125.

        def build_bins_one_sided(num_bins, max_edge, min_edge=-.01):
            """
            Build an array of left bin edges from near zero to a positive value.

            Calculate the edges of bins that get wider as they move outward
            in a geometric sequence (fractional powers of 2).
            The final right bin edge of infinity is implied.

            The array of bin edges produced will be of the form
                [min_edge, 0, a_2, a_3, ..., a_n-1]
            where a_i form a geometric sequence to max_edge
                a_i = max_edge * 2 ** (i / (num_bins - 1))

            Parameters
            ----------
            min_edge, max_edge : float
                The bounding bin edges.
            num_bins : int
                The total number of bins to create.
            """
            bins = [min_edge]
            for i in range(num_bins - 1):
                fraction = float(i) / (num_bins - 1)
                bin_edge = max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)
            return np.array(bins)

        # ball_range_bins, prox_range_bins, bump_mag_bins : array of floats
        #     The set of bin edges for detecting the range of the ball,
        #     the distance to a wall and the magnitude of bump force,
        #     respectively.
        #     These are all one-sided sets of bins, meaning that nonimally
        #     the quantities they measure should all be non-negative.
        #     Because there can be quirks of the simulation, like the bot
        #     colliding with the wall, these can be slightly negative.
        #     The minimum bin range has been adjusted
        #     to include a bin for catching small negative values..
        self.ball_range_bins = build_bins_one_sided(self.n_ball_range,
                                                    max_vision_range,
                                                    min_vision_range)
        self.prox_range_bins = build_bins_one_sided(self.n_prox_range,
                                                    max_prox_range,
                                                    min_prox_range)
        self.bump_mag_bins = build_bins_one_sided(self.n_bump_mag,
                                                  max_bump_mag)

        def build_bins_two_sided(num_bins, max_edge):
            """
            Build an array of left bin edges that is symmetric about zero.

            The width of the bins gets geometrically greater
            (fractional powers of two) as they get further from zero.
            The final right bin edge of infinity is implied.

            The array of bin edges produced will be of the form
                [-BIG, -a_n, ..., -a_2, -a_1, a_1, a_2, ..., a_num_bins]
            where a_i form a geometric sequence to max_edge
                a_i = max_edge * (2 ** (i / (num_bins + 1)) - 1)

            Parameters
            ----------
            max_edge : float
                The inner bin edge of the bins furthest from zero,
                both on the positive and negative side.
            num_bins : int
                The number of bins to create on both the positive and
                the negative sides.
                Create 2n + 1 bins. An odd number seems to work best.
                It creates a small bin centered on zero.
            """
            # Initialize the left-most bin edge to be large and negative.
            bins = [-1e10]

            # Build the negative side.
            for i in range(num_bins):
                j = num_bins - (i + 1)
                fraction = float(j + 1) / (num_bins + 1)
                bin_edge = -max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)
            # Build the positive side.
            for i in range(num_bins):
                fraction = float(i + 1) / (num_bins + 1)
                bin_edge = max_edge * (2.** fraction - 1.)
                bins.append(bin_edge)

            return np.array(bins)

        # Find the number of bins to create on each side of zero.
        n_vel_half = (self.n_vel_per_axis - 1) / 2
        n_acc_half = (self.n_acc_per_axis - 1) / 2
        # v_fwd_bins, v_lat_bins, omega_bins : array of floats
        #     Left bin edges for velocity in the forward direction,
        #     velocity in the lateral direction, and angular
        #     velocity, respectively.
        #     These bins record the sensed velocities of the bot.
        self.v_fwd_bins = build_bins_two_sided(n_vel_half, max_v_fwd)
        self.v_lat_bins = build_bins_two_sided(n_vel_half, max_v_lat)
        self.omega_bins = build_bins_two_sided(n_vel_half, max_omega)
        # a_fwd_bins, a_lat_bins, alpha_bins : array of floats
        #     Left bin edges for acceleration in the forward direction,
        #     acceleration in the lateral direction, and angular
        #     acceleration, respectively.
        #     These bins record the sensed accelerations of the bot.
        self.a_fwd_bins = build_bins_two_sided(n_acc_half, max_a_fwd)
        self.a_lat_bins = build_bins_two_sided(n_acc_half, max_a_lat)
        self.alpha_bins = build_bins_two_sided(n_acc_half, max_alpha)

        # v_heading, v_range : array of floats
        #     Bin counts for vision-based sensing of the ball's
        #     heading and range, relative to the bot.
        self.v_heading = np.zeros(self.n_ball_heading)
        self.v_range = np.zeros(self.n_ball_range)
        # prox : 2D array of floats
        #     Bin counts for each of the bot's proximity sensors.
        self.prox = np.zeros((self.n_prox_heading, self.n_prox_range))
        # bump : 2D array of floats
        #     Bin counts for each of the bot's bump sensors.
        self.bump = np.zeros((self.n_bump_heading, self.n_bump_mag))
        # v_fwd_bot_sensor, v_lat_bot_sensor, omega_bot_sensor : array of floats
        #     Bin counts for the bot's forward velocity, lateral velocity,
        #     and angular velocity sensors, respectively.
        self.v_fwd_bot_sensor = np.zeros(self.n_vel_per_axis)
        self.v_lat_bot_sensor = np.zeros(self.n_vel_per_axis)
        self.omega_bot_sensor = np.zeros(self.n_vel_per_axis)
        # a_fwd_bot_sensor, a_lat_bot_sensor, alpha_bot_sensor : array of floats
        #     Bin counts for the bot's forward acceleration,
        #     lateral acceleration and angular acceleration, respectively.
        self.a_fwd_bot_sensor = np.zeros(self.n_acc_per_axis)
        self.a_lat_bot_sensor = np.zeros(self.n_acc_per_axis)
        self.alpha_bot_sensor = np.zeros(self.n_acc_per_axis)

        # n_catch : float
        #     The number of times the bot has caught the ball this time step.
        self.n_catch = 0.

        # Create a prototype force profile.
        # It is a triangular profile, ramping linearly from 0 to peak over its
        # first half, than ramping linearly back down to 0.
        duration = .25
        peak = 1. # Newtons
        length = int(duration / self.dt)
        # proto_force : array of floats
        #     The triangular prototype force profile.
        self.proto_force = np.ones(length)
        self.proto_force[:length/2] = (np.cumsum(self.proto_force)[:length/2] *
                                       peak / float(length/2))
        self.proto_force[length:length-length/2-1:-1] = (
            self.proto_force[:length/2])
        self.proto_force[length/2] = peak

        # Create a ring buffer for future force information.
        buffer_duration = 1. # seconds
        buffer_length = int(buffer_duration / self.dt)
        # f_x_buffer, f_y_buffer, tau_buffer : array of floats
        #     The near-term future of applied force in the x and y directions
        #     and of the torque. This allows an action to be executed over
        #     a period of time. In this case, 250 milliseconds.
        self.f_x_buffer = np.zeros(buffer_length)
        self.f_y_buffer = np.zeros(buffer_length)
        self.tau_buffer = np.zeros(buffer_length)


    def zero_sensors(self):
        """
        Reset all sensor values to zero.

        Sensor activity is accumuated in bins during each time step.
        At the end of the time step it is passed to the agent, and all the
        bin counts are reset to zero.
        """
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
        Construct a sensor vector from the detector values.

        Concatenate all the separate sensors (detectors) together into a single
        sensor array and reset the detectors back to zero.
        The output is the modified self.sensors.
        This is trivial, except for the indexing and bookkeeping.
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


    def convert_sensors_to_detectors(self, sensors):
        """
        Construct a sensor vector from the detector values.

        This is the inverse operation of convert_detectors_to_sensors.
        For a given array of self.sensor values, break it out into
        all the individual detectors that contributed to it.
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
        """
        Create a text report of all the current detector activity.
        """
        print()
        print('___________________________________')
        print('bump')
        tools.format_decimals(self.bump)
        print('v_heading')
        tools.format_decimals(self.v_heading)
        print('v_range')
        tools.format_decimals(self.v_range)
        print('prox')
        tools.format_decimals(self.prox)
        print('v_fwd_bot_sensor')
        tools.format_decimals(self.v_fwd_bot_sensor)
        print('v_lat_bot_sensor')
        tools.format_decimals(self.v_lat_bot_sensor)
        print('omega_bot_sensor')
        tools.format_decimals(self.omega_bot_sensor)
        print('a_fwd_bot_sensor')
        tools.format_decimals(self.a_fwd_bot_sensor)
        print('a_lat_bot_sensor')
        tools.format_decimals(self.a_lat_bot_sensor)
        print('alpha_bot_sensor')
        tools.format_decimals(self.alpha_bot_sensor)


    def step(self, action):
        """
        Take one time step through the world.
        """

        def convert_actions_to_drives(action):
            """
            Convert the action array into forward and rotational motion.

            There are 20 individual action commands. The first five drive
            the bot forward with a force proportional to 1, 2, 4, 8 and 16
            respectively. The next five commands do the same, but backward.
            The next five commands are for spinning the bot counter-clockwise
            in the same proportions, and the final five are for spinning
            it clockwise.
            """
            self.action = action
            # Find the drive magnitude.
            self.drive = (1 * self.action[0] +
                          2 * self.action[1] +
                          4 * self.action[2] +
                          8 * self.action[3] +
                          16 * self.action[4] -
                          1 * self.action[5] -
                          2 * self.action[6] -
                          4 * self.action[7] -
                          8 * self.action[8] -
                          16 * self.action[9])
            # Find the spin magnitude.
            self.spin = (1 * self.action[10] +
                         2 * self.action[11] +
                         4 * self.action[12] +
                         8 * self.action[13] +
                         16 * self.action[14] -
                         1 * self.action[15] -
                         2 * self.action[16] -
                         4 * self.action[17] -
                         8 * self.action[18] -
                         16 * self.action[19])
            # Find the total effort.
            self.effort = (1 * self.action[0] +
                           2 * self.action[1] +
                           4 * self.action[2] +
                           8 * self.action[3] +
                           16 * self.action[4] +
                           1 * self.action[5] +
                           2 * self.action[6] +
                           4 * self.action[7] +
                           8 * self.action[8] +
                           16 * self.action[9] +
                           1 * self.action[10] +
                           2 * self.action[11] +
                           4 * self.action[12] +
                           8 * self.action[13] +
                           16 * self.action[14] +
                           1 * self.action[15] +
                           2 * self.action[16] +
                           4 * self.action[17] +
                           8 * self.action[18] +
                           16 * self.action[19])


        def calculate_reward():
            """
            Assign reward based on accumulated target events over the
            previous time step.
            """
            self.reward = 0.
            # Reward the bot for catching the ball.
            self.reward += self.n_catch * self.catch_reward
            # The bot also likes gentle contact.
            self.reward += (2 * np.sum(self.bump[:, 0]) +
                            np.sum(self.bump[:, 1])) * self.touch_reward
            # It doesn't like violent contact.
            self.reward -= (2 * np.sum(self.bump[:, -1]) +
                            np.sum(self.bump[:, -2])) * self.bump_penalty
            # And it is just a little bit lazy.
            self.reward -= self.effort * self.effort_penalty

            self.n_catch = 0.

        # Use the convenient internal methods defined above to
        # step the world forward at a high level of abstraction.
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
  
            (self.clock_tick, self.clock_time,
             self.n_catch,
             self.x_ball, self.y_ball,
             self.vx_ball, self.vy_ball,
             self.ax_ball, self.ay_ball,
             self.x_bot, self.y_bot, self.th_bot,
             self.vx_bot, self.vy_bot, self.omega_bot,
             self.ax_bot, self.ay_bot, self.alpha_bot
            ) = cs.clock_step(
                self.clock_tick, self.clock_time,
                n_clockticks, self.clockticks_per_timestep,
                self.ball_range_bins, self.prox_range_bins,
                self.bump_mag_bins,
                self.v_fwd_bins, self.v_lat_bins, self.omega_bins,
                self.a_fwd_bins, self.a_lat_bins, self.alpha_bins,
                self.v_heading, self.v_range,
                self.prox,
                self.bump,
                self.v_fwd_bot_sensor, self.v_lat_bot_sensor,
                self.omega_bot_sensor,
                self.a_fwd_bot_sensor, self.a_lat_bot_sensor,
                self.alpha_bot_sensor,
                self.n_catch,
                self.proto_force,
                self.f_x_buffer, self.f_y_buffer, self.tau_buffer,
                self.drive, self.spin,
                self.x_ball, self.y_ball,
                self.vx_ball, self.vy_ball,
                self.ax_ball, self.ay_ball,
                self.x_bot, self.y_bot, self.th_bot,
                self.vx_bot, self.vy_bot, self.omega_bot,
                self.ax_bot, self.ay_bot, self.alpha_bot)

        calculate_reward()
        self.convert_detectors_to_sensors()
        return self.sensors, self.reward


    def plot_robot(self, axis, x_bot, y_bot, th_bot, alpha=1., dzorder=0):
        """
        Plot the robot and sensors in the current figure and axes.
        """
        # Rixel color is gray (3b3b3b)
        # eye color is light blue (c1e0ec)
        robot_color = (59./255., 59./255., 59./255.)
        eye_color = (193./255., 224./255., 236./255.)
        axis.add_patch(patches.Circle((x_bot, y_bot),
                                      self.r_bot, color=robot_color,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((x_bot, y_bot),
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
        axis.add_patch(patches.Circle((x_left, y_left),
                                      self.r_bot * .1,
                                      color=eye_color,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((xp_left, yp_left),
                                      self.r_bot * .06,
                                      color=tools.copper_shadow,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((x_left, y_left),
                                      self.r_bot * .1,
                                      color=tools.copper_shadow,
                                      linewidth=1., fill=False,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((x_right, y_right),
                                      self.r_bot * .1,
                                      color=eye_color,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((xp_right, yp_right),
                                      self.r_bot * .06,
                                      color=tools.copper_shadow,
                                      alpha=alpha, zorder=-dzorder))
        axis.add_patch(patches.Circle((x_right, y_right),
                                      self.r_bot * .1,
                                      color=tools.copper_shadow,
                                      linewidth=1., fill=False,
                                      alpha=alpha, zorder=-dzorder))


    def plot_sensors(self, axis, x_bot, y_bot, th_bot):
        """
        Visually represent what the sensors are detecting around the robot.
        """
        # Show sensors visually.
        # ball range sensor
        max_alpha = .3
        for i_vision_range in np.nonzero(self.v_range)[0]:
            magnitude = self.v_range[i_vision_range]
            i_range = np.minimum(i_vision_range + 1, self.n_ball_range - 1)
            range_radius = self.r_bot + self.ball_range_bins[i_range]
            alpha = np.minimum(1., magnitude * max_alpha)
            axis.add_patch(patches.Circle((x_bot, y_bot), range_radius,
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
            x_pos = x_bot + np.array([
                0.,
                np.cos(heading_sensor_angle_1) * heading_sensor_radius,
                np.cos(heading_sensor_angle_2) * heading_sensor_radius,
                0.])
            y_pos = y_bot + np.array([
                0.,
                np.sin(heading_sensor_angle_1) * heading_sensor_radius,
                np.sin(heading_sensor_angle_2) * heading_sensor_radius,
                0.])
            axis.fill(x_pos, y_pos, color=tools.oxide,
                      alpha=np.minimum(1., magnitude * max_alpha),
                      zorder=-1)

        # proximity sensors
        for (i_prox, prox_theta) in enumerate(
                np.arange(0., 2 * np.pi,
                          2. * np.pi / self.n_prox_heading)):
            for i_range in np.where(self.prox[i_prox, :] > 0)[0]:
                magnitude = self.prox[i_prox, i_range]
                i_prox_range = np.minimum(i_range, self.n_prox_range - 1)
                prox_range = self.r_bot + self.prox_range_bins[i_prox_range]
                prox_angle = th_bot - prox_theta
                x_pos = x_bot + np.cos(prox_angle) * prox_range
                y_pos = y_bot + np.sin(prox_angle) * prox_range
                prox_sensor_radius = self.r_bot / 10.
                alpha = np.minimum(1., magnitude * max_alpha)
                axis.add_patch(patches.Circle((x_pos, y_pos), prox_sensor_radius,
                                              color=tools.copper,
                                              alpha=alpha,
                                              linewidth=0., fill=True))
                plt.plot([x_bot, x_pos], [y_bot, y_pos],
                         color=tools.copper, linewidth=.5,
                         alpha=alpha,
                         zorder=-10)

        # bump sensors
        max_alpha = .8
        for (i_bump, bump_theta) in enumerate(
                np.arange(0., 2 * np.pi, 2 * np.pi / self.n_bump_heading)):
            bump_angle = th_bot - bump_theta
            x_pos = x_bot + np.cos(bump_angle) * self.r_bot
            y_pos = y_bot + np.sin(bump_angle) * self.r_bot
            for i_mag in np.where(self.bump[i_bump, :] > 0)[0]:
                magnitude = np.minimum(1., self.bump[i_bump, i_mag])

                bump_sensor_radius = ((self.r_bot * i_mag) /
                                      (2. * self.n_bump_mag))
                bump_sensor_radius = np.maximum(0., bump_sensor_radius)
                alpha = np.minimum(1., magnitude * max_alpha)
                axis.add_patch(patches.Circle(
                    (x_pos, y_pos), bump_sensor_radius,
                    color=tools.copper_shadow,
                    alpha=alpha, linewidth=0., fill=True))


        # speed and acceleration sensors
        scale = .1
        dx_pos = self.vx_bot * scale
        dy_pos = self.vy_bot * scale
        dth = self.omega_bot * scale
        self.plot_robot(axis, x_bot + dx_pos, y_bot + dy_pos, th_bot + dth,
                        alpha=.3, dzorder=13)
        ddx_pos = dx_pos + self.ax_bot * scale ** 2
        ddy_pos = dy_pos + self.ay_bot * scale ** 2
        ddth = dth + self.alpha_bot * scale ** 2
        self.plot_robot(axis, x_bot + ddx_pos, y_bot + ddy_pos, th_bot + ddth,
                        alpha=.15, dzorder=16)


    def render(self, dpi=80):
        """
        Make a pretty picture of what's going on in the world

        Parameters
        ----------
        dpi : int
            Dots per inch in the rendered image.
            dpi = 80 for a resolution of 720 lines.
            dpi = 120 for a resolution of 1080 lines.
        """
        fig = plt.figure(num=83)
        fig.clf()
        fig, axis = plt.subplots(num=83, figsize=(self.width, self.depth))

        # The walls
        plt.plot(np.array([0., self.width, self.width, 0., 0.]),
                 np.array([0., 0., self.depth, self.depth, 0.]),
                 linewidth=10, color=tools.copper_shadow)
        # The floor
        axis.fill([0., self.width, self.width, 0., 0.],
                  [0., 0., self.depth, self.depth, 0.],
                  color=tools.light_copper, zorder=-100)
        for x_pos in np.arange(1., self.width):
            plt.plot(np.array([x_pos, x_pos]), np.array([0., self.depth]),
                     linewidth=2, color=tools.copper_highlight,
                     zorder=-99)
        for y_pos in np.arange(1., self.depth):
            plt.plot(np.array([0., self.width]), np.array([y_pos, y_pos]),
                     linewidth=2, color=tools.copper_highlight,
                     zorder=-99)
        # The ball
        axis.add_patch(patches.Circle((self.x_ball, self. y_ball),
                                      self.r_ball, color=tools.oxide))
        axis.add_patch(patches.Circle((self.x_ball, self. y_ball),
                                      self.r_ball,
                                      color=tools.copper_shadow,
                                      linewidth=2., fill=False))

        self.plot_robot(axis, self.x_bot, self.y_bot, self.th_bot)
        self.plot_sensors(axis, self.x_bot, self.y_bot, self.th_bot)

        plt.axis('equal')
        plt.axis('off')
        # Make sure the walls don't get clipped.
        plt.ylim((-.1, self.depth + .1))
        plt.xlim((-.1, self.width + .1))
        fig.canvas.draw()
        # Save the image.
        filename = ''.join([self.name, '_', str(self.frame_counter), '.png'])
        full_filename = os.path.join(self.frames_directory, filename)
        self.frame_counter += 1
        facecolor = fig.get_facecolor()
        plt.savefig(full_filename, format='png', dpi=dpi,
                    facecolor=facecolor, edgecolor='none')


    def visualize_world(self, brain):
        """
        Show what's going on in the world.
        """
        if (self.timestep % self.timesteps_per_frame) != 0:
            return
        timestr = tools.timestr(self.clock_time, s_per_step=1.)
        print(' '.join(['world running for', timestr]))

        # Periodcally show the entire feature set.
        if self.plot_feature_set:
            feature_set = brain.cortex.get_index_projections()[0]
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
                    fig, axis = plt.subplots(num=99, figsize=(3 * self.width,
                                                              3 * self.depth))
                    self.plot_robot(axis, 0., 0., np.pi/2)
                    self.plot_sensors(axis, 0., 0., np.pi/2)

                    plt.axis('equal')
                    plt.axis('off')
                    plt.ylim((-self.depth, self.depth))
                    plt.xlim((-self.width, self.width))
                    fig.canvas.draw()

                    filename = '_'.join(('block', str(block_index).zfill(2),
                                         'feature', str(feature_index).zfill(4),
                                         self.name, 'world.png'))
                    #full_filename = os.path.join(self.module_path,
                    #        'features', filename)
                    full_filename = os.path.join(self.features_directory,
                                                 filename)
                    plt.title(filename)
                    plt.savefig(full_filename, format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Let Becca learn to chase a ball.')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help="Create picture of each of the features.")
    parser.add_argument(
        '-f', '--film', action='store_true',
        help="Create picutres of the world at a video frame rate.")
    args = parser.parse_args()

    filming_flag = bool(args.film)
    plot_features = bool(args.plot)

    default_lifespan = 1e8
    becca.connector.run(World(plot_feature_set=plot_features,
                              lifespan=default_lifespan,
                              filming=filming_flag),
                              restore=True)
