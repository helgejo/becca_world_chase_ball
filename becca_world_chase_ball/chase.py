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

#import matplotlib.patches as patches
#import matplotlib.pyplot as plt
import numpy as np

import becca.connector
from becca.base_world import World as BaseWorld
import becca.tools as tools
import becca_toolbox.ffmpeg_tools as vt
#import becca_toolbox.feature_tools as ft
import becca_world_chase_ball.clock_step as cs
import becca_world_chase_ball.chase_viz as chase_viz


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
        # filming : boolean
        #     An indicator of whether a movie is being created.
        self.filming = filming
        if self.filming:
            # frames_per_second : float
            #    The number of still frames to include in one-second of
            #    rendered video of the world.
            self.frames_per_second = 30.
            # timesteps_per_frame : int
            #     How often, in time steps, to render one frame.
            self.timesteps_per_frame = (timesteps_per_second /
                                        self.frames_per_second)
            # Shorten the lifespan so as not to fill the disk with images.
            self.lifespan = 250
            # Don't plot features while filming
            self.plot_feature_set = False
        else:
            self.timesteps_per_frame = 1e4
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
        #self.name = 'chase'
        #self.name = 'chase_1' # Low-res vision
        #self.name = 'chase_2' # node_sequence_threshold 1e2
        #self.name = 'chase_3' # node_sequence_threshold 3e2
        #self.name = 'chase_4' # node_sequence_threshold 1e3
        #self.name = 'chase_5' # node_sequence_threshold 1e3
        #self.name = 'chase_6' # starting fresh
        #self.name = 'chase_7' # minimal
        self.name = 'chase_8' # 1e4 sequence threshold
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
        self.n_bump_mag = 1
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
        self.n_vel_per_axis = 1
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
            #self.reward += (2 * np.sum(self.bump[:, 0]) +
            #                np.sum(self.bump[:, 1])) * self.touch_reward
            # It doesn't like violent contact.
            #self.reward -= (2 * np.sum(self.bump[:, -1]) +
            #                np.sum(self.bump[:, -2])) * self.bump_penalty
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
                #self.render()
                chase_viz.render(self)
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


    def visualize(self, brain):
        """
        Show what's going on in the world.
        """
        chase_viz.visualize(self, brain)


    def close_world(self, brain):
        """
        Wrap up the world at the end of its lifetime.

        In this case, that means to take accumulated still images
        and stitch them into a movie, if the filming option is active.

        Parameters
        ----------
        brain : Brain
            The brain that lived in the world during its run.
        """
        if self.filming:
            movie_filename = ''.join([self.name, '_',
                                      str(brain.timestep), '.mp4'])
            movie_full_filename = os.path.join(self.log_directory,
                                               movie_filename)
            print(movie_filename)
            print(movie_full_filename)

            vt.make_movie(self.frames_directory,
                          movie_filename=movie_full_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Let Becca learn to chase a ball.')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help="Create picture of each of the features.")
    parser.add_argument(
        '-f', '--film', action='store_true',
        help="Create pictures of the world at a video frame rate.")
    args = parser.parse_args()

    filming_flag = bool(args.film)
    plot_features = bool(args.plot)

    default_lifespan = 1e8
    if filming_flag:
        becca.connector.run(World(plot_feature_set=False, filming=True),
                            restore=True)
    else:
        becca.connector.run(World(plot_feature_set=plot_features,
                                  lifespan=default_lifespan,
                                  filming=False),
                            restore=True)
