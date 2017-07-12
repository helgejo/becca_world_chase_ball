"""
A task in which a circular dog robot chases a ball.

In this task, the robot's sensors inform it about the relative position
of the ball, which changes often, but not about the absolute position
of the ball or about the robot's own absolute position. It is meant to
be similar to robot tasks, where global information is often lacking.
This task also requires a small amount of sequential planning, since
catching the ball typically takes multiple actions.
"""

from __future__ import print_function

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

import becca.tools as tools
import becca_toolbox.feature_tools as ft


def plot_robot(world, axis, x_bot, y_bot, th_bot, lw=.5, alpha=1., dzorder=0):
    """
    Plot the robot and sensors in the current figure and axes.
    """
    # Rixel color is gray (8b8b81)
    # eye color is light blue (c1e0ec)
    robot_color = (139./255., 139./255., 129./255.)
    eye_color = (193./255., 224./255., 236./255.)
    axis.add_patch(patches.Circle((x_bot, y_bot),
                                  world.r_bot, color=robot_color,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((x_bot, y_bot),
                                  world.r_bot,
                                  color=tools.copper_shadow,
                                  linewidth=lw, fill=False,
                                  alpha=alpha, zorder=-dzorder))
    # robot eyes
    x_left = (x_bot + world.r_bot * .7 * np.cos(th_bot) +
              world.r_bot * .25 * np.cos(th_bot + np.pi/2.))
    y_left = (y_bot + world.r_bot * .7 * np.sin(th_bot) +
              world.r_bot * .25 * np.sin(th_bot + np.pi/2.))
    x_right = (x_bot + world.r_bot * .7 * np.cos(th_bot) +
               world.r_bot * .25 * np.cos(th_bot - np.pi/2.))
    y_right = (y_bot + world.r_bot * .7 * np.sin(th_bot) +
               world.r_bot * .25 * np.sin(th_bot - np.pi/2.))
    # pupil locations
    xp_left = (x_bot + world.r_bot * .725 * np.cos(th_bot) +
               world.r_bot * .248 * np.cos(th_bot + np.pi/2.))
    yp_left = (y_bot + world.r_bot * .725 * np.sin(th_bot) +
               world.r_bot * .248 * np.sin(th_bot + np.pi/2.))
    xp_right = (x_bot + world.r_bot * .725 * np.cos(th_bot) +
                world.r_bot * .248 * np.cos(th_bot - np.pi/2.))
    yp_right = (y_bot + world.r_bot * .725 * np.sin(th_bot) +
                world.r_bot * .248 * np.sin(th_bot - np.pi/2.))
    axis.add_patch(patches.Circle((x_left, y_left),
                                  world.r_bot * .1,
                                  color=eye_color,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((xp_left, yp_left),
                                  world.r_bot * .06,
                                  color=tools.copper_shadow,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((x_left, y_left),
                                  world.r_bot * .1,
                                  color=tools.copper_shadow,
                                  linewidth=lw, fill=False,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((x_right, y_right),
                                  world.r_bot * .1,
                                  color=eye_color,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((xp_right, yp_right),
                                  world.r_bot * .06,
                                  color=tools.copper_shadow,
                                  alpha=alpha, zorder=-dzorder))
    axis.add_patch(patches.Circle((x_right, y_right),
                                  world.r_bot * .1,
                                  color=tools.copper_shadow,
                                  linewidth=lw, fill=False,
                                  alpha=alpha, zorder=-dzorder))


def plot_sensors(world, axis, x_bot, y_bot, th_bot, lw=.5):
    """
    Visually represent what the sensors are detecting around the robot.
    """
    # Show sensors visually.
    # ball range sensor
    max_alpha = .3
    for i_vision_range in np.nonzero(world.v_range)[0]:
        magnitude = world.v_range[i_vision_range]
        i_range = np.minimum(i_vision_range, world.n_ball_range - 1)
        range_radius = world.r_bot + world.ball_range_bins[i_range]
        alpha = np.minimum(1., magnitude * max_alpha)
        axis.add_patch(patches.Circle((x_bot, y_bot), range_radius,
                                      color=tools.oxide,
                                      alpha=alpha,
                                      linewidth=10., fill=False))

    # ball heading sensors
    for i_vision_heading in np.nonzero(world.v_heading)[0]:
        magnitude = world.v_heading[i_vision_heading]
        heading_sensor_radius = world.width + world.depth
        d_heading = 2. * np.pi / world.n_ball_heading
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
                      2. * np.pi / world.n_prox_heading)):
        for i_range in np.where(world.prox[i_prox, :] > 0)[0]:
            magnitude = world.prox[i_prox, i_range]
            i_prox_range = np.minimum(i_range, world.n_prox_range - 1)
            prox_range = world.r_bot + world.prox_range_bins[i_prox_range]
            prox_angle = th_bot - prox_theta
            x_pos = x_bot + np.cos(prox_angle) * prox_range
            y_pos = y_bot + np.sin(prox_angle) * prox_range
            prox_sensor_radius = world.r_bot / 10.
            alpha = np.minimum(1., magnitude * max_alpha)
            axis.add_patch(patches.Circle((x_pos, y_pos), prox_sensor_radius,
                                          color=tools.dark_copper,
                                          alpha=alpha,
                                          linewidth=0., fill=True))
            plt.plot([x_bot, x_pos], [y_bot, y_pos],
                     color=tools.copper, linewidth=.5,
                     alpha=alpha,
                     zorder=-10)

    # bump sensors
    max_alpha = .8
    for (i_bump, bump_theta) in enumerate(
            np.arange(0., 2 * np.pi, 2 * np.pi / world.n_bump_heading)):
        bump_angle = th_bot - bump_theta
        x_pos = x_bot + np.cos(bump_angle) * world.r_bot
        y_pos = y_bot + np.sin(bump_angle) * world.r_bot
        for i_mag in np.where(world.bump[i_bump, :] > 0)[0]:
            magnitude = np.minimum(1., world.bump[i_bump, i_mag])

            bump_sensor_radius = ((world.r_bot * i_mag) /
                                  (2. * world.n_bump_mag))
            bump_sensor_radius = np.maximum(0., bump_sensor_radius)
            alpha = np.minimum(1., magnitude * max_alpha)
            axis.add_patch(patches.Circle(
                (x_pos, y_pos), bump_sensor_radius,
                color=tools.copper_shadow,
                alpha=alpha, linewidth=0., fill=True))


    # speed and acceleration sensors
    scale = .1
    dx_pos = world.vx_bot * scale
    dy_pos = world.vy_bot * scale
    dth = world.omega_bot * scale
    plot_robot(world, axis, x_bot - dx_pos, y_bot - dy_pos, th_bot - dth,
               alpha=.3, dzorder=13)
    ddx_pos = dx_pos - world.ax_bot * scale ** 2
    ddy_pos = dy_pos - world.ay_bot * scale ** 2
    ddth = dth - world.alpha_bot * scale ** 2
    plot_robot(world, axis, x_bot - ddx_pos, y_bot - ddy_pos, th_bot - ddth,
               alpha=.15, dzorder=16)


def plot_actions(world, axis, x_bot, y_bot, th_bot, lw=.5):
    """
    Visually represent the actions. Assume they are either 0 or 1.
    """
    delta = .1
    # Forward motion.
    for i_fwd in xrange(0,5):
        if world.actions[i_fwd] > 0.: 
            y_0_unrotated = (i_fwd * delta) + y_bot
            y_1_unrotated = y_0_unrotated
            x_0_unrotated = world.r_bot + x_bot + delta
            x_1_unrotated = x_0_unrotated + delta * 2 ** i_fwd
            x_0 = (x_0_unrotated * np.cos(th_bot) -
                   y_0_unrotated * np.sin(th_bot))
            y_0 = (x_0_unrotated * np.sin(th_bot) +
                   y_0_unrotated * np.cos(th_bot))
            x_1 = (x_1_unrotated * np.cos(th_bot) -
                   y_1_unrotated * np.sin(th_bot))
            y_1 = (x_1_unrotated * np.sin(th_bot) +
                   y_1_unrotated * np.cos(th_bot))
            plt.plot(
                [x_0, x_1],
                [y_0, y_1],
                color=tools.copper_shadow,
                linewidth=.2)

    # Reverse motion.
    for i_rev in xrange(5, 10):
        if world.actions[i_rev] > 0.: 
            i_delta = i_rev - 5
            y_0_unrotated = (i_delta * delta) + y_bot
            y_1_unrotated = y_0_unrotated
            x_0_unrotated = x_bot - world.r_bot - delta
            x_1_unrotated = x_0_unrotated - delta * 2 ** i_delta
            x_0 = (x_0_unrotated * np.cos(th_bot) -
                   y_0_unrotated * np.sin(th_bot))
            y_0 = (x_0_unrotated * np.sin(th_bot) +
                   y_0_unrotated * np.cos(th_bot))
            x_1 = (x_1_unrotated * np.cos(th_bot) -
                   y_1_unrotated * np.sin(th_bot))
            y_1 = (x_1_unrotated * np.sin(th_bot) +
                   y_1_unrotated * np.cos(th_bot))
            plt.plot(
                [x_0, x_1],
                [y_0, y_1],
                color=tools.copper_shadow,
                linewidth=.2)

    #  Counterclockwise motion.
    for i_ccw in xrange(10, 15):
        if world.actions[i_ccw] > 0.: 
            i_delta = i_ccw - 10
            x_0_unrotated = (i_delta * delta) + x_bot
            x_1_unrotated = x_0_unrotated
            y_0_unrotated = world.r_bot + y_bot + delta
            y_1_unrotated = y_0_unrotated + delta * 2 ** i_delta
            x_0 = (x_0_unrotated * np.cos(th_bot) -
                   y_0_unrotated * np.sin(th_bot))
            y_0 = (x_0_unrotated * np.sin(th_bot) +
                   y_0_unrotated * np.cos(th_bot))
            x_1 = (x_1_unrotated * np.cos(th_bot) -
                   y_1_unrotated * np.sin(th_bot))
            y_1 = (x_1_unrotated * np.sin(th_bot) +
                   y_1_unrotated * np.cos(th_bot))
            plt.plot(
                [x_0, x_1],
                [y_0, y_1],
                color=tools.copper_shadow,
                linewidth=.2)

    #  Clockwise motion.
    for i_cw in xrange(15, 20):
        if world.actions[i_cw] > 0.: 
            i_delta = i_cw - 15
            x_0_unrotated = (i_delta * delta) + x_bot
            x_1_unrotated = x_0_unrotated
            y_0_unrotated = y_bot - world.r_bot - delta
            y_1_unrotated = y_0_unrotated - delta * 2 ** i_delta
            x_0 = (x_0_unrotated * np.cos(th_bot) -
                   y_0_unrotated * np.sin(th_bot))
            y_0 = (x_0_unrotated * np.sin(th_bot) +
                   y_0_unrotated * np.cos(th_bot))
            x_1 = (x_1_unrotated * np.cos(th_bot) -
                   y_1_unrotated * np.sin(th_bot))
            y_1 = (x_1_unrotated * np.sin(th_bot) +
                   y_1_unrotated * np.cos(th_bot))
            plt.plot(
                [x_0, x_1],
                [y_0, y_1],
                color=tools.copper_shadow,
                linewidth=.2)

def render(world, dpi=80):
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
    fig, axis = plt.subplots(num=83, figsize=(world.width, world.depth))

    # The walls
    plt.plot(np.array([0., world.width, world.width, 0., 0.]),
             np.array([0., 0., world.depth, world.depth, 0.]),
             linewidth=10, color=tools.copper_shadow)
    # The floor
    axis.fill([0., world.width, world.width, 0., 0.],
              [0., 0., world.depth, world.depth, 0.],
              color=tools.light_copper, zorder=-100)
    for x_pos in np.arange(1., world.width):
        plt.plot(np.array([x_pos, x_pos]), np.array([0., world.depth]),
                 linewidth=2, color=tools.copper_highlight,
                 zorder=-99)
    for y_pos in np.arange(1., world.depth):
        plt.plot(np.array([0., world.width]), np.array([y_pos, y_pos]),
                 linewidth=2, color=tools.copper_highlight,
                 zorder=-99)
    # The ball
    axis.add_patch(patches.Circle((world.x_ball, world. y_ball),
                                  world.r_ball, color=tools.oxide))
    axis.add_patch(patches.Circle((world.x_ball, world. y_ball),
                                  world.r_ball,
                                  color=tools.copper_shadow,
                                  linewidth=2., fill=False))

    plot_robot(world, axis, world.x_bot, world.y_bot, world.th_bot)
    plot_sensors(world, axis, world.x_bot, world.y_bot, world.th_bot)

    plt.axis('equal')
    plt.axis('off')
    # Make sure the walls don't get clipped.
    plt.ylim((-.1, world.depth + .1))
    plt.xlim((-.1, world.width + .1))
    fig.canvas.draw()
    # Save the image.
    filename = ''.join([world.name, '_', str(world.frame_counter), '.png'])
    full_filename = os.path.join(world.frames_directory, filename)
    world.frame_counter += 1
    facecolor = fig.get_facecolor()
    plt.savefig(full_filename, format='png', dpi=dpi,
                facecolor=facecolor, edgecolor='none')


def render_sensors_actions(world, sensors, actions):
    """
    Turn this set of sensors and actions for the chase world into an image.

    Parameters
    ----------
    actions : array of floats
        The set of actions to render. Assumed to be between 0 and 1.
    sensors : array of floats
        The set of sensor values to render. Assumed to be between 0 and 1.
    world : The chase World
        The world to be rendered.
    """
    # Save out a copy of the current sensor and action values.
    saved_sensors = world.sensors
    world.sensor = sensors
    world.convert_sensors_to_detectors(sensors)
    saved_actions = world.action
    world.actions = actions

    axis = plt.gca()
    plot_robot(world, axis, 0., 0., np.pi/2)
    plot_sensors(world, axis, 0., 0., np.pi/2)
    plot_actions(world, axis, 0., 0., np.pi/2)
    plt.axis('equal')
    plt.axis('off')
    plt.ylim((-world.depth, world.depth))
    plt.xlim((-world.width, world.width))
    action_str = 'actions' + np.array_str(np.where(actions > 0.)[0])
    sensor_str = 'sensors' + np.array_str(np.where(sensors > 0.)[0])
    plt.text(-.9 * world.width, -.7 * world.depth, action_str, fontsize=3) 
    plt.text(-.9 * world.width, -.9 * world.depth, sensor_str, fontsize=3) 

    # Restore the saved sensor and action values.
    world.sensors = saved_sensors
    world.convert_sensors_to_detectors(world.sensors)
    world.actions = saved_actions


def visualize(world, brain):
    """
    Show what's going on in the world.
    """
    #if (world.timestep % world.timesteps_per_frame) != 0:
    #    return
    timestr = tools.timestr(world.clock_time, s_per_step=1.)
    print(' '.join(['world running for', timestr]))
    render(world)

    '''
    # Periodcally show the entire feature set.
    if world.plot_feature_set:
        feature_set = ft.get_feature_set(brain)
        for i_level, level_features in enumerate(feature_set):
            print('Features in level', i_level)
            for i_feature, feature in enumerate(level_features):
                print('Feature', i_feature, 'level', i_level)
                fig = plt.figure(num=99)
                fig.clf()
                fig, axarr = plt.subplots(1, 2 ** (i_level + 1),
                                          num=99, figsize=(3 * world.width,
                                                           3 * world.depth))
                for i_snap, snap in enumerate(feature):
                    print('    ', i_snap, ':', np.where(snap > 0)[0])
                    axis = axarr[i_snap]
                    # Convert projection to sensor activities
                    world.convert_sensors_to_detectors(snap)
                    plot_robot(world, axis, 0., 0., np.pi/2)
                    plot_sensors(world, axis, 0., 0., np.pi/2)

                for axis in axarr:
                    plt.sca(axis)
                    plt.axis('equal')
                    plt.axis('off')
                    plt.ylim((-world.depth, world.depth))
                    plt.xlim((-world.width, world.width))

                fig.canvas.draw()

                filename = '_'.join(('level', str(i_level).zfill(2),
                                     'sequence', str(i_feature).zfill(4),
                                     world.name, 'world.png'))
                full_filename = os.path.join(world.features_directory,
                                             filename)
                plt.title(filename)
                plt.savefig(full_filename, format='png')
    '''
