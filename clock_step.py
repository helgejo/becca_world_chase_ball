""" Numba-ized routines for speeding up the chase ball world. """

import math
from numba import autojit, jit, vectorize, void, boolean, float64, int64
import numba
import random

@autojit
def bin_index(val, bins):
    """
    Find the index of the bin that val falls in.

    Assume that ``bins`` are the left bin edges are ordered from least to greatest.
    If val is smaller that the the lowest bin edge, place it in the 
    lowest bin. If it is larger than the highest edge, place it 
    in the highest bin.
    """
    for i in xrange(bins.size - 1):
        if bins[i+1] > val:
            return i
    return bins.size - 1

@autojit
def sector_index(n_sectors, theta):
    """ 
    For sector-based detectors, find the sector based on angle. 
    """
    pi = 3.415922
    theta = theta % (2. * pi)
    return int(n_sectors * theta / (2. * pi))

@autojit
def find_wall_range(theta, x_bot, y_bot, r_bot, width, depth):
    """
    Calculate the range to the nearest wall.
    """
    pi = 3.1415922
    wall_range = 1e10
    # Find distance to West wall
    range_west = x_bot / (math.cos(pi - theta) + 1e-6)
    if range_west < 0.:
        range_west = 1e10
    range_west -= r_bot
    wall_range = min(wall_range, range_west)
    # Find distance to East wall
    range_east = (width - x_bot) / (math.cos(theta) + 1e-6) 
    if range_east < 0.:
        range_east = 1e10
    range_east -= r_bot
    wall_range = min(wall_range, range_east)
    # Find distance to South wall
    range_south = y_bot / (math.sin(-theta) + 1e-6)
    if range_south < 0.:
        range_south = 1e10
    range_south -= r_bot
    wall_range = min(wall_range, range_south)
    # Find distance to North wall
    range_north = (depth - y_bot) / (math.sin(theta) + 1e-6)
    if range_north < 0.:
        range_north = 1e10
    range_north -= r_bot
    wall_range = min(wall_range, range_north)
    return wall_range

@jit(nopython=True)
def clock_step(
        clock_tick, 
        clock_time, 
        n_clockticks,
        clockticks_per_timestep,
        ball_range_bins,
        prox_range_bins,
        bump_mag_bins,
        v_fwd_bins,
        v_lat_bins,
        omega_bins,
        a_fwd_bins,
        a_lat_bins,
        alpha_bins,
        v_heading,
        v_range,
        prox,
        bump,
        v_fwd_bot_sensor,
        v_lat_bot_sensor,
        omega_bot_sensor,
        a_fwd_bot_sensor,
        a_lat_bot_sensor,
        alpha_bot_sensor,
        n_catch,
        proto_force,
        f_x_buffer,
        f_y_buffer,
        tau_buffer,
        drive,
        spin,
        x_ball,
        y_ball,
        vx_ball,
        vy_ball,
        ax_ball,
        ay_ball,
        x_bot,
        y_bot,
        th_bot,
        vx_bot,
        vy_bot,
        omega_bot,
        ax_bot,
        ay_bot,
        alpha_bot):
    """
    Advance the phyisical simulation of the world by one clock tick.
    This is at a much finer temporal granularity. 
    """
    pi = 3.1415922

    # Constants
    dt = .001 # seconds per clock tick
    drive_scale = 3.
    spin_scale = 3.

    # wall parameters
    width = 5.#16.#9. # meters
    depth = 5.#9.#16. # meters
    k_wall =  3000. # Newtons / meter

    # ball parameters
    r_ball = .4 # meters
    k_ball = 3000. # Newtons / meter
    c_ball = 1. # Newton-seconds / meter
    cc_ball = 1. # Newton
    m_ball = 1. # kilogram

    # robot parameters
    r_bot = .8 # meters
    k_bot =  3000. # Newtons / meter
    c_bot = 10. # Newton-seconds / meter
    cc_bot = 1. # Newton
    d_bot = 10. # Newton-meter-seconds / radian
    dd_bot = 1. # Newton-meter
    m_bot = 5. # kilogram
    I_bot = 1. # kilogram-meters**2
    mouth_width_bot = pi / 3. # angle over which the bot can catch the ball

    # Derived constants
    n_ball_heading = v_heading.size
    (n_bump_heading, n_bump_mag) = bump.shape
    (n_prox_heading, n_prox_mag) = prox.shape

    for _ in range(n_clockticks):
        clock_tick += 1
        clock_time = clock_tick * dt

        # Add new force profiles to the buffers
        if abs(drive) > 0.:
            drive_x = drive * math.cos(th_bot)
            drive_y = drive * math.sin(th_bot)
            for proto_index in xrange(proto_force.size):
                tick  = clock_tick + proto_index
                buffer_index = int(tick % f_x_buffer.size)
                f_x_buffer[buffer_index] += (
                    proto_force[proto_index] * drive_x * drive_scale)
                f_y_buffer[buffer_index] += (
                    proto_force[proto_index] * drive_y * drive_scale)
            drive = 0.

        if abs(spin) > 0.:
            for proto_index in xrange(proto_force.size):
                tick  = clock_tick + proto_index
                buffer_index = int(tick % tau_buffer.size)
                tau_buffer[buffer_index] += (
                    proto_force[proto_index] * spin * spin_scale)
            spin = 0.

        # grab next value from force buffers
        buffer_index = int(clock_tick % f_x_buffer.size)
        f_x_drive = f_x_buffer[buffer_index]
        f_y_drive = f_y_buffer[buffer_index]
        tau_drive = tau_buffer[buffer_index]
        f_x_buffer[buffer_index] = 0.
        f_y_buffer[buffer_index] = 0.
        tau_buffer[buffer_index] = 0.

        # contact between the robot and the ball
        delta_bot_ball = (r_ball + r_bot -
                          ((x_ball - x_bot) ** 2 + 
                           (y_ball - y_bot) ** 2) ** .5) 
        dist_bot_ball = -delta_bot_ball
        delta_bot_ball = max(delta_bot_ball, 0.)
        th_bot_ball = math.atan2(y_ball - y_bot,
                                 x_ball - x_bot)
        k_bot_ball = (k_bot * k_ball) / (k_bot + k_ball)
        # ball range detection
        i_vision_range = bin_index(dist_bot_ball, ball_range_bins)
        # ball heading detection
        # the angle of contact relative to the robot's angle
        th_bot_ball_rel = (th_bot - th_bot_ball) % (2. * pi)
        i_vision_heading = sector_index(n_ball_heading, th_bot_ball_rel)
        v_heading[i_vision_heading] += 1. / clockticks_per_timestep
        v_range[i_vision_range] += 1. / clockticks_per_timestep
        
        # ball bump detection
        if delta_bot_ball > 0.:
            i_bump_heading = sector_index(n_bump_heading, th_bot_ball_rel)
            i_bump_mag = bin_index(delta_bot_ball, bump_mag_bins)
            bump[i_bump_heading, i_bump_mag] += 1. / clockticks_per_timestep

        # calculate the forces on the ball
        # wall contact
        delta_ball_N = y_ball + r_ball - depth
        delta_ball_N = max(delta_ball_N, 0.)
        delta_ball_S = r_ball - y_ball
        delta_ball_S = max(delta_ball_S, 0.)
        delta_ball_E = x_ball + r_ball - width
        delta_ball_E = max(delta_ball_E, 0.)
        delta_ball_W = r_ball - x_ball
        delta_ball_W = max(delta_ball_W, 0.)
        k_wall_ball = (k_wall * k_ball) / (k_wall + k_ball)
        f_ball_N_y = -delta_ball_N * k_wall_ball 
        f_ball_S_y =  delta_ball_S * k_wall_ball 
        f_ball_E_x = -delta_ball_E * k_wall_ball 
        f_ball_W_x =  delta_ball_W * k_wall_ball 
        # contact with the robot
        f_ball_bot_x = delta_bot_ball * k_bot_ball * math.cos(th_bot_ball)
        f_ball_bot_y = delta_bot_ball * k_bot_ball * math.sin(th_bot_ball)
        # friction and damping
        f_ball_vx = -vx_ball * c_ball - math.copysign(cc_ball, vx_ball)
        f_ball_vy = -vy_ball * c_ball - math.copysign(cc_ball, vy_ball)

        # wall bump detection
        # Calculate the angle of contact relative to the robot's angle
        # for each of the walls. Assign the contact to the appropriate
        # bump sensor.
        delta_bot_N = y_bot + r_bot - depth
        delta_bot_N = max(delta_bot_N, 0.)
        if delta_bot_N > 0.:
            th_bot_N_rel = (th_bot - pi / 2.) % (2. * pi)
            i_bump_heading = sector_index(n_bump_heading, th_bot_N_rel)
            i_bump_mag = bin_index(delta_bot_N, bump_mag_bins)
            bump[i_bump_heading, i_bump_mag] += 1. / clockticks_per_timestep

        delta_bot_S = r_bot - y_bot
        delta_bot_S = max(delta_bot_S, 0.)
        if delta_bot_S > 0.:
            th_bot_S_rel = (th_bot + pi / 2.) % (2. * pi)
            i_bump_heading = sector_index(n_bump_heading, th_bot_S_rel)
            i_bump_mag = bin_index(delta_bot_S, bump_mag_bins)
            bump[i_bump_heading, i_bump_mag] += 1. / clockticks_per_timestep

        delta_bot_E = x_bot + r_bot - width
        delta_bot_E = max(delta_bot_E, 0.)
        if delta_bot_E > 0.:
            th_bot_E_rel = th_bot % (2. * pi)
            i_bump_heading = sector_index(n_bump_heading, th_bot_E_rel)
            i_bump_mag = bin_index(delta_bot_E, bump_mag_bins)
            bump[i_bump_heading, i_bump_mag] += 1. / clockticks_per_timestep

        delta_bot_W = r_bot - x_bot
        delta_bot_W = max(delta_bot_W, 0.)
        if delta_bot_W > 0.:
            th_bot_W_rel = (th_bot + pi) % (2. * pi)
            i_bump_heading = sector_index(n_bump_heading, th_bot_W_rel)
            i_bump_mag = bin_index(delta_bot_W, bump_mag_bins)
            bump[i_bump_heading, i_bump_mag] += 1. / clockticks_per_timestep

        # wall proximity detection
        # calculate the range detected by each proximity sensor
        #if include_prox:
        for i_prox in range(n_prox_heading): 
            prox_theta = float(i_prox) * 2 * pi / float(n_prox_heading)
            wall_range = find_wall_range(th_bot - prox_theta, 
                            x_bot, y_bot, r_bot, width, depth)
            i_prox_range = bin_index(wall_range, prox_range_bins)
            prox[i_prox, i_prox_range] += 1. / clockticks_per_timestep 

        # calculated the forces on the bot
        # wall contact
        k_wall_bot = (k_wall * k_bot) / (k_wall + k_bot)
        f_bot_N_y = -delta_bot_N * k_wall_bot 
        f_bot_S_y =  delta_bot_S * k_wall_bot 
        f_bot_E_x = -delta_bot_E * k_wall_bot 
        f_bot_W_x =  delta_bot_W * k_wall_bot 
        # contact with the robot
        f_bot_ball_x = -delta_bot_ball * k_bot_ball * math.cos(th_bot_ball)
        f_bot_ball_y = -delta_bot_ball * k_bot_ball * math.sin(th_bot_ball)
        # friction and damping, both proportional and Coulomb
        f_bot_vx = -vx_bot * c_bot - math.copysign(cc_bot, vx_bot)
        f_bot_vy = -vy_bot * c_bot - math.copysign(cc_bot, vy_bot)
        tau_bot_omega = -omega_bot * d_bot -math.copysign(dd_bot, omega_bot)

        # calculate total external forces
        f_ball_x = f_ball_E_x + f_ball_W_x + f_ball_bot_x + f_ball_vx
        f_ball_y = f_ball_N_y + f_ball_S_y + f_ball_bot_y + f_ball_vy
        f_bot_x = f_bot_E_x + f_bot_W_x + f_bot_ball_x + f_bot_vx + f_x_drive
        f_bot_y = f_bot_N_y + f_bot_S_y + f_bot_ball_y + f_bot_vy + f_y_drive
        tau_bot = tau_bot_omega + tau_drive

        # use forces to update accelerations, velocities, and positions
        # ball
        ax_ball = f_ball_x / m_ball
        ay_ball = f_ball_y / m_ball
        vx_ball += ax_ball * dt
        vy_ball += ay_ball * dt
        x_ball += vx_ball * dt
        y_ball += vy_ball * dt
        # robot
        ax_bot = f_bot_x / m_bot
        ay_bot = f_bot_y / m_bot
        alpha_bot = tau_bot / I_bot
        vx_bot += ax_bot * dt
        vy_bot += ay_bot * dt
        omega_bot+= alpha_bot * dt
        x_bot += vx_bot * dt
        y_bot += vy_bot * dt
        th_bot += omega_bot * dt
        th_bot = th_bot % (2 * pi)
        # The robot's speed in the direction of its nose
        v_fwd_bot = vx_bot * math.cos(th_bot) + vy_bot * math.sin(th_bot)
        # The robot's sideways speed (right is positive)
        v_lat_bot = vx_bot * math.sin(th_bot) - vy_bot * math.cos(th_bot)
        # The robot's speed in the direction of its nose
        a_fwd_bot = ax_bot * math.cos(th_bot) + ay_bot * math.sin(th_bot)
        # The robot's sideways speed (right is positive)
        a_lat_bot = ax_bot * math.sin(th_bot) - ay_bot * math.cos(th_bot)

        i_v_fwd = bin_index(v_fwd_bot, v_fwd_bins)
        v_fwd_bot_sensor[i_v_fwd] += 1. / clockticks_per_timestep
        i_v_lat = bin_index(v_lat_bot, v_lat_bins)
        v_lat_bot_sensor[i_v_lat] += 1. / clockticks_per_timestep
        i_omega = bin_index(omega_bot, omega_bins)
        omega_bot_sensor[i_omega] += 1. / clockticks_per_timestep
        i_a_fwd = bin_index(a_fwd_bot, a_fwd_bins)
        a_fwd_bot_sensor[i_a_fwd] += 1. / clockticks_per_timestep
        i_a_lat = bin_index(a_lat_bot, a_lat_bins)
        a_lat_bot_sensor[i_a_lat] += 1. / clockticks_per_timestep
        i_alpha = bin_index(alpha_bot, alpha_bins)
        alpha_bot_sensor[i_alpha] += 1. / clockticks_per_timestep

        # check whether the bot caught the ball
        caught = False
        
        if ( (th_bot_ball_rel < mouth_width_bot / 2.) | 
             (th_bot_ball_rel > (2. * pi - mouth_width_bot / 2.)) ):
        #if (i_vision_heading == 0) | (i_vision_heading == n_ball_heading - 1):
            #if i_vision_range == 0:
            if delta_bot_ball > 0.:
                caught = True

        if caught:
            n_catch += 1.
            # when caught, the ball jumps to a new location
            good_location = False
            while not good_location:
                x_ball = r_ball + random.random() * (
                        width - 2 * r_ball)
                y_ball = r_ball + random.random() * (
                        depth - 2 * r_ball)
                vx_ball = random.gauss(0.,1.)
                vy_ball = random.gauss(0.,1.)
                # check that the ball doesn't splinch the robot
                delta_bot_ball = (r_ball + r_bot -
                                  ((x_ball - x_bot) ** 2 + 
                                   (y_ball - y_bot) ** 2) ** .5) 
                if delta_bot_ball < 0.:
                    good_location = True

    return(
        clock_tick, 
        clock_time, 
        n_catch,
        x_ball,
        y_ball,
        vx_ball,
        vy_ball,
        ax_ball,
        ay_ball,
        x_bot,
        y_bot,
        th_bot,
        vx_bot,
        vy_bot,
        omega_bot,
        ax_bot,
        ay_bot,
        alpha_bot)
