import matplotlib
import numpy as np
import csv
import math
from plots_and_gifs import gif_plot, save_plot, turn_pngs_to_gif, show_me_the_room
from utilities import give_sortable_name, avg_temp_deviation, analytic_solution_dirichlet_test, AnalyticSolution
from room import Room
from constants import OUTSIDE_TEMP0, OUTSIDE_TEMP1, OUTSIDE_TEMP_REALISTIC, IDEAL_TEMP
from constants import diffusivity_of_air, heat_capacity_of_air, thermal_conductivity_of_air, density_of_air, UValues
import time

# The next line removes the limit of 356 plots.
matplotlib.use('Agg')


def time_step_to_actual_time(time_step_num, dt):
    """
    Returns the time as a number in the interval [0,24). That is,
    the number of hours after midnight.
    """
    return (time_step_num * dt / 3600) % 24


# Object that keeps track of the U-values on the boundary of the room.
UVALS = None


def solver(room, save_plots=False, plot_plane=False, verbose=True, rotation_angle=False,
           elevation_angle=False):
    global UVALS
    UVALS = UValues(room)
    heat_deviation = avg_temp_deviation(room.curr_temp, IDEAL_TEMP(0))

    if room.test_mode == 'test_time_dependent_robin_bc':
        exact_solution = AnalyticSolution(room, 25)
        current_exact_solution = exact_solution.get(0)
        with open('TestResultsContainer/TestResults', 'a') as csv_file:
            writer = csv.writer(csv_file)
            average_exact_solution = np.average(current_exact_solution)
            average_room = np.average(room.curr_temp)
            average_deviation_from_analytic_solution = np.average(
                np.abs(room.curr_temp - current_exact_solution))
            max_deviation_from_analytic_solution = np.abs(
                room.curr_temp - current_exact_solution).max()
            writer.writerow([-1, average_room, average_exact_solution, average_deviation_from_analytic_solution,
                             max_deviation_from_analytic_solution])
            print(
                f'Avg. deviation={average_deviation_from_analytic_solution} Max deviation={max_deviation_from_analytic_solution}')

    if room.store_results_as_csv:
        # Save room average temp in celsius to CSV file for plotting against ideal temp.
        with open('TestResultsContainer/TestResults', 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([-1, np.average(room.curr_temp) - 273.15,
                             IDEAL_TEMP(time_step_to_actual_time(0, room.DT)) - 273.15])

    # Todo: For some reason the maximum amount of frames we fit in a gif is 355
    saveFrame = 1 if room.TIME_STEPS < 354 else int(
        math.ceil(room.TIME_STEPS / 354))
    plot_count = -1

    for t in range(room.TIME_STEPS):

        # This first section is reserved for testing, and will otherwise be ignored.
        if room.test_mode == 'test_temp_converges_to_outside_temp_3d_room':
            print(
                f'Start of time step {t}.Room max={np.amax(room.curr_temp)}. Room min={np.amin(room.curr_temp)}')
            with open('TestResultsContainer/TestResults', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([t, np.amax(room.curr_temp), np.amin(
                    room.curr_temp), np.average(room.curr_temp)])
        elif room.test_mode == 'test_steady_state_dirichlet_bc':
            with open('TestResultsContainer/TestResults', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [t, np.abs(room.curr_temp - analytic_solution_dirichlet_test(room)).max()])

        if verbose:
            progress_bar_text = str(round(100 * t / room.TIME_STEPS, 2))
            if len(str.split(progress_bar_text, '.')[1]) < 2:
                progress_bar_text += '0'
            print(
                f'Time step {t} of {room.TIME_STEPS}. {progress_bar_text}% complete.')
        if save_plots and t % saveFrame == 0:
            plot_count += 1
            rotate = -90 + plot_count * rotation_angle if rotation_angle is not None else False
            elevate = elevation_angle if elevation_angle is not None else False
            save_plot(room, room.ROOM_LENGTH, room.ROOM_WIDTH, room.ROOM_HEIGHT,
                      'all', time=t * room.DT,
                      filename=f'time_step{give_sortable_name(room.TIME_STEPS, t)}', rotate=rotate, elevate=elevate,
                      plane=plot_plane)

        # Do one iteration using the finite difference method
        explicit_3d_ovens(room, t)

        if room.store_results_as_csv:
            # Save room average temp and ideal temp in celsius to CSV file for plotting.
            with open('TestResultsContainer/TestResults', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([t, np.average(room.curr_temp) - 273.15,
                                 IDEAL_TEMP(time_step_to_actual_time(t, room.DT)) - 273.15])

        # Update deviation from ideal temperature
        heat_deviation += avg_temp_deviation(room.curr_temp,
                                             IDEAL_TEMP(time_step_to_actual_time(t, room.DT)))

        if room.test_mode == 'test_time_dependent_robin_bc':
            current_exact_solution = exact_solution.get(t)
            with open('TestResultsContainer/TestResults', 'a') as csv_file:
                writer = csv.writer(csv_file)
                average_exact_solution = np.average(current_exact_solution)
                average_room = np.average(room.curr_temp)
                average_deviation_from_analytic_solution = np.average(
                    np.abs(room.curr_temp - current_exact_solution))
                max_deviation_from_analytic_solution = np.abs(
                    room.curr_temp - current_exact_solution).max()
                writer.writerow([t, average_room, average_exact_solution, average_deviation_from_analytic_solution,
                                 max_deviation_from_analytic_solution])
                print(
                    f'Avg dev={average_deviation_from_analytic_solution} Max dev.={max_deviation_from_analytic_solution}')

    if save_plots:
        turn_pngs_to_gif()

    return room.curr_temp, heat_deviation


def explicit_3d_ovens(room, time_step) -> None:
    """ Solves explicitly for temperature in the next time step.
    i_ means i-1, I means i+1 (similarly for j and k)
    Heat loss on the boundary is handled using 'ghost points' (or 'fictitious nodes').
    The ghost points are named depending on which side of the boundary they correspond to:
    east, west, north, south, up, down. The values of the ghost points are
    set on every iteration. Certain variables are defined outside of loop
    to reduce look-ups and function calls.
    """
    global UVALS
    # Room properties.
    nx, ny, nz, dx, dt = room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS, room.DX, room.DT
    outside_temperature = room.get_outside_temp(time_step_to_actual_time(time_step, dt))
    kappa = thermal_conductivity_of_air(300)

    # Todo: This implementation in not compatible with 'test_diriclet_bc'
    # Embed the room temperature into a matrix with an extra layer
    # This extra layer will be used to store ghost points
    # Not all points in this extra layer will be ghost points, some will just be zero.
    ex_temp = np.zeros((nx + 2, ny + 2, nz + 2))
    ex_temp[1:nx + 1, 1:ny + 1, 1:nz + 1] = room.curr_temp

    # Value of ghost points set by analytic continuation and the heat flux boundary condition.
    if room.test_mode == 'test_3d_oven_energy_output':
        # We use forward approximation of ghost points in this test.
        # Ghost points EAST boundary
        ex_temp[nx + 1, 1:ny + 1, 1:nz + 1] = room.curr_temp[nx - 1, :, :] - (dx / kappa) * np.multiply(
            room.curr_temp[nx - 1, :, :] - outside_temperature, UVALS.u_east)
        # Ghost_points WEST boundary
        ex_temp[0, 1:ny + 1, 1:nz + 1] = room.curr_temp[0, :, :] - (dx / kappa) * np.multiply(
            room.curr_temp[0, :, :] - outside_temperature, UVALS.u_west)
        # Ghost_points NORTH boundary
        ex_temp[1:nx + 1, ny + 1, 1:nz + 1] = room.curr_temp[:, ny - 1, :] - (dx / kappa) * np.multiply(
            room.curr_temp[:, ny - 1, :] - outside_temperature, UVALS.u_north)
        # Ghost_points SOUTH boundary
        ex_temp[1:nx + 1, 0, 1:nz + 1] = room.curr_temp[:, 0, :] - (dx / kappa) * np.multiply(
            room.curr_temp[:, 0, :] - outside_temperature, UVALS.u_south)
        # Ghost_points UP boundary
        ex_temp[1:nx + 1, 1:ny + 1, nz + 1] = room.curr_temp[:, :, nz - 1] - (dx / kappa) * np.multiply(
            room.curr_temp[:, :, nz - 1] - outside_temperature, UVALS.u_up)
        # Ghost_points DOWN boundary
        ex_temp[1:nx + 1, 1:ny + 1, 0] = room.curr_temp[:, :, 0] - (dx / kappa) * np.multiply(
            room.curr_temp[:, :, 0] - outside_temperature, UVALS.u_down)
    else:
        # Use central difference approximation to compute ghost points in normal runs(more accurate in general)
        # Ghost points EAST boundary
        ex_temp[nx + 1, 1:ny + 1, 1:nz + 1] = room.curr_temp[nx - 2, :, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[nx - 1, :, :] - outside_temperature, UVALS.u_east)
        # Ghost_points WEST boundary
        ex_temp[0, 1:ny + 1, 1:nz + 1] = room.curr_temp[1, :, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[0, :, :] - outside_temperature, UVALS.u_west)
        # Ghost_points NORTH boundary
        ex_temp[1:nx + 1, ny + 1, 1:nz + 1] = room.curr_temp[:, ny - 2, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, ny - 1, :] - outside_temperature, UVALS.u_north)
        # Ghost_points SOUTH boundary
        ex_temp[1:nx + 1, 0, 1:nz + 1] = room.curr_temp[:, 1, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, 0, :] - outside_temperature, UVALS.u_south)
        # Ghost_points UP boundary
        ex_temp[1:nx + 1, 1:ny + 1, nz + 1] = room.curr_temp[:, :, nz - 2] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, :, nz - 1] - outside_temperature, UVALS.u_up)
        # Ghost_points DOWN boundary
        ex_temp[1:nx + 1, 1:ny + 1, 0] = room.curr_temp[:, :, 1] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, :, 0] - outside_temperature, UVALS.u_down)

    # Get the heat_capacity, diffusivity and conductivity of air
    room_c = room.c
    alpha_air = diffusivity_of_air(300)
    rho = density_of_air(300)

    # These are the constants in the equation
    # T_{ijk}^{m+1} = A*T_{ijk}^m + B*[ T_{Ijk}^m + T_{i_jk}^m + T_{iJk}^m+T_{ij_k}^m+T_{ijK}^m + T_{ijk_}^m ]
    B = alpha_air * dt / dx ** 2
    A = 1 - 6 * B
    # Apply finite difference method to all points in the central block at once
    ex_temp[1:nx + 1, 1:ny + 1, 1:nz + 1] = A * ex_temp[1:nx + 1, 1:ny + 1, 1:nz + 1] + \
                                            B * (ex_temp[0:nx, 1:ny + 1, 1:nz + 1] + ex_temp[2:nx + 2, 1:ny + 1, 1:nz + 1]
                                              + ex_temp[1:nx + 1, 0:ny, 1:nz + 1] + ex_temp[1:nx + 1, 2:ny + 2, 1:nz + 1]
                                              + ex_temp[1:nx + 1, 1:ny + 1, 0:nz] + ex_temp[1:nx + 1, 1:ny + 1, 2:nz + 2])

    room.curr_temp = ex_temp[1:nx + 1, 1:ny + 1, 1:nz + 1]

    # Compute the temperature increase in EVERY ROOM NODE due to the oven output
    # Todo: If you want to define a smarter oven, replace the line below.
    oven_wattage = room.get_oven_wattage(time_step)
    oven_contribution = dt * (2 * oven_wattage) / (room_c * rho *
                                                   room.ROOM_LENGTH * room.ROOM_WIDTH * room.ROOM_HEIGHT)
    room.curr_temp += oven_contribution


def main():
    # For easier modifications of parameters.
    c = heat_capacity_of_air(300)
    dx = 1 / 5
    dt = 1
    time_length = 24 * 3600
    time_steps = int(time_length / dt) + 1
    oven_wattage = 170 * np.ones(time_steps)
    room_starting_temp = IDEAL_TEMP(0)
    room_dimensions = (4, 6, 2.4)
    type_of_room = 'standard'
    type_of_oven = '3d'
    outside_temp_function = OUTSIDE_TEMP_REALISTIC

    room = Room(c=c, DX=dx, DT=dt, TIME_LENGTH=time_length, ROOM_STARTING_TEMP=room_starting_temp,
                room_dims=room_dimensions, room_type=type_of_room, OUTSIDE_TEMP_FUNCTION=outside_temp_function,
                OVEN_WATTAGE=oven_wattage, oven_type=type_of_oven)

    # To save data containing average temperature and ideal temperature to /TestResultsContainer/TestResults
    room.store_results_as_csv = False

    # Verify that the room is not contradictory.
    show_me_the_room(room)
    print(room)

    # For plotting the temperature through a plane cut through the room
    # _, heat_deviation = solver(room, save_plots=True, plot_plane={"z": 2}, mode='explicit_3d_ovens', verbose=True,
    #                           rotation_angle=0, elevation_angle=None)
    time1 = time.time()
    # If you want to see the entire room in the plot.
    # put save_plots=True if you want to create a time plot of the entire room in /animations/
    # Note that save_plots=True makes the code run somewhat slower.
    _, heat_deviation = solver(room, save_plots=False, plot_plane='all', verbose=True, rotation_angle=1,
                               elevation_angle=None)

    # Compute electricity cost in norwegian kroner (NOK)
    energy_used_in_joules = room.DT * np.sum(room.OVEN_WATTAGE)
    electricity_cost_nok_per_kwh = 3
    energy_used_in_kilowatt_hours = 2.78 * 10 ** (-7) * energy_used_in_joules
    energy_expense_in_nok = electricity_cost_nok_per_kwh * energy_used_in_kilowatt_hours

    # Cost functional. Did we get a good result? Lower means better for both terms.
    print(f'\nOven expense in NOK = {energy_expense_in_nok}')
    print(f'Temperature deviation  = {heat_deviation}')
    print(f'Computation took {time.time() - time1} seconds.')
    return


if __name__ == '__main__':
    main()
