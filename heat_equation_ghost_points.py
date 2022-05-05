import matplotlib
import numpy as np
import csv
import math
from plots_and_gifs import gif_plot, save_plot, turn_pngs_to_gif, show_me_the_room
from utilities import give_sortable_name, avg_temp_deviation, analytic_solution_dirichlet_test, AnalyticSolution
from room import Room
from constants import OUTSIDE_TEMP0, OUTSIDE_TEMP1, OUTSIDE_TEMP_REALISTIC, IDEAL_TEMP
from constants import diffusivity_of_air, heat_capacity_of_air, thermal_conductivity_of_air, density_of_air, UValues
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


def solver(room, save_plots=False, plot_plane=False, verbose=True, rotation_angle=False, elevation_angle=False):
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
                    [t, np.abs(room.curr_temp-analytic_solution_dirichlet_test(room)).max()])

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
                      filename=f'time_step{give_sortable_name(room.TIME_STEPS, t)}', rotate=rotate, elevate=elevate, plane=plot_plane)

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
                    np.abs(room.curr_temp-current_exact_solution))
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
    Heat loss on the boundary is handled using actual ghost points. Ghost points are separated into
    six 2d-arrays named depending on which side they are of the boundary:
    east, west, north, south, up, down. The values of the ghost points are
    set on every iteration. Certain variables are defined outside of loop
    to reduce look-ups and function calls. See 'Prosjektrapport' for details.
    """
    global UVALS
    # Room properties.
    nx, ny, nz, dx, dt = room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS, room.DX, room.DT
    outside_temperature = room.get_outside_temp(time_step_to_actual_time(time_step, dt))
    kappa = thermal_conductivity_of_air(300)

    # Value of ghost points set by analytic continuation and the heat flux boundary condition.
    if room.test_mode == 'test_3d_oven_energy_output':
        # We use forward approximation of ghost points in this test.
        ghost_east = room.curr_temp[nx-1, :, :] - (dx / kappa)*np.multiply(
            room.curr_temp[nx-1, :, :]-outside_temperature, UVALS.u_east)
        ghost_west = room.curr_temp[0, :, :] - (dx / kappa) * np.multiply(
            room.curr_temp[0, :, :] - outside_temperature, UVALS.u_west)
        ghost_north = room.curr_temp[:, ny-1, :] - (dx / kappa) * np.multiply(
            room.curr_temp[:, ny-1, :] - outside_temperature, UVALS.u_north)
        ghost_south = room.curr_temp[:, 0, :] - (dx / kappa) * np.multiply(
            room.curr_temp[:, 0, :] - outside_temperature, UVALS.u_south)
        ghost_up = room.curr_temp[:, :, nz-1] - (dx / kappa) * np.multiply(
            room.curr_temp[:, :, nz-1] - outside_temperature, UVALS.u_up)
        ghost_down = room.curr_temp[:, :, 0] - (dx / kappa) * np.multiply(
            room.curr_temp[:, :, 0] - outside_temperature, UVALS.u_down)
    else:
        # Use central difference approximation to compute ghost points in normal runs(more accurate in general)
        ghost_east = room.curr_temp[nx - 2, :, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[nx - 1, :, :] - outside_temperature, UVALS.u_east)
        ghost_west = room.curr_temp[1, :, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[0, :, :] - outside_temperature, UVALS.u_west)
        ghost_north = room.curr_temp[:, ny - 2, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, ny - 1, :] - outside_temperature, UVALS.u_north)
        ghost_south = room.curr_temp[:, 1, :] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, 0, :] - outside_temperature, UVALS.u_south)
        ghost_up = room.curr_temp[:, :, nz - 2] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, :, nz - 1] - outside_temperature, UVALS.u_up)
        ghost_down = room.curr_temp[:, :, 1] - (2 * dx / kappa) * np.multiply(
            room.curr_temp[:, :, 0] - outside_temperature, UVALS.u_down)

    # Todo: If you want to define a smart oven, replace the line below.
    oven_wattage = room.get_oven_wattage(time_step)
    room_c = room.c
    # Get the thermal diffusivity and conductivity of air from table
    alpha_air = diffusivity_of_air(300)
    rho = density_of_air(300)

    '''
    This part is not in use, since diffusion alone is not "fast enough."
    # We cannot hard code the volume of the oven because of the discretization.
    volume_of_ovens = dx**3 * len(room.ovens)
    '''
    next_temp = np.zeros(np.shape(room.curr_temp))

    # Finite difference method is applied to every point in the grid.
    # When a point is 'missing' a neighbour point, the value of the
    # corresponding ghost point is used.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if room.test_mode == 'test_steady_state_dirichlet_bc' and (i in {0, nx-1} or j in {0, ny-1} or k in {0, nz - 1}):
                    next_temp[i, j, k] = room.curr_temp[i, j, k]
                    continue
                ijk = room.curr_temp[i, j, k]
                Ijk = ghost_east[j, k] if i == nx - 1 else room.curr_temp[i+1, j, k]
                i_jk = ghost_west[j, k] if i == 0 else room.curr_temp[i-1, j, k]
                iJk = ghost_north[i, k] if j == ny - 1 else room.curr_temp[i, j+1, k]
                ij_k = ghost_south[i, k] if j == 0 else room.curr_temp[i, j - 1, k]
                ijK = ghost_up[i, j] if k == nz - 1 else room.curr_temp[i, j, k+1]
                ijk_ = ghost_down[i, j] if k == 0 else room.curr_temp[i, j, k-1]

                x_diffusion = alpha_air * dt * (Ijk - 2 * ijk + i_jk) / dx ** 2
                y_diffusion = alpha_air * dt * (iJk - 2 * ijk + ij_k) / dx ** 2
                z_diffusion = alpha_air * dt * (ijK - 2 * ijk + ijk_) / dx ** 2

                # Compute the temperature increase caused by the ovens.
                oven_contribution = dt * (2 * oven_wattage) / (room_c * rho *
                                                               room.ROOM_LENGTH * room.ROOM_WIDTH * room.ROOM_HEIGHT)

                '''
                # We have commented away the ovens' geometry; heat is uniformly distributed over the entire room.
                # The wattage is multiplied by 2, since there are two ovens.
                oven_contribution = 0
                if (i, j, k) in room.ovens:
                    oven_contribution = dt*(2*oven_wattage)/(room_c * rho * volume_of_ovens)
                '''

                next_temp[i, j, k] = ijk + x_diffusion + y_diffusion + z_diffusion + oven_contribution

    room.curr_temp = next_temp


def main():
    # For easier modifications of parameters.
    c = heat_capacity_of_air(300)
    dx = 1/5
    dt = 1
    time_length = 24 * 3600
    time_steps = int(time_length / dt) + 1
    oven_wattage = 200 * np.ones(time_steps)
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

    # If you want to see the entire room in the plot.
    _, heat_deviation = solver(room, save_plots=False, plot_plane='all', verbose=True, rotation_angle=1,
                               elevation_angle=None)

    # Compute electricity cost in norwegian kroner (NOK)
    energy_used_in_joules = room.DT*np.sum(room.OVEN_WATTAGE)
    electricity_cost_nok_per_kwh = 3
    energy_used_in_kilowatt_hours = 2.78 * 10 ** (-7) * energy_used_in_joules
    energy_expense_in_nok = electricity_cost_nok_per_kwh * energy_used_in_kilowatt_hours

    # Cost functional. Did we get a good result? Lower means better for both terms.
    print(f'\nOven expense in NOK = {energy_expense_in_nok}')
    print(f'Temperature deviation  = {heat_deviation}')
    return


if __name__ == '__main__':
    main()
