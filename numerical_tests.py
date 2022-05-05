from room import Room
from utilities import avg_temp
from constants import U_wall, OUTSIDE_TEMP_REALISTIC
from constants import heat_capacity_of_air, density_of_air
from heat_equation_ghost_points import solver
import numpy as np
import math
from asyncio import constants
from os import environ
environ["execution_mode"] = "TEST"


def run_tests(verbose):
    print("--------------------------------------------------------")
    print("RUNNING TESTING SUITE")

    # from heat_equation_ghost_points import debug
    # DT = 1/200
    # debug()
    # environ["TEST"] = "400th"
    # from heat_equation_ghost_points import debug
    # DT = 1/400
    # debug()
    # Todo: The tests are currently using heat_equation_ghost_points
    # Todo: Adapt to using the faster heat_equation_ghost_points_FAST
    cp = heat_capacity_of_air(300)
    test_steady_state_dirichlet_bc(heat_capacity_c=cp, verbose=True)
    test_time_dependent_robin_bc(heat_capacity_c=cp, verbose=True)
    test_3d_room(heat_capacity_c=cp, verbose=verbose)
    test_constant_temperature_3d_room(heat_capacity_c=cp, verbose=True)
    test_temp_converges_to_outside_temp_3d_room(heat_capacity_c=cp, verbose=True)
    test_3d_oven_energy_output(heat_capacity_c=cp, time_steps=50, verbose=True)
    test_boundary_conditions_3d_ovens(heat_capacity_c=cp, verbose=True)
    test_energy_conservation(cp, verbose=True)
    test_energy_conservation(cp, verbose=True)
    solver(False, 'implicit', False)


def test_3d_room(heat_capacity_c, verbose):
    """
    Tests if the 3d room makes sense, i.e. if the
    windows and the door lie in the interior of the walls
    (not outside the wall, and do not touch the corners of the room)
    """
    print("\nRUNNING 'test_3d_room'")
    temp = 273.15
    def outside_temp(t): return temp
    dx = 1 / 5
    # one time step setup
    dt = 1 / 10
    time_length = 1 / 10
    oven_wattage = 1000 * np.ones(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(2, 2, 2), room_type='standard',
                ROOM_STARTING_TEMP=temp, OUTSIDE_STARTING_TEMP=temp, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    for index in room.windows:
        if index[2] == 0:
            print(f'Warning: room.window has a down-boundary node {index}.')
        if index[2] == room.HEIGHT_STEPS-1:
            print(f'Warning: room.window has a up-boundary node {index}.')
        if index[0] == 0 and index[1] == 0:
            print(f'Warning: room.window has south-west edge node {index}.')
        if index[0] == 0 and index[1] == room.WIDTH_STEPS-1:
            print(f'Warning: room.window has north-west edge node {index}.')
        if index[0] == room.LENGTH_STEPS:
            print(f'Warning: room.window has a east-boundary node {index}')

    for index in room.door:
        if index[0] != room.LENGTH_STEPS-1:
            print('Warning: Door has node (i,j,k) with i!=room.LENGTH_STEPS-1')
        if index[1] in {0, room.WIDTH_STEPS-1}:
            print('Warning: Door has a north or south boundary node!')
        if index[2] in {0, room.HEIGHT_STEPS-1}:
            print('Warning: Door has up or down boundary node!')
    print("[TEST COMPLETE]:'test_3d_room' (No output = PASSED)")
    print('\n')


def test_constant_temperature_3d_room(heat_capacity_c, verbose):
    """ Tests if temperature remains constant under these assumptions:
    Ovens are turned off.
    The outside temperature is constant and equal to the initial temperature.
    Both are set equal to 253.15 kelvin = -20 Celsius
    """
    print("\nRUNNING 'test_constant_temperature_3d_room'")
    temp = -500
    def outside_temp(t): return temp
    dx = 1 / 5
    dt = 1 / 10
    time_length = 10  # sec
    oven_wattage = np.zeros(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(20, 20, 5), room_type='standard',
                ROOM_STARTING_TEMP=temp, OUTSIDE_STARTING_TEMP=temp, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    final_room_temp, _ = solver(
        room, save_plots=False, verbose=False, rotation_angle=False)

    # create a 3d array containing only 'temp', and check that it is equal to final_room_temp
    expected_final_room_temp = temp * \
        np.ones((room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS))

    if np.array_equiv(final_room_temp, expected_final_room_temp):
        print("[PASSED] 'test_constant_temperature_3d_room'")
    else:
        print("[FAILED] 'test_constant_temperature_3d_room'")
    print('\n')


def test_temp_converges_to_outside_temp_3d_room(heat_capacity_c, verbose):
    """ Tests if the temperature in the room converges to the
    fixed outside temperature when the ovens are turned off.
    This test should be run over a long time interval.
    """
    print("\nRUNNING 'test_temp_converges_to_outside_temp_3d_room'")
    room_initial_temp = 293.15
    def outside_temp(t): return 173.15
    dx = 1 / 5
    dt = 1
    time_length = 43200  # sec
    oven_wattage = np.zeros(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(4, 6, 3), room_type='standard',
                ROOM_STARTING_TEMP=room_initial_temp, OUTSIDE_STARTING_TEMP=173.15, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    room.test_mode = 'test_temp_converges_to_outside_temp_3d_room'
    _, _ = solver(room, save_plots=True, verbose=False, rotation_angle=False)

    max_allowed_temp_deviation = 1
    # The test is failed if the temperature in the room deviates too much from 'outside_temp'
    room_max = np.amax(room.curr_temp)
    room_min = np.amin(room.curr_temp)
    print(
        f'Room max temp={room_max}, Room min temp={room_min}, Outside temp={173.15}')
    print(f'Max allowed temp deviation: {max_allowed_temp_deviation} kelvin.')
    if abs(room_max-173.15) > max_allowed_temp_deviation or abs(room_min-173.15) > max_allowed_temp_deviation:
        print("[FAILED] 'test_temp_converges_to_outside_temp_3d_room'")
    else:
        print("[PASSED] 'test_temp_converges_to_outside_temp_3d_room'")
    print('\n')


def test_3d_oven_energy_output(heat_capacity_c, time_steps=1, verbose=True):
    """ Checks if the oven outputs the correct amount of heat energy into the room
    during one time-step. The room is set up to have perfect insulation
    (No heat is lost through the walls at all.)
    """
    print("\nRUNNING 'test_3d_oven_energy_output'")
    # Initial and outside temperature is zero celsius
    temp = 273.15
    def outside_temp(t): return temp
    dx = 1 / 5
    dt = 1 / 10
    time_length = 1/10 * time_steps
    # 1000 Watt per oven: 2000W in total.
    oven_wattage = 1000 * np.ones(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(6, 4, 2), room_type='perfectly_insulated',
                ROOM_STARTING_TEMP=temp, OUTSIDE_STARTING_TEMP=temp, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    room.test_mode = 'test_3d_oven_energy_output'
    rho = density_of_air(300)
    initial_heat_energy = rho * room.c * dx**3 * np.sum(room.curr_temp)
    _, _ = solver(room, save_plots=False, verbose=True, rotation_angle=False)
    final_heat_energy = rho * room.c * dx**3 * np.sum(room.curr_temp)
    added_heat_energy = final_heat_energy - initial_heat_energy

    # Expected added energy= 2000W * (1/10)s * time_steps
    expected_added_heat_energy = 2000 * dt * time_steps
    print(f'number of time steps = {time_steps}')
    print(f'theoretical added energy = {expected_added_heat_energy} Joules')
    print(f'numerically added energy = {added_heat_energy} Joules')
    print(
        f'Relative miss={abs(added_heat_energy-expected_added_heat_energy)/expected_added_heat_energy}')
    print(
        f'abs(numerical - theoretical) = {abs(added_heat_energy-expected_added_heat_energy)} Joules')
    if abs(added_heat_energy-expected_added_heat_energy) < 0.5:
        print("[PASSED] 'test_3d_oven_energy_output'")
    else:
        print(
            f"[FAILED] 'test_3d_oven_energy_output'. abs(numerical - theoretical)={abs(added_heat_energy-expected_added_heat_energy)} Joules.")
    print('\n')


def test_boundary_conditions_3d_ovens(heat_capacity_c, verbose):
    """ Tests if the boundary condition works as expected over 1 time-step
    in a room with 3D ovens.
    Outside temperature set to -40 celsius==233.15 K
    The initial temperature is set to 0 celsius == 273.15 K everywhere inside the
    room. This is to minimize the effect of heat movement due to diffusion inside the room.
    The ovens are turned off (0 wattage).

    ==> The test seems to fail no matter how it is modified. Why is this??
    """
    print("\nRUNNING 'test_boundary_conditions_3d_ovens'")
    inside_temp = 273.15
    def outside_temp(t): return 233.15
    dx = 1 / 5
    # We do one time-step:
    dt = 1 / 40
    time_length = 1/40
    # Ovens are turned off!
    oven_wattage = np.zeros(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(2, 2, 2), room_type='standard',
                ROOM_STARTING_TEMP=inside_temp, OUTSIDE_STARTING_TEMP=233.15, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    '''Every node is in the center of a perfect cube of side length dx
    We have to count corner nodes multiple times; this explains the awful wall_area expression.
    The equation delta Q = UA(T-T_out)dt tells us how much heat we expect to loose: We have to sum this over the 
    walls, windows and door, since they have different U-values. 
    '''
    # Todo: The properties of air used should correspond to a temperature of: 293.15
    # Todo: The test fails for both central and forward ghost points.
    from constants import U_wall, U_window, U_door

    # Compute surface area of walls. We must count corners multiple times, but compensate for overcounting the 8 corners

    surface_area_walls = dx**2 * \
        (len(room.walls) + 4*room.HEIGHT_STEPS +
         4*room.WIDTH_STEPS+4*room.LENGTH_STEPS-8)
    surface_area_door = dx**2 * len(room.door)
    surface_area_windows = dx**2 * len(room.windows)
    expected_energy_loss_windows = surface_area_windows * \
        U_window * dt * (273.15 - 233.15)
    expected_energy_loss_door = surface_area_door * \
        U_door * dt * (273.15 - 233.15)
    expected_energy_loss_wall = surface_area_walls * \
        U_wall * dt * (273.15 - 233.15)

    '''
    # It may be that the test fails because we interpret the area elements at the nodes incorrectly.
    # Here is another interpretation, with hard-coded surface areas. [This also fails the test]
    room.test_mode = 'test_3d_oven_energy_output'
    expected_energy_loss_windows = 1.08 * U_window * dt * (273.15 - 233.15)
    expected_energy_loss_door = 0.84 * U_door * dt * (273.15 - 233.15)
    expected_energy_loss_wall = 22.08 * U_wall * dt * (273.15 - 233.15)
    '''
    rho = density_of_air(300)
    expected_energy_loss_total = expected_energy_loss_windows + \
        expected_energy_loss_door + expected_energy_loss_wall
    initial_heat_energy = rho * room.c * dx**3 * np.sum(room.curr_temp)
    _, _ = solver(room, save_plots=False, verbose=False, rotation_angle=False)
    final_heat_energy = rho * room.c * dx**3 * np.sum(room.curr_temp)
    lost_heat_energy = initial_heat_energy - final_heat_energy
    print(f'theoretical energy lost = {expected_energy_loss_total}')
    print(f'numerical energy lost = {lost_heat_energy}')
    difference = abs(expected_energy_loss_total-lost_heat_energy)
    if difference < 0.1:
        print(
            f"abs(theoretical-numerical)={abs(expected_energy_loss_total-lost_heat_energy)}")
        print(
            f"[PASSED] 'test_boundary_conditions_3d_ovens' with heat capacity = {heat_capacity_c} J/(kg K)")
    else:
        print(
            f"abs(theoretical-numerical)={abs(expected_energy_loss_total - lost_heat_energy)}")
        print(
            f"[FAILED] 'test_boundary_conditions_3d_ovens' with heat capacity = {heat_capacity_c} J/(kg K)")
    print('\n')


def test_energy_conservation(c, verbose):
    # Checks if energy is conserved.
    # Initial energy (thermal) + added energy  = final energy (thermal) + escaped energy
    # Assuming oven is at 500W at all times (except when initialized..)
    tol = 5E-2  # relative error tolerance in solutions (5e-2)=0.05=5%
    DX = 1/5
    DT = 1/20
    TIME_LENGTH = 250  # sec
    room = Room(c, DX, DT, TIME_LENGTH, room_type='simple',
                room_dims=(2, 2, 2), ROOM_STARTING_TEMP=233.15)

    result_temp, _ = solver(room, save_plots=True, verbose=False)
    for indices in room.ovens:
        result_temp[indices] -= room.oven_temperature
    rho = density_of_air(300)
    LHS = rho * room.ROOM_LENGTH * room.ROOM_WIDTH * room.ROOM_HEIGHT * \
        room.c * (avg_temp(result_temp) - room.ROOM_STARTING_TEMP)
    RHS = -room.Q_lost  # + OVEN_WATTAGE @ np.ones(OVEN_WATTAGE.shape)
    if abs((RHS - LHS)/LHS) < tol:
        print("[PASSED] ENERGY CONSERVATION TEST")
    else:
        print("[FAILED] ENERGY CONSERVATION TEST")
        if verbose:
            print('LHS', LHS, '\nRHS', RHS)


def test_steady_state_dirichlet_bc(heat_capacity_c, verbose):
    """ Checks that the steady state solution at (t=infinity) of the heat equation matches the analytic solution
    when the boundary has a fixed temperature distribution.
    In this test we have to disable the ghost points, fix the wall-node temperatures, and apply the
    finite difference scheme only to nodes inside the room that are not directly near the wall.
    Initial temp of the room should be close to 0, so that convergence does not take forever.
    """
    # Room dimensions, outside temp is irrelevant -it is not used.
    inside_temp = 0
    def outside_temp(t): return 0
    dx = 1 / 10
    dt = 1
    time_length = 86400
    # Ovens are turned off!
    oven_wattage = np.zeros(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(2, 2, 2), room_type='standard',
                ROOM_STARTING_TEMP=inside_temp, OUTSIDE_STARTING_TEMP=0, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')
    for i in range(room.LENGTH_STEPS):
        for j in range(room.WIDTH_STEPS):
            for k in range(room.HEIGHT_STEPS):
                room.curr_temp[i, j, k] = math.sin(
                    math.pi * i / (room.LENGTH_STEPS - 1)) * math.sin(math.pi * k / (room.HEIGHT_STEPS - 1))
    room.test_mode = 'test_steady_state_dirichlet_bc'
    final_room_temp, _ = solver(
        room, save_plots=True, verbose=True, rotation_angle=1)
    if verbose:
        # We compute the awful steady state solution, see 'Prosjektrapport'
        target_room_temp = np.zeros(
            (room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS))
        lambda_ = (math.pi/room.ROOM_LENGTH)**2 + (math.pi/room.ROOM_HEIGHT)**2
        for i in range(room.LENGTH_STEPS):
            for j in range(room.WIDTH_STEPS):
                for k in range(room.HEIGHT_STEPS):
                    numerator = math.sinh(
                        (room.ROOM_WIDTH-j*dx)*math.sqrt(lambda_)) + math.sinh(j*dx*math.sqrt(lambda_))
                    sine_factors = math.sin(
                        math.pi * i / (room.LENGTH_STEPS - 1)) * math.sin(math.pi * k / (room.HEIGHT_STEPS - 1))
                    target_room_temp[i, j, k] = numerator * sine_factors / \
                        math.sinh(room.ROOM_HEIGHT*math.sqrt(lambda_))
        difference = final_room_temp - target_room_temp
        print(f'Highest deviation={np.abs(difference).max()}')


def test_time_dependent_robin_bc(heat_capacity_c, verbose):
    """ IndexObjectForImplicitSchemes against exact solution of time dependent problem,
    but the walls must have the same U value everywhere
    Starting temp must equal the ideal temp!
    """
    print("Running 'test_time_dependent_robin_bc'\n")
    inside_temp = 291.15
    outside_temp = OUTSIDE_TEMP_REALISTIC
    dx = 2
    dt = 1
    time_length = 86400
    # Ovens are turned off!
    oven_wattage = np.zeros(int(time_length / dt)+1)
    room = Room(c=heat_capacity_c, DX=dx, DT=dt, TIME_LENGTH=time_length, room_dims=(20, 20, 20), room_type='standard',
                ROOM_STARTING_TEMP=inside_temp, OUTSIDE_STARTING_TEMP=0, OUTSIDE_TEMP_FUNCTION=outside_temp,
                OVEN_WATTAGE=oven_wattage, oven_type='3d')

    room.test_mode = 'test_time_dependent_robin_bc'
    _, _ = solver(room, save_plots=False, verbose=True, rotation_angle=1)


run_tests(verbose=True)
