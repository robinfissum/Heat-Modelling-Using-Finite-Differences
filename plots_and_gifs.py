import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import imageio
import math
import os
import glob

PLOT_FOLDER = "animations/"


# Saves a plot of the room nodes showing the temperature throughout the room.
def save_plot(room, length, width, height, points_per_dim, filename='placeholder', rotate=False, elevate=False, plane=None, time=None):
    """
    room: (Room) room object, whose room.curr_temp is to be plotted
    length: (number) length of the room
    width: (number) width of the room
    height: (number) height of the room
    points_per_dim: (number or 'all') number of temperatures to be show in the plot in each dimension, totally n^3 points.
                    if 'all' is passed, then all points will be show.
    filename: (string, OPTIONAL) name of plot when saved in animations/
    rotate: (integer, OPTIONAL) specifies counterclockwise rotation of plots.
    elevate (integer, OPTIONAL) specifies how much the plot is tilted/seen from above
    returns: nothing, but has the side effect of storing the plot as a png in the animations folder
    """

    if room.test_mode == 'test_steady_state_dirichlet_bc':
        matrix = room.curr_temp
    else:
        matrix = room.curr_temp - 273.15
    x_num, y_num, z_num = np.shape(matrix)
    if isinstance(points_per_dim, int):
        if not math.gcd(*matrix.shape) % points_per_dim == 0:
            raise ValueError(
                "Problematic scaling. Choose 'points_per_dim' so it divides each dimension of 'matrix'")
        else:
            n = points_per_dim
            x_skip_factor = x_num // n
            y_skip_factor = y_num // n
            z_skip_factor = z_num // n
            n1 = n2 = n3 = n
    elif isinstance(points_per_dim, str):
        if points_per_dim != 'all':
            raise ValueError(
                f'Please choose a valid "Points per dim". Points per dim was {points_per_dim}')
        else:
            x_skip_factor = y_skip_factor = z_skip_factor = 1
            n1 = x_num
            n2 = y_num
            n3 = z_num
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Ensure this is set appropriately wrt. oven-and outside temperature
    if room.test_mode == 'test_steady_state_dirichlet_bc':
        scale_colors = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    else:
        scale_colors = colors.TwoSlopeNorm(vmin=-40, vcenter=0, vmax=360)

    # har n punkter langs fra 0 til length, map til rett idx i matrix
    idxs = [[], [], []]
    temp = np.array([])

    for x in range(n1):
        for y in range(n2):
            for z in range(n3):
                x = x if not plane or (
                    plane and not "x" in plane) else plane["x"]
                y = y if not plane or (
                    plane and not "y" in plane) else plane["y"]
                z = z if not plane or (
                    plane and not "z" in plane) else plane["z"]

                idxs[0].append(length/(x_num-1) * x_skip_factor * x)
                idxs[1].append(width/(y_num-1) * y_skip_factor * y)
                idxs[2].append(height/(z_num-1) * z_skip_factor * z)
                temp = np.append(
                    temp, matrix[x_skip_factor * x, y_skip_factor * y, z_skip_factor*z])
    im = ax.scatter(idxs[0], idxs[1], idxs[2], c=temp,
                    marker='o', cmap=plt.cm.coolwarm, norm=scale_colors)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = f"{length}x{width}x{height}m room" + \
        (f", time={(time/60):.1f}min" if time is not None else "")
    plt.title(title)
    plt.colorbar(im)
    if rotate:
        ax.view_init(azim=rotate)
    if elevate:
        ax.view_init(elev=elevate)
    plt.savefig(PLOT_FOLDER + filename + ".png")
    plt.close()

# To center the heat plot such that 0 degrees get no color, positive temperatures are red, and negative blue
# Todo: v_min and v_max should be set according to the highest and lowest temperature, possibly
# Todo: such that v_min = - v_max
# See: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
# divnorm = colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)


# create gif out of temperature matrix in 3d
def gif_plot(matrix, length, width, height, points_per_dim, FRAMES=36):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # har n punkter langs fra 0 til length, map til rett idx i matrix
    n = points_per_dim
    idxs = [[], [], []]
    for x in range(n):
        for y in range(n):
            for z in range(n):
                idxs[0].append(math.floor(matrix.shape[0]*x/n))
                idxs[1].append(math.floor(matrix.shape[1]*y/n))
                idxs[2].append(math.floor(matrix.shape[2]*z/n))

    im = ax.scatter(idxs[0], idxs[1], idxs[2], c=matrix[idxs[0],
                    idxs[1], idxs[2]], marker='o', cmap=plt.cm.coolwarm)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(im)
    plt.title("Room temperature")
    filenames = [PLOT_FOLDER + str(i)+".PNG" for i in range(2*FRAMES)]
    for i in range(FRAMES):
        ax.view_init(azim=20+i)
        plt.savefig(PLOT_FOLDER + str(i) + ".png")
        plt.savefig(PLOT_FOLDER + str(2*FRAMES-1-i) + ".png")
    with imageio.get_writer('animations/movie.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in filenames:
        os.remove(filename)


def turn_pngs_to_gif():
    filenames = glob.glob("./" + PLOT_FOLDER + "*.png")
    filenames.sort()
    with imageio.get_writer(PLOT_FOLDER + "time_plot.gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in filenames:
        os.remove(filename)


def _set_axes_equal(ax: plt.Axes):
    """
    Helper function.
    Sets 3D plot axes to equal scale.
    Source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    """
    Another helper function to scale the 3D plot.
    """
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def show_me_the_room(room):
    import matplotlib
    matplotlib.use('TkAgg')
    oven_x_vals = np.array([])
    oven_y_vals = np.array([])
    oven_z_vals = np.array([])

    window_x_vals = np.array([])
    window_y_vals = np.array([])
    window_z_vals = np.array([])

    door_x_vals = np.array([])
    door_y_vals = np.array([])
    door_z_vals = np.array([])

    edges_x_vals = np.array([])
    edges_y_vals = np.array([])
    edges_z_vals = np.array([])
    for index in room.ovens:
        oven_x_vals = np.append(oven_x_vals, index[0])
        oven_y_vals = np.append(oven_y_vals, index[1])
        oven_z_vals = np.append(oven_z_vals, index[2])
    for index in room.windows:
        window_x_vals = np.append(window_x_vals, index[0])
        window_y_vals = np.append(window_y_vals, index[1])
        window_z_vals = np.append(window_z_vals, index[2])
    for index in room.door:
        door_x_vals = np.append(door_x_vals, index[0])
        door_y_vals = np.append(door_y_vals, index[1])
        door_z_vals = np.append(door_z_vals, index[2])

    # This can be done smarter, but I choose to stick to the python laziness principle.
    for i in range(room.LENGTH_STEPS):
        edges_x_vals = np.append(edges_x_vals, (i, i, i, i))
        edges_y_vals = np.append(
            edges_y_vals, (0, room.WIDTH_STEPS-1, 0, room.WIDTH_STEPS-1))
        edges_z_vals = np.append(
            edges_z_vals, (0, 0, room.HEIGHT_STEPS-1, room.HEIGHT_STEPS-1))
    for j in range(1, room.WIDTH_STEPS-1):
        edges_x_vals = np.append(
            edges_x_vals, (0, room.LENGTH_STEPS-1, room.LENGTH_STEPS-1, 0))
        edges_y_vals = np.append(edges_y_vals, (j, j, j, j))
        edges_z_vals = np.append(
            edges_z_vals, (0, 0, room.HEIGHT_STEPS - 1, room.HEIGHT_STEPS - 1))
    for k in range(1, room.HEIGHT_STEPS-1):
        edges_x_vals = np.append(
            edges_x_vals, (0, 0, room.LENGTH_STEPS - 1, room.LENGTH_STEPS-1))
        edges_y_vals = np.append(
            edges_y_vals, (0, room.WIDTH_STEPS-1, 0, room.WIDTH_STEPS-1))
        edges_z_vals = np.append(edges_z_vals, (k, k, k, k))

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')

    # Creating plot
    ax.scatter3D(door_x_vals, door_y_vals, door_z_vals, color='green')
    ax.scatter3D(oven_x_vals, oven_y_vals, oven_z_vals, color='red')
    ax.scatter3D(window_x_vals, window_y_vals, window_z_vals, color='blue')
    ax.scatter3D(edges_x_vals, edges_y_vals, edges_z_vals, color='black')
    ax.set_xlabel(f'x ({room.ROOM_LENGTH}m;{room.LENGTH_STEPS}n)')
    ax.set_ylabel(f'y ({room.ROOM_WIDTH}m;{room.WIDTH_STEPS}n)')
    ax.set_zlabel(f'z ({room.ROOM_HEIGHT}m;{room.HEIGHT_STEPS}n)')
    plt.title('$\mathcal{THE} \;\; \mathcal{ROOM}$')

    # These lines make the x,y and z axes have the same aspect ration.
    ax.set_box_aspect([1, 1, 1])
    _set_axes_equal(ax)

    # show plot
    plt.show()
