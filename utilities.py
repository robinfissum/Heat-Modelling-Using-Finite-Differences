# this file contains arbitrary helper functions for the main programs
import numpy as np
import math
from room import Room
from constants import OUTSIDE_TEMP_REALISTIC, diffusivity_of_air
import scipy.interpolate
import time


def avg_temp(matrix):
    """
    matrix: (ndarray) the temperature matrix of the room
    returns: (number) average value of the matrix/ average temperature in the room
    """
    return np.average(matrix)


def avg_temp_deviation(matrix, ideal_temp):
    """
    matrix: (ndarray) the temperature matrix of the room
    ideal: (number) the ideal temperature of the room
    returns: (number) the average deviation of each point in the room from the ideal temperature
    The result should be scaled before it is interpreted.
    """
    return np.sum(np.abs(matrix - ideal_temp))


def give_sortable_name(time_steps, t):
    """
    This functions should be used when you are generating many images that should be in sortable order alphanumerically
    timesteps: the total number names you want to generate
    t: the idx of the current item you are at
    returns: (string) a name, with a certain number of 0s prefixed to make the names sortable
    """
    total_digits = math.floor(math.log10(time_steps))
    used_digits = 0 if t == 0 else math.floor(math.log10(t))
    return (total_digits-used_digits)*"0" + str(t)


def analytic_solution_dirichlet_test(room):
    """
    This function returns the steady state solution to the system div(grad(T))=0 restricted
    to T=0 on the east, west, up and down boundaries on the room, and
    T(x,0,z) = sin(pi x /L)sin(pi z / H)
    T(x,W,z) = sin(pi x /L)sin(pi z / H).
    This function is used when running the solver from 'test_steady_state_dirichlet_bc'
    in 'numerical_tests.py'
    """
    dx = room.DX
    lambda_ = (math.pi/room.ROOM_LENGTH)**2 + (math.pi/room.ROOM_HEIGHT)**2
    analytic_solution = np.zeros(
        (room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS))
    for i in range(room.LENGTH_STEPS):
        for j in range(room.WIDTH_STEPS):
            for k in range(room.HEIGHT_STEPS):
                numerator = math.sinh(
                    (room.ROOM_WIDTH-j*dx)*math.sqrt(lambda_)) + math.sinh(j*dx*math.sqrt(lambda_))
                sine_factors = math.sin(
                    math.pi * i / (room.LENGTH_STEPS - 1)) * math.sin(math.pi * k / (room.HEIGHT_STEPS - 1))
                analytic_solution[i, j, k] = numerator * sine_factors / \
                    math.sinh(room.ROOM_HEIGHT*math.sqrt(lambda_))
    return analytic_solution


# TRANGE is the time interval in seconds split into 144 equal parts.
TRANGE = np.linspace(0, 86400, 145)
OUTSIDE_TEMPP = np.array([-11.8, -12, -12, -12.2, -12.4,
                          -12.3, -12.2, -12.3, -12.5, -12.6,
                          -12.7, -12.8, -13.1, -13, -13.4,
                          -13.4, -13.5, -13.3, -13.6, -13.3,
                          -13.9, -14.9, -15.6, -16, -17.2,
                          -17.3, -17.8, -17.3, -17.3, -17.3,
                          -17.5, -17.7, -17.9, -18.1, -18.2,
                          -18.5, -19.2, -19.4, -19.4, -19.2,
                          -19.3, -19.6, -19.5, -19.1, -19.1,
                          -19.7, -19.4, -19.5, -20.3, -20.2,
                          -20.4, -20.8, -21.1, -21, -20.5,
                          -20.1, -20.3, -20.3, -20, -19.7,
                          -18.7, -18.6, -18.2, -17.9, -17.8,
                          -17.5, -17.1, -16.5, -14.9, -14.6,
                          -14.8, -14.5, -13.9, -13, -12.5,
                          -12.1, -11.8, -11.4, -10.9, -10.4,
                          -9.8, -9.5, -9.3, -9.2, -9,
                          -8.8, -8.2, -8, -7.7, -7.4,
                          -7.2, -6.5, -5.8, -5.8, -5.9,
                          -6.1, -6.5, -6.5, -6.5, -6.5,
                          -6.5, -6.7, -6.5, -6.4, -6.3,
                          -6.3, -6.1, -6, -6, -6,
                          -6.1, -6, -5.9, -5.8, -5.7,
                          -5.7, -5.9, -5.8, -5.5, -5.4,
                          -4.8, -5.1, -5.1, -5.1, -5.1,
                          -5.1, -5.2, -5.1, -4.9, -4.9,
                          -5.1, -5.2, -5.3, -5.2, -5.2,
                          -5.9, -5.8, -5.4, -5.5, -5.8,
                          -5.7, -5.5, -5.4, -5.8, -6]) + 273.15


def time_step_to_actual_time(time_step_num, dt):
    return (time_step_num * dt / 3600) % 24


class AnalyticSolution:
    """
    This is an APPROXIMATION to the time dependent solution of the heat equation
    in the room when the ovens are turned off. It differs from the actual analytic solution
    in that the temperature on the boundary will always equal T_out(t)
    """
    def __init__(self, room: Room, N: int):
        self.L = room.ROOM_LENGTH
        self.W = room.ROOM_WIDTH
        self.H = room.ROOM_HEIGHT
        self.Nx = room.LENGTH_STEPS
        self.Ny = room.WIDTH_STEPS
        self.Nz = room.HEIGHT_STEPS
        self.dx = room.DX
        self.dt = room.DT
        # Todo: May be adjusted if the solution is not accurate enough.
        # N specifies how many indices we sum k,l and m over. (Run time =O(N^3))
        self.N = N
        # Matrices containing time-independent coefficients that we will need.
        # The indices of these matrices have nothing to do with the room.
        self.LAMBDA = np.zeros((self.N, self.N, self.N))
        self.D = np.zeros((self.N, self.N, self.N))
        self.C = np.zeros((self.N, self.N, self.N))
        self.alpha = diffusivity_of_air(300)
        self.splines = None
        self.sine_products = np.zeros(
            (self.Nx, self.Ny, self.Nz, self.N, self.N, self.N))
        self.setup()

    def setup(self):
        # Compute coefficients necessary for the solution. We can ignore those terms with either of
        # k, l or m even, since then the result is zero.
        print('Precomputing coefficients for the analytic solution of the time dependent heat- \n'
              'equation with Robin boundary conditions. May take a while.')
        pi = math.pi
        for k in range(1, self.N):
            for l in range(1, self.N):
                for m in range(1, self.N):
                    if k*l*m % 2 != 0:
                        self.D[k, l, m] = 75.120274001808250 * 8 / (k * l * m)
                        self.C[k, l, m] = 0.258012275465596 * 8 / (k * l * m)
                    self.LAMBDA[k, l, m] = (
                        k * pi / self.L)**2 + (l * pi / self.W)**2 + (m * pi / self.H) ** 2

        # create N^3 integrable splines
        exp = math.exp
        sin = math.sin
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        # Block array of integrable (linear) splines of size (k,l,m)
        self.splines = []

        '''
        We need to create splines that can be integrated (Last summand of T_{klm})
        k is the degree of the spline. 
        OUTSIDE_TEMPP has 145 equispaced elements over 86400 seconds. 
        Thus, the s-th element is at 86400*s/144 seconds. (s in {0,1,2,...,144}=145 elements)
        '''
        for k in range(0, N):
            self.splines.append([])
            for l in range(0, N):
                self.splines[k].append([])
                for m in range(0, N):
                    # Todo: Raises overflow err. if self.N is too big, say>15 (because exponential becomes too big)
                    # Todo: This error can be "circumvented" by choosing the room dims large(makes LAMBDAS smaller).
                    spline_values = np.array(
                        [exp(self.alpha*self.LAMBDA[k, l, m]*86400*s/144)*OUTSIDE_TEMPP[s] for s in range(145)])
                    self.splines[k][l].append(
                        scipy.interpolate.InterpolatedUnivariateSpline(TRANGE, spline_values, k=1))

        '''We precompute the terms sin(k pi x / L)sin(l pi y / W)sin(m pi z / H) and store the results in 
        a 6-dimensional numpy array 'sine_products'. If we call this array A, then 
        A[i,j,k,k2,l,m] = sin(k2*pi*i / (Nx - 1))*sin(l*pi*j/(Ny - 1))*sin(m*pi*k/(Nz - 1))
        is the (k2,l,m)-th space dependent term for the node with indices (i,j,k).
        '''
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # We are at the node p_(i,j,k). Sum up the solution terms T_{k2,l,m}*Sin(..)*Sin(..)*Sin(..)
                    for k2 in range(1, N):
                        for l in range(1, N):
                            for m in range(1, N):
                                self.sine_products[i, j, k, k2, l, m] = sin(
                                    k2*pi*i / (Nx - 1))*sin(l*pi*j/(Ny - 1))*sin(m*pi*k/(Nz - 1))

    def get(self, time_step):
        # Compute analytic solution at current time step
        exp = math.exp
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        current_out_temp_kelvin = OUTSIDE_TEMP_REALISTIC(
            time_step_to_actual_time(time_step, self.dt))
        current_time_in_seconds = time_step*self.dt
        # Matrix containing analytic solution at current time step
        solution_matrix = np.zeros((Nx, Ny, Nz))
        T_klm = np.zeros((N, N, N))

        # Loop over all coefficient indices (>0)
        for k in range(1, N):
            for l in range(1, N):
                for m in range(1, N):
                    temporary_sum = 0
                    # We compute the terms T_{klm} in the solution
                    temporary_sum += self.D[k, l, m] * \
                        exp(-self.alpha *
                            self.LAMBDA[k, l, m] * current_time_in_seconds)
                    temporary_sum -= self.C[k, l, m] * current_out_temp_kelvin

                    current_spline = self.splines[k][l][m]
                    integral_from_0_to_current_time = current_spline.integral(
                        0, current_time_in_seconds)

                    temporary_sum += self.alpha*self.C[k, l, m]*self.LAMBDA[k, l, m]*exp(
                        -self.alpha*self.LAMBDA[k, l, m]*current_time_in_seconds)*integral_from_0_to_current_time
                    T_klm[k, l, m] = temporary_sum

        # Loop over all the room nodes
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    ''' We are at node p_(i,j,k). Sum up the solution terms T_{k2,l,m}*Sin(..)*Sin(..)*Sin(..)
                    over all triples of (k2, l, m) inside the set {1,2,3,...,N}^3 
                    '''
                    solution_matrix[i, j, k] = np.sum(np.multiply(
                        T_klm, self.sine_products[i, j, k, :, :, :]))

        # We need to add the current outside temperature to the solution.
        solution_matrix += current_out_temp_kelvin

        return solution_matrix
