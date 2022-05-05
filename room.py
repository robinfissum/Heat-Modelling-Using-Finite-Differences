import numpy as np
import time

"""
Too make things cleaner, there should probably be a separation of the two objects "room" and "problem instanse", where properties such as DT, time length etc are 
properties of the problem instance and not the room, but too keep it simple the room class will contain everything
"""


class Room:
    def __init__(self,
                 c,
                 DX,
                 DT,
                 TIME_LENGTH,
                 room_type="standard",
                 ROOM_STARTING_TEMP=293.15,
                 OUTSIDE_STARTING_TEMP=253.15,  # -20deg celcius
                 OUTSIDE_TEMP_FUNCTION=None,
                 IS_OVEN_OFF=False,
                 OVEN_WATTAGE=None,
                 # tuple with the dimensions of the room (length, width, height)
                 room_dims=None,
                 oven_type='3d'
                 ):

        self.c = c
        self.Q_lost = 0
        self.oven_temperature = 432.15
        self.room_type = room_type

        self.DX = DX
        self.DT = DT

        # the default room is 2x2x2 meters
        self.ROOM_LENGTH = room_dims[0] if room_dims else 2
        self.ROOM_WIDTH = room_dims[1] if room_dims else 2
        self.ROOM_HEIGHT = room_dims[2] if room_dims else 2
        self.TIME_LENGTH = TIME_LENGTH

        self.LENGTH_STEPS = int(self.ROOM_LENGTH / DX) + 1
        self.WIDTH_STEPS = int(self.ROOM_WIDTH / DX) + 1
        self.HEIGHT_STEPS = int(self.ROOM_HEIGHT / DX) + 1
        self.TIME_STEPS = int(self.TIME_LENGTH / DT) + 1

        self.ROOM_STARTING_TEMP = ROOM_STARTING_TEMP
        self.OUTSIDE_STARTING_TEMP = OUTSIDE_STARTING_TEMP
        self.OUTSIDE_TEMP_FUNCTION = OUTSIDE_TEMP_FUNCTION

        # store the properties of the door in the room
        self.DOOR_WIDTH = int(0.5 / DX)
        self.DOOR_HEIGHT = int(1 / DX)
        self.DOOR_PLACEMENT_WIDTH = int(self.WIDTH_STEPS / 2)
        self.DOOR_PLACEMENT_HEIGHT = int(self.HEIGHT_STEPS / 2)

        # store the properties of the windows in the room
        self.WINDOW_LENGTH = int(0.5 / DX)
        self.WINDOW_WIDTH = int(0.5 / DX)
        self.WINDOW_HEIGHT = int(0.5 / DX)

        self.WINDOW_PLACEMENT_LENGTH = int(self.LENGTH_STEPS / 2)
        self.WINDOW_PLACEMENT_WIDTH = int(self.WIDTH_STEPS / 2)
        self.WINDOW_PLACEMENT_HEIGHT = int(self.HEIGHT_STEPS / 2)

        # store the properties of the oven in the room. Oven height=0.6m etc.
        self.OVEN_LENGTH = int(0.60 / DX)
        self.OVEN_WIDTH = int(0.078 / DX)
        self.OVEN_HEIGHT = int(0.37 / DX)
        self.OVEN_TYPE = None
        self.A = self.OVEN_LENGTH*self.OVEN_HEIGHT  # area

        self.test_mode = None
        self.store_results_as_csv = False

        if OVEN_WATTAGE is not None:
            if len(OVEN_WATTAGE) != self.TIME_STEPS:
                raise ValueError(
                    'The number of wattage entries differs from the number of time steps!')
            else:
                self.OVEN_WATTAGE = OVEN_WATTAGE
        else:
            # this means that the oven is on the entire simulation
            self.OVEN_WATTAGE = 500*np.ones(self.TIME_STEPS)
        self.IS_OVEN_OFF = IS_OVEN_OFF

        # create a temperature matrix and somewhere to store potential doors/windows etc...
        # Use sets of tuples, since looping them are way faster.
        self.curr_temp = self.initialize_constant_starting_temp(
            ROOM_STARTING_TEMP)
        self.prev_temp = self.curr_temp
        self.door = set()
        self.windows = set()
        self.ovens = set()
        self.walls = set()

        # 'standard' room_type has boundary with windows and correct u-values etc.
        if room_type in {'standard', 'perfectly_insulated', 'poorly_insulated'}:
            self.initialize_windows()
            self.initialize_door()
            if oven_type == '2d':
                self.OVEN_TYPE = '2d'
                self.initialize_ovens()
            elif oven_type == '3d':
                self.OVEN_TYPE = '3d'
                self.initialize_3d_ovens()
            else:
                raise ValueError(
                    'Ovens must be 2d or 3d. There are no other options.')
            self.initialize_walls()

        elif room_type == "simple":  # 6 walls, no windows, no door, no oven
            self.initialize_walls()

    def get_oven_wattage(self, timestep):
        if self.IS_OVEN_OFF:
            return 0
        else:
            return self.OVEN_WATTAGE[timestep]

    def get_outside_temp(self, time_step):
        if self.OUTSIDE_TEMP_FUNCTION is None:
            raise ValueError(
                'No outside-temperature function has been specified!')
        else:
            return self.OUTSIDE_TEMP_FUNCTION(time_step)

    def initialize_constant_starting_temp(self, temp):
        return np.ones((self.LENGTH_STEPS, self.WIDTH_STEPS, self.HEIGHT_STEPS))*temp

    def initialize_windows(self):
        # print(f"wpw: {WINDOW_PLACEMENT_WIDTH}, wph: {WINDOW_PLACEMENT_HEIGHT}, wpl: {WINDOW_PLACEMENT_LENGTH}, wl: {WINDOW_WIDTH}, wh: {WINDOW_HEIGHT}, wl: {WINDOW_LENGTH}")
        for j in range(self.WINDOW_PLACEMENT_WIDTH - self.WINDOW_WIDTH, self.WINDOW_PLACEMENT_WIDTH + self.WINDOW_WIDTH):
            for k in range(self.WINDOW_PLACEMENT_HEIGHT - self.WINDOW_HEIGHT, self.WINDOW_PLACEMENT_HEIGHT + self.WINDOW_HEIGHT):
                self.windows.add((0, j, k))
        for i in range(self.WINDOW_PLACEMENT_LENGTH - self.WINDOW_LENGTH, self.WINDOW_PLACEMENT_LENGTH + self.WINDOW_LENGTH):
            for k in range(self.WINDOW_PLACEMENT_HEIGHT - self.WINDOW_HEIGHT, self.WINDOW_PLACEMENT_HEIGHT + self.WINDOW_HEIGHT):
                self.windows.add((i, 0, k))
                self.windows.add((i, self.WIDTH_STEPS-1, k))

    def initialize_ovens(self):
        for i in range(self.WINDOW_PLACEMENT_LENGTH - self.OVEN_LENGTH, self.WINDOW_PLACEMENT_LENGTH + self.OVEN_LENGTH):
            for k in range(0, self.OVEN_HEIGHT):
                self.ovens.add((i, 0, k))
                self.ovens.add((i, self.WIDTH_STEPS-1, k))
        if len(self.ovens) == 0:
            print('NOTE: Room initialized with zero oven nodes. ')
            time.sleep(3)

    def initialize_3d_ovens(self):
        """Ovens have thickness and lie 2 layers from the boundary. We make
        sure the oven is at least two nodes thick.
        Raises error if the two ovens turn out to overlap due to bad parameters.
        """
        for i in range(self.WINDOW_PLACEMENT_LENGTH - self.OVEN_LENGTH, self.WINDOW_PLACEMENT_LENGTH + self.OVEN_LENGTH):
            for k in range(2, max(3, self.OVEN_HEIGHT)+2):
                for j in range(0, max(2, self.OVEN_WIDTH)):
                    if (i, j+2, k) in self.ovens:
                        raise ValueError(
                            'Ovens were initialized on top of each other.')
                    else:
                        self.ovens.add((i, j+2, k))
                    if (i, self.WIDTH_STEPS-3-j, k) in self.ovens:
                        raise ValueError(
                            'Ovens were initialized on top of each other.')
                    else:
                        self.ovens.add((i, self.WIDTH_STEPS-3-j, k))

    def initialize_door(self):
        """ We force the door to not intersect with the floor or ceiling by demanding that
        2 <= k <= room.HEIGHT_STEPS-2. If we don't do this, then it is harder to test our code.
        """
        for j in range(self.DOOR_PLACEMENT_WIDTH - self.DOOR_WIDTH, self.DOOR_PLACEMENT_WIDTH + self.DOOR_WIDTH):
            for k in range(max(2, self.DOOR_PLACEMENT_HEIGHT - self.DOOR_HEIGHT), min(self.HEIGHT_STEPS-1, self.DOOR_PLACEMENT_HEIGHT + self.DOOR_HEIGHT)):
                self.door.add((self.LENGTH_STEPS-1, j, k))

    def initialize_walls(self):
        windows_and_stuff = self.windows.union(self.ovens, self.door)
        for j in range(0, self.WIDTH_STEPS):
            for k in range(0, self.HEIGHT_STEPS):
                if (0, j, k) not in windows_and_stuff:
                    self.walls.add((0, j, k))
                if (self.LENGTH_STEPS-1, j, k) not in windows_and_stuff:
                    self.walls.add((self.LENGTH_STEPS-1, j, k))

        for i in range(0, self.LENGTH_STEPS):
            for k in range(0, self.HEIGHT_STEPS):
                if (i, 0, k) not in windows_and_stuff:
                    self.walls.add((i, 0, k))
                if (i, self.WIDTH_STEPS-1, k) not in windows_and_stuff:
                    self.walls.add((i, self.WIDTH_STEPS-1, k))

        # Floor and ceiling
        for i in range(0, self.LENGTH_STEPS):
            for j in range(0, self.WIDTH_STEPS):
                if (i, j, 0) not in windows_and_stuff:
                    self.walls.add((i, j, 0))
                if (i, j, self.HEIGHT_STEPS-1) not in windows_and_stuff:
                    self.walls.add((i, j, self.HEIGHT_STEPS-1))

    def __str__(self):
        foo = f'>>> Room description:\nDims={(self.ROOM_LENGTH, self.ROOM_WIDTH, self.ROOM_HEIGHT)}\n' \
              f'Room nodes={(self.LENGTH_STEPS, self.WIDTH_STEPS, self.HEIGHT_STEPS)}\n' \
              f'Room type={self.room_type}\n' \
              f'c={self.c}, dx={self.DX}, dt={self.DT}\n' \
              f'Time length={self.TIME_LENGTH}\n' \
              f'Room starting temp={self.ROOM_STARTING_TEMP}\n' \
              f'Oven type={self.OVEN_TYPE}\n' \
              f'Volume ovens={self.DX**3 * len(self.ovens)}\n' \
              f'Num nodes per oven={(self.OVEN_LENGTH, self.OVEN_WIDTH, self.OVEN_HEIGHT)}'
        if self.OVEN_LENGTH*self.OVEN_WIDTH*self.OVEN_HEIGHT == 0:
            foo += '\n[Oven thickness was zero in one direction. This was manually changed.]\n\n'
        return foo
