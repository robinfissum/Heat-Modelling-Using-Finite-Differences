import os
import math
from room import Room
import numpy as np
import scipy.interpolate


# U-values.
# https://dibk.no/regelverk/byggteknisk-forskrift-tek17/14/14-3/?fbclid=IwAR0DO0ZHzi8ABSp7removoBgX-ZZJOddCcmzwflSv0adHIMIZ09lQquzxsM
# Unit: W/(m^2 K)
U_wall = 0.22
U_window = 1.0
U_door = 0.64
# Todo: Not implemented
U_ceiling = 0.16
U_floor = 0.16


# Constants related to a real panel oven.
# https://www.elektroimportoren.no/namron-panelovn-800w-matt-hvit/5401376/Product.html?Referer=googleshopping&gclid=Cj0KCQiA09eQBhCxARIsAAYRiymPVol3Gd_826wwcO9uD0YKjehJMIzSwfhSBkw1cHlk3RpWDSOTWB4aAluDEALw_wcB&gclsrc=aw.ds
Q = 800
hc = 10


# Outside temperatures in Kelvin. I.e., real valued functions F:[0,24) -> R.
def OUTSIDE_TEMP0(time):
    return -20 + 273


def OUTSIDE_TEMP1(time):
    """ Piecewise linear outside temperature.
    Assuming that the sunrise is about 7 o'clock, and that the temperature is
    at its highest (-8 celsius) at 13:00. The temperature is -20 celsius at 20:00.
    """
    if time < 7:
        return -20 + 273
    elif time < 13:
        return -20 + (time - 7) * 2 + 273
    elif time < 18:
        return -8 - (time - 13) + 273
    elif time < 22:
        return -13 - (time - 18) * 7 / 4 + 273
    else:
        return -20 + 273


'''
Fra https://seklima.met.no/observations/
(Norsk klimaservicesenter; dette er databasen som Yr.no bruker)
og viser lufttemperaturen hvert 10'ende minutt (målt 2m over bakken)
på værstasjonen FV30 Rugldalen, Røros (SN67890), 700moh, 03.02.2022
(Dette var døgnet med lavest minimumstemperatur i februar 2022)
Funksjonen beregnes med lineær interpolasjon. 
'''
OUTSIDE_TEMP_REALISTIC = scipy.interpolate.interp1d(np.linspace(0, 24, 145),
                                                    np.array([-11.8, -12, -12, -12.2, -12.4,
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
                                                              -5.7, -5.5, -5.4, -5.8, -6]) + 273.15)


def IDEAL_TEMP(time):
    """ Returns the ideal temperature inside the room, in Kelvin,
    'time' hours after midnight 00:00.
    """
    if time > 23 or time < 7:
        return 18 + 273.15
    else:
        return 22 + 273.15


'''
Properties of air as a function of temperature. Interpolated using cubic splines.
The cubic spline can be used as a function, e.g.: diffusivity_of_air(temp_in_kelvin) 
These values are taken from this table:  
http://www.thermalfluidscentral.org/encyclopedia/index.php/Thermophysical_Properties:_Air_at_1_atm
This table comes mostly from "Incorpera's Principles of heat and mass transfer"
'''

temperature_range = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                              1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                              2500, 3000])
# Unit: m^2/s
diffusivity_range = np.array([2.54, 5.84, 10.3, 15.9, 22.5, 29.9, 38.3, 47.2, 56.7, 66.7, 76.9, 87.3, 98, 109, 120, 131,
                              143, 155, 168, 195, 224, 238, 303, 350, 390, 435, 482, 534, 589, 646, 714, 783, 869, 960,
                              1570]) * math.pow(10, -6)
# Unit: kg/m^3
density_range = np.array([3.5562, 2.3364, 1.7458, 1.3947, 1.1614, 0.995, 0.8711, 0.7740, 0.6964, 0.6329, 0.5804, 0.5356,
                          0.4975, 0.4643, 0.4354, 0.4097, 0.3868, 0.3666, 0.3482, 0.3166, 0.2902, 0.2679, 0.2488,
                          0.2322, 0.2177, 0.2049, 0.1935, 0.1833, 0.1741, 0.1658, 0.1582, 0.1513, 0.1448, 0.1389,
                          0.1135])

# Unit: J/(Kg K)
heat_capacity_range = np.array([1.032, 1.012, 1.007, 1.006, 1.007, 1.009, 1.014, 1.021, 1.030, 1.040, 1.051, 1.063,
                                1.075, 1.087, 1.099, 1.110, 1.121, 1.131, 1.141, 1.159, 1.175, 1.189, 1.207, 1.23,
                                1.248, 1.267, 1.286, 1.307, 1.337, 1.372, 1.417, 1.478, 1.558, 1.665, 2.726]) * 1000

# Unit: W/(m K)
thermal_conductivity_range = np.array([9.34, 13.8, 18.1, 22.3, 26.3, 30.0, 33.8, 37.3, 40.7, 43.9, 46.9, 49.7, 52.4,
                                       54.9, 57.3, 59.6, 62.0, 64.3, 66.7, 71.5, 76.3, 82, 91, 100, 106, 113, 120, 128,
                                       137, 147, 160, 175, 196, 222, 486]) * math.pow(10, -3)

diffusivity_of_air = scipy.interpolate.UnivariateSpline(temperature_range, diffusivity_range)
density_of_air = scipy.interpolate.UnivariateSpline(temperature_range, density_range)
heat_capacity_of_air = scipy.interpolate.UnivariateSpline(temperature_range, heat_capacity_range)
thermal_conductivity_of_air = scipy.interpolate.UnivariateSpline(temperature_range, thermal_conductivity_range)


class UValues:
    """
    Computes U-values for the boundary and stores them in memory.
    # Todo: Note that the U-values for corner points are multiply stored, so make sure they are equal..
    """

    def __init__(self, room: Room):
        self.nx, self.ny, self.nz = room.LENGTH_STEPS, room.WIDTH_STEPS, room.HEIGHT_STEPS
        self._room = room
        self.u_east = np.zeros((self.ny, self.nz))
        self.u_west = np.zeros((self.ny, self.nz))
        self.u_north = np.zeros((self.nx, self.nz))
        self.u_south = np.zeros((self.nx, self.nz))
        self.u_up = np.zeros((self.nx, self.ny))
        self.u_down = np.zeros((self.nx, self.ny))
        self.setup()

    def u_value(self, i, j, k):
        """
        Returns U-values for boundary nodes. The U-value for the wall is the same in all cases,
        but the U-values for the door and windows are set higher if the room is 'poorly_insulated'
        """
        if self._room.room_type == 'poorly_insulated':
            if (i, j, k) in self._room.windows:
                return 4.5
            elif (i, j, k) in self._room.door:
                return 1.8
        u_val = U_wall
        if (i, j, k) in self._room.windows:
            u_val = U_window
        elif (i, j, k) in self._room.door:
            u_val = U_door
        return u_val

    def setup(self):
        """
        Computes all the U-values and stores them in memory as six 2-dimensional numpy arrays.
        """
        print('Precomputing U-value matrices in UValues-object. May take a while.')
        if self._room.room_type == 'perfectly_insulated':
            # return. Then all U-values are zero.
            return
        elif self._room.test_mode == 'test_time_dependent_robin_bc':
            # Not sure if it is really necessary to keep them constant in this test.
            self.u_east = np.ones((self.ny, self.nz))
            self.u_west = np.ones((self.ny, self.nz))
            self.u_north = np.ones((self.nx, self.nz))
            self.u_south = np.ones((self.nx, self.nz))
            self.u_up = np.ones((self.nx, self.ny))
            self.u_down = np.ones((self.nx, self.ny))
            return
        else:
            for j in range(self.ny):
                for k in range(self.nz):
                    self.u_east[j, k] = self.u_value(self.nx - 1, j, k)
                    self.u_west[j, k] = self.u_value(0, j, k)

            for i in range(self.nx):
                for k in range(self.nz):
                    self.u_north[i, k] = self.u_value(i, self.ny - 1, k)
                    self.u_south[i, k] = self.u_value(i, 0, k)

            for i in range(self.nx):
                for j in range(self.ny):
                    self.u_up[i, j] = self.u_value(i, j, self.nz - 1)
                    self.u_down[i, j] = self.u_value(i, j, 0)
            return


if os.environ.get("execution_mode") == "TEST":
    pass
