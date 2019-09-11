
#x range: +- 200
#z range: 500, 2000
#vx range: +- 10
#vz range: -30, 10

#m range: 8000, 12000

import pandas as pd
import numpy as np
import scipy as sp
from scipy import integrate as spint
import matplotlib.pyplot as plt

from numpy import sin, cos, arcsin, arctan2
import random

class Rocket:

    def __init__(self, x_0 = 0, z_0 = 1000, vx_0 = 0, vz_0 = -20, m_0 = 10000, disperse=True, alpha=0, gamma1=1):

        #everything in si units

        self.alpha = alpha
        self.gamma1 = gamma1


        if disperse:
            start_range = ([-200, 200], [500, 2000], [-10,10], [-30, 10], [8000, 12000])
            self.x_0, self.z_0, self.vx_0, self.vz_0, self.m_0 = (random.uniform(*s) for s in start_range)

        else:
            self.x_0, self.z_0, self.vx_0, self.vz_0, self.m_0  = x_0, z_0, vx_0, vz_0, m_0

        # store into state vector
        self.s_0 = (self.x_0, self.z_0, self.vx_0, self.vz_0, self.m_0)

        self.g = 1.6229 #moons gravity

        self.c1 = 44000 #max thrust
        self.c2 = 311*9.81 #isp


        self.vec_dynamics = np.vectorize(self.dynamics)

        self.t_char = (self.vz_0+(self.vz_0**2 + 2*self.g*self.z_0)**0.5)/self.g

        #bounds on the state vector
        self.x_range  =  -500,  500
        self.z_range  =     0, 2100
        self.vx_range =  -100,  100
        self.vz_range =  -100,   50
        self.m_range  =     0, self.m_0

        self.ranges = [self.x_range, self.z_range, self.vx_range, self.vz_range, self.m_range]

    def __repr__(self):

        s = 'Rocket \n'
        s += f'Initial State: {self.s_0}'

        return s

    def dynamics(self, s, u):

        (x, z, vx, vz, m) = s
        (u1, u2) = u

        ds = [None]*5

        ds[0] = vx
        ds[1] = vz
        ds[2] = self.c1*(u1/m)*sin(u2)
        ds[3] = self.c1*(u1/m)*cos(u2) - self.g


        ds[4] = -(self.c1/self.c2)*u1

        return tuple(ds)

    def propagate(self, s, u, dt=0.1):
        # s is the state
        # u is the control
        # dt is the time step (seconds)


        ds = self.dynamics(s , u)

        return tuple([s[i] + ds[i]*dt for i in range(len(s))])


    def lunar_guidance(self):
        #using DOI: 10.2514/1.45779, but adapt to 2 dimensional case

        af1, af2 = 0, 0
        rf1, rf2 = 0, 0
        vf1, vf2 = 0, 0


        #compute t_go:
        tgo = self.tgo = abs(3*(rf2-self.z_0)/(2*vf2 + self.vz_0))


        C01 = af1 -(6/tgo)*(vf1 + self.vx_0) + (12/tgo**2)*(rf1-self.x_0)
        C02 = af2 -(6/tgo)*(vf2 + self.vz_0) + (12/tgo**2)*(rf2-self.z_0)

        C11 = -6*af1/tgo + (6/tgo**2)*(5*vf1 + 3*self.vx_0) - (48/tgo**3)*(rf1 - self.x_0)
        C12 = -6*af2/tgo + (6/tgo**2)*(5*vf2 + 3*self.vz_0) - (48/tgo**3)*(rf2 - self.z_0)

        C21 = (6*af1/tgo**2) - (12/tgo**3)*(2*vf1 + self.vx_0) + (36/tgo**4)*(rf1-self.x_0)
        C22 = (6*af2/tgo**2) - (12/tgo**3)*(2*vf2 + self.vz_0) + (36/tgo**4)*(rf2-self.z_0)

        print(f'C22 = {C22} should be 0')

        self.Cs = ((C01, C11, C21), (C02, C12, C22))

        # reconstruct trajectory accel
        t = self.t_span = np.linspace(0, tgo)
        self.net_accel_x = net_accel_1 = C01 + C11*t + C21*t**2
        self.net_accel_z = net_accel_2 = C02 + C12*t + C22*t**2


        self.vx_sol = C01*t + 0.5*C11*t**2 + (1/3)*C21*t**3 + self.vx_0
        self.vz_sol = C02*t + 0.5*C12*t**2 + (1/3)*C22*t**3 + self.vz_0

        self.x_sol = (1/2)*C01*t**2 + (1/6)*C11*t**3 + (1/12)*C21*t**4 + self.vx_0*t + self.x_0
        self.z_sol = (1/2)*C02*t**2 + (1/6)*C12*t**3 + (1/12)*C22*t**4 + self.vz_0*t + self.z_0

        return

    def lunar_guidance_live(self, h_ratio=0.5, tmax = 1000):
        #using DOI: 10.2514/1.45779, but adapt to 2 dimensional case

        af1, af2 = 0, 0
        rf1, rf2 = 0, 0
        vf1, vf2 = 0, 0

        t = self.t = [0,]

        dt = 0.1
        x, z, vx, vz, m = ([s,] for s in self.s_0)

        self.x, self.z, self.vx, self.vz, self.m = x, z, vx, vz, m

        u1_mag = self.u1_mag = []
        u2_angle = self.u2_angle = []



        while z[-1] > 0 and t[-1]<tmax:
            #recompute path from current point
            #compute t_go:
            tgo = self.tgo = abs(3 * (rf2 - z[-1])/(2 * vf2 + vz[-1]))

            C01 = af1 -(6/tgo)*(vf1 + vx[-1]) + (12/tgo**2)*(rf1 - x[-1])
            C02 = af2 -(6/tgo)*(vf2 + vz[-1]) + (12/tgo**2)*(rf2 - z[-1])

            #C11 = -6*af1/tgo + (6/tgo**2)*(5*vf1 + 3*vx[-1]) - (48/tgo**3)*(rf1 - x[-1])
            #C12 = -6*af2/tgo + (6/tgo**2)*(5*vf2 + 3*vz[-1]) - (48/tgo**3)*(rf2 - z[-1])

            #C21 = (6*af1/tgo**2) - (12/tgo**3)*(2*vf1 + vx[-1]) + (36/tgo**4)*(rf1-x[-1])
            #C22 = (6*af2/tgo**2) - (12/tgo**3)*(2*vf2 + vz[-1]) + (36/tgo**4)*(rf2-x[-1])

            # reconstruct trajectory accel
            #t = self.t_span = np.linspace(0, tgo)
            #self.net_accel_x = net_accel_1 = C01 + C11*t + C21*t**2
            #self.net_accel_z = net_accel_2 = C02 + C12*t + C22*t**2

            #calc accel
            a_x = C01
            a_z = C02 + self.g

            #calc control vector
            if z[-1] > h_ratio*z[0]: #force free fall for the first half of the journey, and then try to recover.
                u1_mag.append(0)
                u2_angle.append(0)
            else:
                u1_mag.append((1/self.c1)*m[-1]*(a_x**2 + a_z**2)**0.5)
                u2_angle.append(arctan2(a_x, a_z))


            #store states and control for convenience
            sf = (x[-1], z[-1], vx[-1], vz[-1], m[-1])
            uf = (u1_mag[-1], u2_angle[-1])

            #propagate
            new_state = self.propagate(sf, uf, dt=dt)
            #x  =
            x.append(new_state[0])
            #z  =
            z.append(new_state[1])
            #vx =
            vx.append(new_state[2])
            #vz =
            vz.append(new_state[3])
            #m  =
            m.append(new_state[4])

            #t =
            t.append(t[-1]+dt)

        #get the right length
        u1_mag.append(u1_mag[-1])
        u2_angle.append(u2_angle[-1])

        if t[-1] >= tmax or abs(x[-1]) > 0.1 or abs(z[-1]) > 0.1 or abs(vx[-1]) > 0.1 or abs(vz[-1]) > 0.1 or m[-1] < 100:
            self.lunar_success = False
        else:
            self.lunar_success = True

        return
