import numpy as np
import sys
import time
import os
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
import math
import datetime
sys.path.append(os.path.join('../'))

from src.particle.base import Calculater
from src.particle.Tornade import TornadeSystem


if __name__ == '__main__':
    cutoff_r = 1.0
    e = 0.1
    mass = 1.0
    eps = 1
    sigma = 0.5
    dt = 0.01
    system = TornadeSystem(cutoff_r=cutoff_r, NX=100,
                           NY=100, NZ=100, e=e, mass=mass, eps=eps, sigma=sigma)
    time = 0
    system.start_output_gif(fps=0.1)
    while time <= 2.5:
        system.update(dt=dt, calc_method=Calculater.runge_kutta)
        time = system.get_time()
        print("Time : ", time)
        # system.show_snapshot()
    system.stop_output_gif(filename="Particle_TornadeSystem.gif")
