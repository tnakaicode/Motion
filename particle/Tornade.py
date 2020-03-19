import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
import math
import datetime


from .base import Calculater, Particle, ParticleSystem

# the system of acting force to rotate like as tornade


class TornadeSystem(ParticleSystem):
    def __init__(self, cutoff_r, NX, NY, NZ, e, mass, eps, sigma):
        super().__init__(cutoff_r, NX=NX, NY=NY, NZ=NZ, e=e, mass=mass, eps=eps, sigma=sigma)

    # generate 3000 particles in random position
    def init_particles(self):
        for _ in range(3000):
            x = self.X_MAX * random.random()
            y = self.Y_MAX * random.random()
            z = self.Z_MAX * random.random()
            self.particles.append(
                Particle(pos=[x, y, z], vel=[0.0, 0.0, 0.0], mass=self.mass, type=1))

    # Rotating force around x-axis
    # rot (r) == [0, 0, 1]
    def force(self, pos, vel, particle, t):
        # - 0.5 * np.array(vel)
        return 0.05 * pos[2] * np.array([-(pos[1] - self.Y_MAX / 2), pos[0] - self.X_MAX / 2, 0.0])
