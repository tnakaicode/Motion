import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
import math
import datetime

from .base import Calculater, Particle, ParticleSystem

# the particle group in box under the gravity


class BoxGravitySystem(ParticleSystem):
    def __init__(self, cutoff_r, NX, NY, NZ, e, mass, eps, sigma):
        super().__init__(cutoff_r, NX=NX, NY=NY, NZ=NZ, e=e, mass=mass, eps=eps, sigma=sigma)

    # generate the particles on the bottom of box
    # 10*10 particles
    # set the initial velocity
    # as direction to upper
    def init_particles(self):
        for x in np.linspace(0.1 * self.X_MAX, 0.2 * self.X_MAX, 10):
            for y in np.linspace(0.1 * self.Y_MAX, 0.2 * self.Y_MAX, 10):
                for z in np.linspace(0.1 * self.Z_MAX, 0.2 * self.Z_MAX, 10):
                    self.particles.append(Particle(pos=[x, y, z], vel=[
                                          0.1 * self.X_MAX, 0.05 * self.Y_MAX, 0.5 * self.Z_MAX], mass=self.mass))

    # gravity
    def force(self, pos, vel, particle, t):
        return np.array([0.0, 0.0, -particle.mass * 9.8])
