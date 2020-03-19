import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
import math
import datetime


from .base import Calculater, Particle, ParticleSystem

# System of bullet driving into the wall


class BulletWallSystem(ParticleSystem):
    def __init__(self, cutoff_r, NX, NY, NZ, e, mass, eps, sigma):
        super().__init__(cutoff_r, NX=NX, NY=NY, NZ=NZ, e=e, mass=mass, eps=eps, sigma=sigma)

    # set tha ball has a initial velocity
    # set the fixed wall
    def init_particles(self):
        # generate the ball
        bullet_center = [0.35 * self.X_MAX, 0.5 * self.Y_MAX, 0.5 * self.Z_MAX]
        for i in range(200):
            r = 0.05 * self.X_MAX * (random.random() - 0.5) * 2
            phi = 2 * np.pi * random.random()
            theta = np.pi * random.random()
            self.particles.append(Particle(pos=[bullet_center[0] + r * np.sin(theta) * np.cos(phi), bullet_center[1] + r * np.sin(
                theta) * np.sin(phi), bullet_center[2] + r * np.cos(theta)], vel=[0.2 * self.X_MAX, 0, 0], mass=self.mass, type=1))

        # generate the wall
        for x in np.linspace(0.49 * self.X_MAX, 0.50 * self.X_MAX, 2):
            for y in np.linspace(0.3 * self.Y_MAX, 0.7 * self.Y_MAX, 30):
                for z in np.linspace(0.3 * self.Z_MAX, 0.7 * self.Z_MAX, 30):
                    self.particles.append(
                        Particle(pos=[x, y, z], vel=[0.0, 0.0, 0.0], mass=self.mass, type=2))

    # no force from another system
    def force(self, pos, vel, particle, t):
        return np.array([0.0, 0.0, 0])
