import numpy as np
from abc import abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
import math
import datetime


class Particle():
    def __init__(self, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], force=[0.0, 0.0, 0.0], mass=1.0, type=-1):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.force = np.array(force)
        self.mass = mass
        self.type = type

    # the force acting between particles under Lennard-Jones potential
    @staticmethod
    def force(pos1, pos2, eps=1.0, sigma=1.0):
        # return np.array([0, 0, 0])
        r2 = ((pos2 - pos1)**2).sum()
        if abs(r2) < 0.00001:
            return np.array([0.0, 0.0, 0.0])
        sig6_div_r6 = (sigma**2 / r2)**3
        #print("force : ", 24 * eps * (1/r2) * sig6_div_r6 * (1 - 2 * sig6_div_r6) * (pos2 - pos1))
        return 24 * eps * (1 / r2) * sig6_div_r6 * (1 - 2 * sig6_div_r6) * (pos2 - pos1)


class Calculater():
    # Update pos(ition) and vel(ocity) of particles after dt time
    # used by Runge-Kutta method
    @staticmethod
    def runge_kutta(particle, adjuscent_particles, ps, dt, t, eps, sigma):
        k1 = dt * particle.vel
        l1 = 0
        for p in adjuscent_particles:
            l1 += dt * Particle.force(pos1=particle.pos,
                                      pos2=p.pos, eps=eps, sigma=sigma) / particle.mass
        l1 += dt * ps.force(particle.pos, particle.vel, particle, t)

        k2 = dt * (particle.vel + k1 / 2)
        l2 = 0
        for p in adjuscent_particles:
            l2 = dt * Particle.force(particle.pos +
                                     l1 / 2, p.pos, eps, sigma) / particle.mass
        l2 += dt * ps.force(particle.pos + l1 / 2,
                            particle.vel + k1 / 2, particle, t + dt / 2)

        k3 = dt * (particle.vel + k2 / 2)
        l3 = 0
        for p in adjuscent_particles:
            l3 = dt * Particle.force(particle.pos +
                                     l2 / 2, p.pos, eps, sigma) / particle.mass
        l3 += dt * ps.force(particle.pos + l2 / 2,
                            particle.vel + k2 / 2, particle, t + dt / 2)

        k4 = dt * (particle.vel + k3)
        l4 = 0
        for p in adjuscent_particles:
            l4 = dt * Particle.force(particle.pos + l3,
                                     p.pos, eps, sigma) / particle.mass
        l4 += dt * ps.force(particle.pos + l3,
                            particle.vel + k3, particle, t + dt)

        particle.pos += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        particle.vel += (l1 + 2 * l2 + 2 * l3 + l4) / 6

    # Update pos(ition) and vel(ocity) of particles after dt time
    # used by Euler method
    @staticmethod
    def euler(particle, adjuscent_particles, ps, dt, t, eps, sigma):
        f = 0
        for p in adjuscent_particles:
            f += dt * Particle.force(pos1=particle.pos,
                                     pos2=p.pos, eps=eps, sigma=sigma) / particle.mass
        f += dt * ps.force(particle.pos, particle.vel, particle, t)
        particle.pos += dt * particle.vel
        particle.vel += dt * f


class ParticleSystem():
    #
    def __init__(self, cutoff_r, NX, NY, NZ, e=0.9, mass=1, eps=1, sigma=1):
        self.cutoff_r = cutoff_r
        self.NX, self.NY, self.NZ = NX, NY, NZ
        self.X_MAX = cutoff_r * NX
        self.Y_MAX = cutoff_r * NY
        self.Z_MAX = cutoff_r * NZ
        self.e = e
        self.mass = mass
        self.eps = eps
        self.sigma = sigma
        self.time = 0.0
        self.prev_timestamp = 0.0
        self.fps = 0.1
        self.particles = []
        self.snapshots = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.gif_output_mode = False
        self.init_particles()
        self.update_cells()
        self.setup_particle_type_dict()

    # Calculate pos(ition) and vel(ocity) of all particles after dt time
    def update(self, dt, calc_method):
        self.time += dt
        # update_forces_on_particle()
        self.update_pos_and_vel(dt, calc_method)
        self.update_cells()

    # Calculate pos(ition) and vel(ocity) of all particles used by "calc_method"
    def update_pos_and_vel(self, dt, calc_method):
        for particle in self.particles:
            calc_method(particle=particle, adjuscent_particles=self.get_particles_adjuscent_cell(
                particle), ps=self, dt=dt, t=self.time, eps=self.eps, sigma=self.sigma)

            # reflect on end of space
            # e := reflect coef
            if particle.pos[0] < 0:
                #particle.pos[0] += 2*(0 - particle.pos[0])
                particle.pos[0] = 0.0
                particle.vel[0] = -self.e * particle.vel[0]
            if particle.pos[0] > self.X_MAX:
                #particle.pos[0] -= 2*(particle.pos[0] - self.X_MAX)
                particle.pos[0] = self.X_MAX - 0.0001
                particle.vel[0] = -self.e * particle.vel[0]
            if particle.pos[1] < 0:
                #particle.pos[1] += 2*(0 - particle.pos[1])
                particle.pos[1] = 0.0
                particle.vel[1] = -self.e * particle.vel[1]
            if particle.pos[1] > self.Y_MAX:
                #particle.pos[1] -= 2*(particle.pos[1] - self.Y_MAX)
                particle.pos[1] = self.Y_MAX - 0.0001
                particle.vel[1] = -self.e * particle.vel[1]
            if particle.pos[2] < 0:
                #particle.pos[2] += 2*(0 - particle.pos[2])
                particle.pos[2] = 0.0
                particle.vel[2] = -self.e * particle.vel[2]
            if particle.pos[2] > self.Z_MAX:
                particle.pos[2] = self.Z_MAX - 0.0001
                #particle.pos[2] -= 2*(particle.pos[2] - self.Z_MAX)
                particle.vel[2] = -self.e * particle.vel[2]
        if self.gif_output_mode is True:
            self.take_snapshot(save_to_snapshots=True)

    # detect the cell number (index) into the particle
    # get the list of particle in 26-cells neighberhood of the cell
    def get_particles_adjuscent_cell(self, particle):
        particles = []
        index = self.get_cellindex_from_pos(particle.pos)
        for i in range(index[0] - 1, index[0] + 1):
            for j in range(index[1] - 1, index[1] + 1):
                for k in range(index[2] - 1, index[2] + 1):
                    try:
                        for p in self.cell[i][j][k] if self.cell[i][j][k] is not None else []:
                            if p is not particle:
                                particles.append(p)
                    except IndexError:
                        continue
        return particles

    # detect the cell number (index) into the particle
    # detect from the pos(ition)
    def get_cellindex_from_pos(self, pos):
        return [int(pos[0] / self.cutoff_r), int(pos[1] / self.cutoff_r), int(pos[2] / self.cutoff_r)]

    # Update all partice belonging to which cell
    def update_cells(self):
        self.cell = [[[[] for i in range(self.NX)] for j in range(
            self.NY)] for k in range(self.NZ)]
        for p in self.particles:
            try:
                index = self.get_cellindex_from_pos(p.pos)
                self.cell[index[0]][index[1]][index[2]].append(p)
            except Exception as e:
                print("Exception : ", e)
                print("pos : ", p.pos)
                # input()

    # get list of particles
    def get_particles(self):
        return self.particles

    # get time
    def get_time(self):
        return self.time

    # Take a Snap shot
    # DO NOT show display
    def take_snapshot(self, save_to_snapshots=False):
        if self.time - self.prev_timestamp > self.fps:
            self.ax.set_title("Time : {}".format(self.time))
            self.ax.set_xlim(0, self.X_MAX)
            self.ax.set_ylim(0, self.Y_MAX)
            self.ax.set_zlim(0, self.Z_MAX)
            scats = []
            for type, particles in self.particle_dict.items():
                x_list = []
                y_list = []
                z_list = []
                for p in particles[0]:
                    x_list.append(p.pos[0])
                    y_list.append(p.pos[1])
                    z_list.append(p.pos[2])
                scats.append(self.ax.scatter(
                    x_list, y_list, z_list, c=particles[1]))
            if save_to_snapshots is True:
                self.snapshots.append(scats)
                print(len(self.snapshots))
            self.prev_timestamp = self.time

    # show the status of particles
    def show_snapshot(self):
        if self.time - self.prev_timestamp > 0.1:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.take_snapshot()
            # plt.savefig('box_gravity_time-{}.png'.format(self.time))
            plt.show()
            self.prev_timestamp = self.time

    # turn on the mode to write gif
    def start_output_gif(self, fps=0.1):
        self.gif_output_mode = True
        self.snapshots = []
        self.take_snapshot(save_to_snapshots=True)

    # turn off the mode to write gif
    # and
    # write gif file
    def stop_output_gif(self, filename="hoge.gif"):
        print("stop_output_gif : ", len(self.snapshots))
        self.gif_output_mode = False
        ani = animation.ArtistAnimation(self.fig, self.snapshots)
        ani.save(filename, writer='imagemagick')
        self.snapshots = []

    # make a dictionary of every type of particle
    def setup_particle_type_dict(self):
        self.particle_dict = {}
        for p in self.particles:
            if str(p.type) not in self.particle_dict:
                # allocate the color in random
                # each type of particle in dictionary
                self.particle_dict[str(p.type)] = [[], tuple(
                    [random.random() for _ in range(3)])]
            self.particle_dict[str(p.type)][0].append(p)

    # detect Kinetic Energy of all paricle
    def get_kinetic_energy(self):
        return sum([(p.mass * np.array(p.vel)**2).sum() / 2 for p in self.particles])

    # initialize the status of particle
    @abstractmethod
    def init_particles(self):
        raise NotImplementedError()

    # the force action to system
    @abstractmethod
    def force(self, pos, vel, particle, t):
        raise NotImplementedError()
