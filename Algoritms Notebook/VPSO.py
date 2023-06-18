import random 
from Particle import VParticle
import copy 
import math 
import numpy as np

class VPSO(object):
    def __init__(self, objective_function, space_initializer, n_iter, max_evaluations, part_size,
               inertia=0.5, cognitive=1, social=2, vel_max=10, topology='GBEST',
               clerck=True,  auto_coef = False, min_w=0.4, vol_init=0.01, vol_final=0.001):
        self.objective_function = objective_function # função de avalição de custo
        self.space_initializer = space_initializer # posições iniciais dos peixes

        self.dim = objective_function.dim
        self.minf = objective_function.minf # limite minimo da função
        self.maxf = objective_function.maxf # limite máximo da função
        self.n_iter = n_iter
        self.max_evaluations = max_evaluations
        self.topology = topology
        self.clerck = clerck
        self.auto_coef = auto_coef
        
        self.min_w = min_w
        self.prev_weight = 0.0
        self.curr_weight = 0.0
        self.barycenter = np.zeros(self.dim)
        self.step_vol_init = vol_init
        self.step_vol_final = vol_final
        self.curr_step_vol = self.step_vol_init * (self.maxf - self.minf)
        self.max_delta_fitness = 0

        self.part_size = part_size  # quantidade de peixes
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        self.v_max = vel_max
        self.g_best = None

        self.optimum_fitness_tracking_iter = []
        self.optimum_fitness_tracking_eval = []
  
    def __init_fitness_tracking(self):
        self.optimum_fitness_tracking_iter = []
        self.optimum_fitness_tracking_eval = []
    
    def __gen_aromatic_intensity(self, fitness):
        return self.n_iter / fitness
    
    def __init_particle(self, pos):
        particle = VParticle(self.dim)
        particle.pos = pos
        particle.fitness = self.objective_function.evaluate(particle.pos)
        particle.weight = self.__gen_aromatic_intensity(particle.fitness)
        self.optimum_fitness_tracking_eval.append(self.g_best.fitness)
        return particle

    def __init_pso(self):
        self.g_best = VParticle(self.dim)
        self.particles = []

        positions = self.space_initializer.sample(self.objective_function, self.part_size)

        for idx in range(self.part_size):
            particle = self.__init_particle(positions[idx])
            self.particles.append(particle)
        self.update_bests_particle()
        self.optimum_fitness_tracking_iter.append(self.g_best.fitness)

    def update_bests_particle(self):
        for i, p in enumerate(self.particles):
            p.update_bests()
            if p.fitness < self.g_best.fitness or self.g_best.fitness == np.inf:
                self.g_best = copy.copy(p)

    def get_lbest(self, particleA, particleB):
    #if betterthan(particleA.fitness, particleB.fitness, maximization=True):
        return particleA if particleA.fitness <= particleB.fitness else particleB

    def constriction_factor(self, iteration):
        """It was found that when rho < 4, the swarm would slowly
        "spiral" toward and around the best found solution in the
        search space with no guarantee of convergence, while for
        rho > 4 convergence would be quick and guaranteed. (Defining a Standard
        for Particle Swarm Optimization)"""

        rho = self.c1 + self.c2
        if rho < 4:
            print(f"rho = {rho}")

        return (2 / abs(2 - rho - math.sqrt(rho**2 - 4 * rho)))

    def collective_movement(self):
        self.max_delta_fitness = 0
        for i, particle in enumerate(self.particles):
            if self.topology == "LBEST": # best of neighbors
                best = self.get_lbest(self.particles[(i - 1) % len(self.particles)], self.particles[(i + 1) % len(self.particles)])
            else: # best of all
                best = self.g_best
  
            new_pos = np.zeros((self.dim,), dtype=float)
            for dim in range(self.dim):

                r1 = random.random()
                r2 = random.random()
                vel_cognitive = self.c1 * r1 * (particle.best_pos[dim] - particle.pos[dim])
                vel_social = self.c2 * r2 * (best.pos[dim] - particle.pos[dim])
                particle.vel[dim] = self.w * particle.vel[dim] + (vel_cognitive + vel_social)

                new_pos[dim] = particle.pos[dim] + particle.vel[dim]

                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf
                    
            self.optimum_fitness_tracking_eval.append(self.g_best.fitness)
            fitness = self.objective_function.evaluate(new_pos)
            particle.delta_fitness = abs(fitness - particle.fitness)
            delta_pos = new_pos - particle.pos
            particle.delta_pos = delta_pos
            particle.pos = new_pos
            particle.fitness = fitness
            if particle.delta_fitness > self.max_delta_fitness:
                self.max_delta_fitness = particle.delta_fitness
    
    def sniffing(self):
        self.prev_weight = self.curr_weight
        self.curr_weight = 0.0
        discount_factor = -0.1
        for particle in self.particles:
            if self.max_delta_fitness:
                gain_loss_weight = (particle.delta_fitness / self.max_delta_fitness)
            else:
                gain_loss_weight =  discount_factor

            particle.weight = particle.weight + gain_loss_weight
            
            if particle.weight < self.min_w:
                particle.weight = self.min_w

            self.curr_weight += particle.weight

    def fss_volitive_movement(self):
        # self.total_pack_ai()
        barycenter = self.calculate_barycenter()
        self.barycenter = barycenter

        if self.curr_weight > self.prev_weight:
            self.curr_mult_vol = 1
            direction = -1
        else:
            direction = +1
            
        for particle in self.particles:
            new_pos = np.zeros((self.dim,), dtype=float)

            jump = 1
            numerator = (particle.pos - barycenter)
            #denominator = np.linalg.norm(wolf.pos - barycenter) # euclidean distance
            denominator = 1

            volitive_move = jump * self.curr_step_vol * \
                np.random.uniform(0, 1, size=self.dim) * \
                (numerator/denominator)

            new_pos = particle.pos + direction * volitive_move

            new_pos[new_pos < self.minf] = self.minf
            new_pos[new_pos > self.maxf] = self.maxf

            fitness = self.objective_function.evaluate(new_pos)
            self.optimum_fitness_tracking_eval.append(self.g_best.best_fitness)                                    
            particle.fitness = fitness
            particle.pos = new_pos

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=float)
        density = 0.0

        for particle in self.particles:
            density += particle.weight
            for dim in range(self.dim):
                barycenter[dim] += (particle.pos[dim] * particle.weight)
        
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def calc_coef(self, cur_iter):
        if self.clerck:
            self.w = self.constriction_factor(cur_iter)

        if self.auto_coef:
            max_iter = self.n_iter
            self.w = (0.4/max_iter**2) * (cur_iter - max_iter) ** 2 + 0.4
            self.c1 = -3 * cur_iter / max_iter + 3.5
            self.c2 =  3 * cur_iter / max_iter + 0.5

        self.curr_step_vol = self.step_vol_init - cur_iter * float(
            self.step_vol_init - self.step_vol_final) / self.n_iter


    def optimize(self):
        self.__init_fitness_tracking()
        self.__init_pso()
        i = 0
        while self.objective_function.evaluations < self.max_evaluations:

            self.calc_coef(i)
            self.collective_movement()
            self.sniffing()
            self.fss_volitive_movement()
            self.update_bests_particle()
            self.optimum_fitness_tracking_iter.append(self.g_best.fitness)
            i+=1
