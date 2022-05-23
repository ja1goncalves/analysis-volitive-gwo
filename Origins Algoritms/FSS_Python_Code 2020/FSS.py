import numpy as np
import copy


# This code was based on in the following references:
# [1] "A Novel Search Algorithm based on Fish School Behavior" published in 2008 by Bastos Filho, Lima Neto,
# Lins, D. O. Nascimento and P. Lima
# [2] "An Enhanced Fish School Search Algorithm" published in 2013 by Bastos Filho and  D. O. Nascimento

# Created by:
# Clodomir Santana Jr. (cjsj@ecomp.poli.br)
# Elliackin Figueredo (emnf@ecomp.poli.br)
# Mariana Macedo (mgmm@ecomp.poli.br)
# Pedro Santos (pjbls@ecomp.poli.br)


class Fish(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.weight = np.nan
        self.cost = np.nan
        self.has_improved = False


class FSS(object):
    def __init__(self, objective_function, search_space_initializer, n_iter, school_size, step_individual_init,
                 step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale):
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer

        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf
        self.n_iter = n_iter

        self.school_size = school_size
        self.step_individual_init = step_individual_init
        self.step_individual_final = step_individual_final
        self.step_volitive_init = step_volitive_init
        self.step_volitive_final = step_volitive_final

        self.curr_step_individual = self.step_individual_init * (self.maxf - self.minf)
        self.curr_step_volitive = self.step_volitive_init * (self.maxf - self.minf)
        self.min_w = min_w
        self.w_scale = w_scale
        self.prev_weight_school = 0.0
        self.curr_weight_school = 0.0
        self.best_fish = None

        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def __gen_weight(self):
        return self.w_scale / 2.0

    def __init_fss(self):
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []

    def __init_fish(self, pos):
        fish = Fish(self.dim)
        fish.pos = pos
        fish.weight = self.__gen_weight()
        fish.cost = self.objective_function.evaluate(fish.pos)
        self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        return fish

    def __init_school(self):
        self.best_fish = Fish(self.dim)
        self.best_fish.cost = np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        positions = self.search_space_initializer.sample(self.objective_function, self.school_size)

        for idx in range(self.school_size):
            fish = self.__init_fish(positions[idx])
            self.school.append(fish)
            self.curr_weight_school += fish.weight
        self.prev_weight_school = self.curr_weight_school
        self.update_best_fish()
        self.optimum_cost_tracking_iter.append(self.best_fish.cost)

    def max_delta_cost(self):
        max_ = 0
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0
        for fish in self.school:
            self.curr_weight_school += fish.weight

    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=np.float)
        density = 0.0

        for fish in self.school:
            density += fish.weight
            for dim in range(self.dim):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def update_steps(self, curr_iter):
        self.curr_step_individual = self.step_individual_init - curr_iter * float(
            self.step_individual_init - self.step_individual_final) / self.n_iter

        self.curr_step_volitive = self.step_volitive_init - curr_iter * float(
            self.step_volitive_init - self.step_volitive_final) / self.n_iter

    def update_best_fish(self):
        for fish in self.school:
            if self.best_fish.cost > fish.cost:
                self.best_fish = copy.copy(fish)

    def feeding(self):
        for fish in self.school:
            if self.max_delta_cost():
                fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    def individual_movement(self):
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.float)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + (self.curr_step_individual * np.random.uniform(-1, 1))
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf
            cost = self.objective_function.evaluate(new_pos)
            self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            if cost < fish.cost:
                fish.delta_cost = abs(cost - fish.cost)
                fish.cost = cost
                delta_pos = np.zeros((self.dim,), dtype=np.float)
                for idx in range(self.dim):
                    delta_pos[idx] = new_pos[idx] - fish.pos[idx]
                fish.delta_pos = delta_pos
                fish.pos = new_pos
            else:
                fish.delta_pos = np.zeros((self.dim,), dtype=np.float)
                fish.delta_cost = 0

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dim,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dim):
                cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
        for dim in range(self.dim):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.float)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf

            fish.pos = new_pos

    def collective_volitive_movement(self):
        self.total_school_weight()
        barycenter = self.calculate_barycenter()
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.float)
            for dim in range(self.dim):
                if self.curr_weight_school > self.prev_weight_school:
                    new_pos[dim] = fish.pos[dim] - ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                else:
                    new_pos[dim] = fish.pos[dim] + ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf

            cost = self.objective_function.evaluate(new_pos)
            self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            fish.cost = cost
            fish.pos = new_pos

    def optimize(self):
        self.__init_fss()
        self.__init_school()

        for i in range(self.n_iter):
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()
            self.optimum_cost_tracking_iter.append(self.best_fish.cost)
            #print "Iteration: ", i, " Cost: ", self.best_fish.cost
