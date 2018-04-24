# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-04-09 12:02:28
# @Last Modified by:   Kicc Shen
# @Last Modified time: 2018-04-22 22:00:59
from PyOptimize.General_Opt import Optimizer
import numpy as np
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis


class pyGaft(Optimizer):
    def __init__(self, objfunc, var_bounds, individual_size, max_iter, max_or_min, **kwargs):
        super().__init__(objfunc)
        self.max_iter = max_iter

        # 定义个体 / 种群
        self.individual = BinaryIndividual(
            ranges=var_bounds, eps=0.001)
        self.population = Population(
            indv_template=self.individual, size=individual_size).init()

        # Create genetic operators.
        # selection = RouletteWheelSelection()
        selection = TournamentSelection()
        crossover = UniformCrossover(pc=0.8, pe=0.5)
        mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

        self.engine = GAEngine(population=self.population, selection=selection,
                               crossover=crossover, mutation=mutation,
                               analysis=[FitnessStore])

        @self.engine.fitness_register
        def fitness(indv):
            """
            适应度函数： 注意这里默认为优化得到最小值
            :param indv:
            :return:
            """
            x = indv.solution

            if max_or_min == 'max':
                return objfunc(x, **kwargs)
            else:
                return -objfunc(x, **kwargs)

        @self.engine.analysis_register
        class ConsoleOutputAnalysis(OnTheFlyAnalysis):
            interval = 1
            master_only = True

            def register_step(self, g, population, engine):
                best_indv = population.best_indv(engine.fitness)
                msg = 'Generation: {}, best fitness: {:.3f}'.format(
                    g, engine.fitness(best_indv))
                # self.logger.info(msg)

            def finalize(self, population, engine):
                best_indv = population.best_indv(engine.fitness)
                x = best_indv.solution
                y = engine.fitness(best_indv)
                msg = 'Optimal solution: ({}, {})'.format(x, y)
                # self.logger.info(msg)

    def run(self):
        self.engine.run(ng=self.max_iter)


if __name__ == '__main__':
    from PyOptimize.General_Opt import Test_function

    P = [[1, 1, 2], [1, -1, 3], [3, 2, 1], [1, -5, 1], [2, 1, -2]]
    r = [1, 1, 1, 1, 1]
    u = 1
    n = 1

    X = [[1, 1, 2], [1, -1, 3], [3, 2, 1], [1, -5, 1], [2, 1, -2]]
    y = [1, 1, 1, 1, 1]

    def Rosenbrock(x):
        return Test_function().Rosenbrock(x)

    def Loss(x, **kwargs):
        return Test_function().Loss(x, **kwargs)

    def LTR(a, **kwargs):
        return Test_function().LTR(a, **kwargs)

    GAFT_Test = pyGaft(objfunc=LTR, var_bounds=[(-2, 2)] * 3,
                       individual_size=50, max_iter=10,
                       X=X, y=y).run()
