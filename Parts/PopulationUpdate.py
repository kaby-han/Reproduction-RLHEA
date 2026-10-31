import numpy as np
from MPEAX_OffspringGeneration import build_adjacency_matrix
from ObjFunctionAndPenalty import ObjectiveFunction


def dist_ab(routes_a, routes_b, Nc, Nf):
    """计算 A 与 B 的海明距离"""
    # ****** 1. 将路径转换为邻接矩阵 ******
    matrix_a = build_adjacency_matrix(Nc + Nf, routes_a)
    matrix_b = build_adjacency_matrix(Nc + Nf, routes_b)

    # ****** 2. 删去共同边并计数 A 的非共同边数 ******
    # 计算 matrix_a 中大于 0 的位置，转换为整数（0 或 1）
    positive_a = (matrix_a > 0).astype(int)
    positive_b = (matrix_b > 0).astype(int)

    # 计算按位与操作，使用数学方法
    matrix_a_inter = matrix_a * positive_a * (1 - positive_b)

    # 计算非共同边的数量并返回距离
    dist = np.sum(matrix_a_inter > 0)

    return dist


def dist_apop(solution, pop, Nc, Nf):
    """计算解与种群的海明距离"""
    # 种群为空
    if not pop:
        return 1000

    # 计算 A 与种群中每个解之间的距离
    dist_pop = [dist_ab(solution, routes, Nc, Nf) for routes in pop]

    # 返回最小距离
    return min(dist_pop)


def calculate_fit(solution, population, max_pd, min_pd, max_obj, min_obj, distance_matrix, Nc, Nf, psi):
    """计算解的适应度"""
    # 补充解的目标函数值和解与种群的海明距离
    solution_obj = ObjectiveFunction(solution, distance_matrix)
    solution_dis = dist_apop(solution, population, Nc, Nf)  # 解到种群的距离

    obj_value = (max_obj - solution_obj) / (max_obj - min_obj)
    pd_value = (solution_dis - min_pd) / (max_pd - min_pd)

    return psi * obj_value + (1 - psi) * pd_value


def population_update(solution_a, population, distance_matrix, Nf, Nc, psi):
    """更新种群"""
    # 解与种群的距离不为0，修正种群
    if dist_apop(solution_a, population, Nc, Nf):
        print('*'*5 + f'种群更新' + '*'*5)

        population.append(solution_a)

        if len(population) <= 20:
            return population
        else:
            # 计算修正种群中每个解与种群的海明距离
            # 计算修正种群中每个解与种群的海明距离
            pd = []
            for i in range(len(population)):
                routes = population[i]
                new_pop = population.copy()
                new_pop.pop(i)
                pd.append(dist_apop(routes, new_pop, Nc, Nf))

            pd_max, pd_min = max(pd), min(pd)

            # 计算种群的每个解的目标函数值
            population_obj = [ObjectiveFunction(u, distance_matrix) for u in population]
            pop_max_obj, pop_min_obj = max(population_obj), min(population_obj)


            # 计算种群每个解的适应度
            population_fit = [calculate_fit(u, population, pd_max, pd_min, pop_max_obj, pop_min_obj, distance_matrix, Nc, Nf, psi)
                                   for u in population]
            # 删去适应度最小的解
            population.pop(np.argmin(population_fit))

    return population