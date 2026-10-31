import numpy as np
import random
import math
from ObjFunctionAndPenalty import ObjectiveFunction
from PopulationUpdate import dist_apop
from PopulationInitialization import initial_random_method
from RL_SOVND import RL_SOVND


def ReplacePopulation(pop, pop_length, adaptive_stores, adaptive_store_length, distance_matrix, Nf, Nd, Nc, Nv,
                      Q, R, neighbor_structures, neighbor_delta, cus_demand, x, y, global_best_obj, capacity, up, epsilon, alpha, gamma, xi):
    # 1. 删去一半解，并保留最优的
    population_obj = [ObjectiveFunction(u, distance_matrix) for u in pop]
    min_obj_idx = np.argmin(population_obj)
    pop_new = [pop[min_obj_idx]]

    remaining_idx = [i for i in range(pop_length) if i != min_obj_idx]
    # 1. 随机选择一半个体加入新种群
    remaining_idx = np.random.choice(remaining_idx, size=(pop_length // 2 - 1), replace=False)
    pop_new.extend(pop[i] for i in remaining_idx)

    # 2. 处理自适应记忆存储器，保证存储器中的解足够
    store_length = len(adaptive_stores)
    choose_length = min(store_length, adaptive_store_length // 2)

    # 有choose_length个解可被选择加入新种群
    append_idx = np.random.choice(adaptive_stores, size=choose_length, replace=False)

    # 3. 补足种群 (自适应存储器/随机解生成方法)
    count_store = 0  # 记录从自适应存储器取用解的次数
    count = 0  # 在新种群中添加解的数量
    while count < math.ceil(pop_length / 2):  # 限制只放原种群一半数量的新种群
        if count_store <= choose_length and random.random() < 0.5:
            # 从自适应存储器中选择解
            idx = np.random.choice(append_idx)
            routes = adaptive_stores[idx]

            if dist_apop(routes, pop_new, Nc, Nf):
                # 在种群中无相同解
                pop_new.append(routes)
                count += 1

            append_idx = np.delete(append_idx, np.where(append_idx == idx))  # 移除已选中的解
            count_store += 1
        else:
            # 随机生成新解 → 用RL-SOVND改进
            routes = initial_random_method(Nf, Nd, Nc, Nv)
            routes, _, Q, R = RL_SOVND(routes, Q, R, neighbor_structures, neighbor_delta, distance_matrix, cus_demand, x, y,
                              global_best_obj, Nd, capacity, up, epsilon, alpha, gamma, xi)

            if dist_apop(routes, pop_new, Nc, Nf):
                # 在种群中无相同解
                pop_new.append(routes)
                count += 1

    print(f'在种群中补充的解的数量为{count}')
    print(f'使用了自适应存储器更新了{count_store}次')
    return pop_new, Q, R