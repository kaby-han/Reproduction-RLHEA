import random
import math
import numpy as np
from ObjFunctionAndPenalty import ObjectiveFunction, ObjectiveFunction_Penalty
from VariableNeighborhood import (relocate_delta, swap_delta, two_opt_delta, two_relocate_delta,
                                  NodeArc_swap_delta, ArcArc_swap_delta, swap_star)

Nv = 5
Nf = 5
Nc = 20
depot_frequency = {depot: 0 for depot in range(Nc, Nc + Nf)}  # 高质量解中仓库的使用频率 (字典)

def build_Qtable_Reward(n_states, actions):
    q_table = np.zeros((n_states, actions))
    reward = np.zeros((n_states, actions))
    return q_table, reward


def RL_SOVND(paths, Q, R, neighbor_structures, neighbor_delta, distance_matrix, cus_demand, x, y,
            global_optimum_obj, Nd, capacity, up=4, epsilon=0.7, alpha=0.2, gamma=0.85, xi=0.95):
    """x, y, distance_matrix, cus_demand, neighbor_structure, neighbor_delta在函数中全局使用"""
    """Nd, capacity, up=4, epsilon=0.7, alpha=0.2, gamma=0.85, xi=0.95参数在函数中全局使用"""

    def choose_action(st, curr_state):
        """选择下一个动作"""
        # 当前未被探索过的邻域算子
        available_action_name = [n for n in actions if n not in st]
        available_action_idx = [neighbor_structures[n][1] for n in available_action_name]

        # 当前状态各动作的Q值
        state_table = Q[neighbor_structures[curr_state][1], available_action_idx]

        # epsilon贪心策略：epislon的概率均匀选择动作，1-epislon概率选择Q值最大的动作
        if (np.random.uniform() < epsilon) or ((state_table == 0).all()):
            # 随机选择邻域
            action_name = np.random.choice(available_action_name)
        else:
            action_name = available_action_name[np.argmax(state_table)]

        return action_name, action_name, available_action_name  # 返回名称

    def solution_improve(neighbor_operation, routes):
        """第一改进策略更新当前解"""
        current_objective = ObjectiveFunction(routes, distance_matrix)
        local_best_solution = []
        local_best_obj = float('inf')
        count = 0

        # 终止条件二：达到设定迭代次数
        while count < 300:
            # 生成邻域解并计算其目标函数值
            if neighbor_operation == "swap_star":
                local_routes = swap_star(routes, x, y, distance_matrix)
            else:
                local_routes = neighbor_structures[neighbor_operation][0](routes, neighbor_delta)
            local_objective = ObjectiveFunction(local_routes, distance_matrix)

            # 终止条件一：如果发现更好的解，返回
            if local_objective < current_objective:
                return local_routes, local_objective, current_objective

            if local_objective < local_best_obj:
                local_best_solution, local_best_obj = local_routes, local_objective

            count += 1

        return local_best_solution, local_best_obj, current_objective


    def feasible_judge(routes):
        """判断解是否为可行解：使用仓库数量；车辆路径负载"""
        depots_used = {route[0] for route in routes}  # 计算仓库使用数量
        # 是否有超载情况
        overloaded = any(
            sum(cus_demand[customer] for customer in route[1:-1]) > capacity for route in routes)

        if len(depots_used) > Nd or overloaded:
            return False
        else:
            return True


    actions = list(neighbor_structures.keys())  # 可执行的动作列表
    #print(f"所有可执行的动作列表{actions}")
    count_f, count_i = 0, 0  # 初始化连续接受的可/不可行解计数器
    improve = True
    beta = ObjectiveFunction(paths, distance_matrix) / sum(cus_demand)  # 初始惩罚系数 beta
    local_optimum_paths = paths  # 记录局部最优解
    print(paths)
    local_optimum_obj = ObjectiveFunction(local_optimum_paths, distance_matrix)  # 记录局部最优解的目标函数值

    count = 0
    while improve:
        improve = False
        ne = 0  # 初始化已探索邻域结构的数量
        statement = []  # 设定已探索邻域的当前状态

        while ne < len(actions):
            #print("*"*20 + f"第{count+1}次改进的第{ne+1}个邻域结构" + "*"*20)
            # 当前状态 → 刚添加的邻域算子/初始时随机选择
            if statement:
                state = state_next
            else:
                state = random.choice(actions)

            # ****** 1. 用 Q-learning 的方式选择待考察的邻域(用邻域名字表示) ******
            N_delta, state_next, curr_state_avail_actions = choose_action(statement, state)

            #print(f"N_delta为{N_delta}，下一个状态为{state_next}")

            # ****** 2. 用 n-delta 进行邻域改进 ****** (paths_better就是Sr: 一个改进解/全部邻域范围内的最优解)
            #print(f'原解为{paths}')
            neighbor_paths, neighbor_obj, current_obj = solution_improve(N_delta, paths)
            #print(f"邻域搜索解为{neighbor_paths}")
            a, b = validate_numbers_in_range(neighbor_paths)

            if not a or not b:
                #print(f'邻域搜索解存在问题')
                return None, None, Q, R

            #print(f"当前目标函数值为：{current_obj}，邻域搜索后目标函数值为{neighbor_obj}")
            statement.append(N_delta)  # 更新已探索邻域的当前状态
            #print(f"当前状态的情况为{statement}")

            # ****** 3. 更新 Q 和 R 的值 ******
            current_state_idx, action_idx, next_state_idx = (
                neighbor_structures[state][1], neighbor_structures[N_delta][1],
                neighbor_structures[state_next][1])

            # 贝尔曼最优公式的参数
            reward, q_predict = R[current_state_idx, action_idx], Q[current_state_idx, action_idx]

            # 判断是否为最后一个可用状态
            if len(statement) == 7:
                q_target = reward
            else:
                next_state_avail_action = [neighbor_structures[n][1] for n in actions if n not in statement]
                q_target = reward + gamma * Q[next_state_idx, next_state_avail_action].max()

            # 更新当前 状态st 采取 动作a 对应 Q表 Q(st, a)
            Q[current_state_idx, action_idx] += alpha * (q_target - q_predict)

            # 用于奖励矩阵更新的参数值
            delta_r = current_obj - neighbor_obj
            delta_b = global_optimum_obj - neighbor_obj

            # 更新奖励矩阵
            if delta_r >= 0:
                R[current_state_idx, action_idx] = (xi * R[current_state_idx, action_idx] + delta_r
                                              + max(0, delta_b) * math.exp(len(actions) - len(curr_state_avail_actions)))
            else:
                R[current_state_idx, action_idx] = xi * R[current_state_idx, action_idx] + delta_r

            # ****** 4. 判断带惩罚的目标函数值 ******
            neighbor_obj_penalty = ObjectiveFunction_Penalty(neighbor_paths, distance_matrix, cus_demand, capacity, beta)
            current_obj_penalty = ObjectiveFunction_Penalty(paths, distance_matrix, cus_demand, capacity, beta)
            if neighbor_obj_penalty < current_obj_penalty:
                paths = neighbor_paths  # 在F目标函数下接收更优解
                #print(f"！！！！！在不可行域F下更新！！！！！")
                if feasible_judge(paths) and neighbor_obj < local_optimum_obj:
                    #print(f'更新前的局部最优解值为{local_optimum_obj}')
                    #print(f'更新后的局部最优解值为{neighbor_obj}')
                    local_optimum_paths = neighbor_paths  # 更新局部最优解
                    local_optimum_obj = neighbor_obj
                    #print(f"！！！！！在可行域下更新！！！！！")
                    count_f += 1  # 有改进的可行解，If更新
                else:
                    count_i += 1

                # ****** 5. 调整惩罚系数beta ******
                if count_i == up:
                    beta = beta * (1.5 + random.randint(0,1))
                    count_i = 0
                if count_f == up:
                    beta = beta / (1.5 + random.randint(0,1))
                    count_f = 0

                # 解有改进 → 结束本次邻域结构探索循环
                improve = True
                count += 1
                break
            else:
                ne += 1

    #print(f'返回的局部最优解为：{local_optimum_paths}')
    #print(f'返回的局部最优目标函数值为：{local_optimum_obj}')
    return local_optimum_paths, local_optimum_obj, Q, R


def update_frequency(routes, frequent_depot):
    """用高质量解更新仓库使用频率"""
    for i in range(len(routes)):
        frequent_depot[routes[i][0]] += 1
    return frequent_depot
