from collections import Counter
import numpy as np
import math
import copy
import random
import time
import csv


def validate_numbers_in_range(original_list, lower_bound=0, upper_bound=11):
    """
    验证原始嵌套列表中的数字是否在指定范围内且无缺失或重复。

    参数:
    original_list (list of lists): 要验证的嵌套列表
    lower_bound (int): 范围下限，默认0
    upper_bound (int): 范围上限，默认49

    返回:
    bool: 如果满足条件（无缺失、无重复），返回True；否则返回False。
    """
    # 步骤1：扁平化列表并转换为Python原生整数（消除np.int64类型影响）
    flattened = []
    for sublist in original_list:
        for num in sublist:
            flattened.append(int(num))

    # 步骤2：筛选出指定范围内的数字
    num_in_range = [num for num in flattened if lower_bound <= num <= upper_bound]

    # 步骤3：验证两个核心条件
    # 条件1：去重后是否包含指定范围所有数字（无缺失）
    unique_nums = sorted(list(set(num_in_range)))
    expected_nums = list(range(lower_bound, upper_bound + 1))  # 生成标准列表
    is_complete = unique_nums == expected_nums

    # 条件2：原始筛选列表是否无重复（筛选列表长度 == 去重后列表长度）
    is_no_duplicate = len(num_in_range) == len(unique_nums)

    # 步骤4：如果有缺失或重复，则返回False
    if not is_complete and not is_no_duplicate:
        return False, True
    elif is_complete and not is_no_duplicate:
        return True, False
    elif not is_complete and is_no_duplicate:
        return False, False
    else:
        return True, True


# -------------------- 1. 种群初始化 --------------------
def initial_greed_method(Nf, Nd, Nc, Nv, distance_matrix):
    """贪心方法初始化路径"""
    open_depot = random.sample(list(range(Nc, Nc + Nf)), Nd)  # 随机选择开放的仓库，并记录索引
    # 复制距离矩阵，用于去除被服务的结点，确保只服务一次
    copy_dep_cus_distance = distance_matrix[:, :Nc].copy()
    copy_cus_cus_distance = distance_matrix[:Nc, :Nc].copy()
    unconnected = 10000  # 避免路径被重复使用

    # ****** 1. 路径初始化 ******
    routes = []
    for _ in range(Nv):
        # 找到所有仓库到客户的最小距离和对应的索引
        min_times = [np.min(copy_dep_cus_distance[i]) for i in open_depot]
        min_indices = [np.argmin(copy_dep_cus_distance[i]) for i in open_depot]

        # 选择用时最短的仓库-客户组合
        nearest_idx = np.argmin(min_times)
        depot_idx = open_depot[nearest_idx]
        customer_idx = min_indices[nearest_idx]

        # 添加路线并移除已分配的客户
        routes.append([int(depot_idx), int(customer_idx)])  # 强制转换为Python的int类型
        copy_dep_cus_distance[:, customer_idx] = unconnected

    # ****** 2. 客户循环入队 ******
    for i in range(len(routes)):
        copy_cus_cus_distance[:, routes[i][-1]] = unconnected  # 路径初始化后，尾节点已被服务，无法抵达
    for i in range(Nc - Nv):
        veh_idx = i % Nv  # 车辆索引，循环重置
        tail = routes[veh_idx][-1]
        nearest = np.argmin(np.where(copy_cus_cus_distance[tail] > 0, copy_cus_cus_distance[tail], np.inf))
        routes[veh_idx].append(int(nearest))  # 强制转换为Python的int类型
        copy_cus_cus_distance[:, nearest] = unconnected

    # ****** 3. 完整路径 ******
    for route in routes:
        route.append(route[0])  # 闭环

    print(f'贪心方法生成解')
    return routes  # 返回的是Python list结构


def initial_random_method(Nf, Nd, Nc, Nv):
    """随机方法初始化路径"""
    open_depot = random.sample(range(Nc, Nc + Nf), Nd)  # 随机选择开放的仓库
    avail_cus = list(range(Nc))  # 客户列表

    # 初始化路径：为每个车辆分配一个仓库
    routes = [[random.choice(open_depot)] for _ in range(Nv)]

    # 客户循环入队
    for i, cus in enumerate(avail_cus):
        routes[i % Nv].append(cus)  # 循环将客户分配给各个车辆

    # 每条路径都加上起始仓库
    for route in routes:
        route.append(route[0])  # 完成闭环

    print('随机方法生成解')
    return routes




# -------------------- 2. MPEAX交叉算子 --------------------
def build_adjacency_matrix(size, paths):
    """构建邻接矩阵"""
    matrix = np.zeros((size, size))
    for path in paths:
        for i in range(len(path) - 1):
            matrix[path[i], path[i + 1]] = 1
    return matrix


def MPEAX_Offspring(Nf, Nc, father_a, father_b, distance_matrix):
    """
    Multi-Parents Edge Assembly Crossover

    参数：
    Nf: 仓库数
    Nc: 客户数
    father_a: 父代 a
    father_b: 父代 b
    distance: 距离矩阵

    返回：
    solution：两个解生成的子代
    """
    # ********** 1. 邻接矩阵和 Gab 建立 **********
    # 1. 构建邻接矩阵并处理共同边
    matrix_a = build_adjacency_matrix(Nc + Nf, father_a)  # 作用域为整个函数
    matrix_b = build_adjacency_matrix(Nc + Nf, father_b)  # 作用域为整个函数

    # 创建父a和父b的深复制（用于删弧）
    matrix_a_inter = copy.deepcopy(matrix_a)
    matrix_b_inter = copy.deepcopy(matrix_b)

    # 创建Gab
    common_edges = (matrix_a > 0) & (matrix_b > 0)   # 处理共同边
    matrix_a_inter[common_edges] = 0
    matrix_b_inter[common_edges] = 0
    Gab = matrix_a_inter + matrix_b_inter   # Gab = A和B不同边的集合

    # ********** 2. 构建一条AB_sequence **********
    def Construct_ABsequence(Gab, matrix_a_inter, matrix_b_inter):

        def select_start_edge(matrix):
            """随机选择起始边"""
            tail_poss = np.where(matrix[:Nc, :] > 0)[0]
            head_poss = np.where(matrix[:, :Nc] > 0)[1]

            def get_random_edge(poss, matrix_row=None):
                """根据提供的位置选择随机的起点和终点"""
                idx = random.randint(0, len(poss) - 1)
                tail = poss[idx]
                head = random.choice(np.where(matrix_row[tail] > 0)[0]) \
                    if matrix_row is not None \
                    else random.choice(np.where(matrix[tail] > 0)[0])
                return tail, head

            if len(tail_poss) and len(head_poss):
                if random.random() < 0.5:
                    tail, head = get_random_edge(tail_poss, matrix)
                    orient = 'out'
                else:
                    head, tail = get_random_edge(head_poss, matrix.T)  # 转置矩阵，用于选择入边
                    orient = 'in'
            elif len(tail_poss):
                tail, head = get_random_edge(tail_poss, matrix)
                orient = 'out'
            elif len(head_poss):
                head, tail = get_random_edge(head_poss, matrix.T)
                orient = 'in'
            else:
                return None, None, None

            return tail, head, orient

        def select_next_edge(current_tail, current_head, current_depot_tag,
                             matrix_a_inter, matrix_b_inter, from_a, Nc, Nf):
            """选择下一条边"""
            # 确定要搜索的矩阵（切换到另一个父代）
            search_matrix = matrix_b_inter if from_a else matrix_a_inter

            # 根据节点类型选择策略 → 所有车辆都被使用，连接仓库的弧一定是两两凑对
            if current_tail >= Nc and current_depot_tag == 0:  # 尾节点是仓库
                return depot_tail(search_matrix, Nc, Nf)
            elif current_head >= Nc and current_depot_tag == 0:  # 头节点是仓库
                return depot_head(search_matrix, Nc, Nf)
            else:  # 两个都是客户点
                return customer(tail, head, search_matrix)

        def depot_tail(matrix, Nc, Nf):
            """尾节点为仓库时的选择逻辑"""
            depot_tag = 1  # 仓库万能边被禁用

            # 寻找从仓库出发的边
            useful_depot = np.where(matrix[Nc:Nc + Nf, :Nc] > 0)[0]

            # 转换边
            if useful_depot.size > 0:
                depot_as_tail = Nc + random.choice(useful_depot)
                possible_heads = np.where(matrix[depot_as_tail] > 0)[0]
                # 仓库出发可到达客户结点
                if possible_heads.size > 0:
                    return depot_as_tail, random.choice(possible_heads), depot_tag
            else:
                return None, None, depot_tag

        def depot_head(matrix, Nc, Nf):
            """头节点为仓库时的选择逻辑"""
            depot_tag = 1  # 仓库万能边被禁用

            # 寻找指向仓库的边
            useful_depot = np.where(matrix[:Nc, Nc:Nc + Nf] > 0)[1]

            # 转换边
            if useful_depot.size > 0:
                depot_as_head = Nc + random.choice(useful_depot)
                possible_tails = np.where(matrix[:, depot_as_head] > 0)[0]
                # 仓库出发可到达客户结点
                if possible_tails.size > 0:
                    return random.choice(possible_tails), depot_as_head, depot_tag
            else:
                return None, None, depot_tag

        def customer(current_tail, current_head, matrix):
            """头节点是客户点时的选择逻辑"""
            depot_tag = 0  # 恢复万能边功能

            if arc_num == 1 and degree == 'out':
                useful_tails = np.where(matrix[:, current_head] > 0)[0]
                if useful_tails.size > 0:
                    return random.choice(useful_tails), current_head, depot_tag
                else:
                    return None, None, depot_tag

            if tail >= Nc:  # 尾节点是仓库
                # 寻找从客户指向当前头节点的弧
                useful_tails = np.where(matrix[:, current_head] > 0)[0]
                if useful_tails.size > 0:
                    return random.choice(useful_tails), current_head, depot_tag
                else:
                    return None, None, depot_tag
            elif head >= Nc:  # 头节点是仓库
                # 寻找从当前尾结点指向客户的弧
                useful_heads = np.where(matrix[current_tail, :] > 0)[0]
                if useful_heads.size > 0:
                    return current_tail, random.choice(useful_heads), depot_tag
                else:
                    return None, None, depot_tag

            # 寻找从当前头节点出发的边
            useful_heads = np.where(matrix[current_tail] > 0)[0]

            if useful_heads.size <= 0:  # 无可到达的客户结点
                useful_tails = np.where(matrix[:, current_head] > 0)[0]
                if useful_tails.size > 0:
                    return random.choice(useful_tails), current_head, depot_tag
                else:
                    return None, None, depot_tag

            # 转换边
            return current_tail, random.choice(useful_heads), depot_tag

        # 1. 随机选择起始边
        tail, head, degree = select_start_edge(Gab)

        if tail is None:  # 没有找到有效起始边
            return None, None

        if degree == 'out':
            start_tail, start_head = tail, None
        else:
            start_tail, start_head = None, head

        # 2. 构建AB序列
        AB_sequence = []
        arc_num = 1  # 头弧标签
        depot_tag = 0  # 万能边标签

        # 3. 循环搜索弧，直到返回初始点并有相同的弧方向
        while True:
            AB_sequence.append((tail, head))
            Gab[tail, head] = 0

            # 确定当前边来源并切换到另一个父代
            current_from_a = matrix_a_inter[tail, head] == 1
            matrix_a_inter[tail, head] = 0
            matrix_b_inter[tail, head] = 0

            # 检查 1.非起始段 2.尾结点相同（回到起点） 3.头结点相同（指向起点）
            if (arc_num >= 2) and (tail == start_tail or head == start_head):
                break

            # 选择下一条边
            tail, head, depot_tag = select_next_edge(tail, head, depot_tag, matrix_a_inter, matrix_b_inter,
                                                     current_from_a, Nc, Nf) # type: ignore
            arc_num += 1

            if tail is None:  # 没有找到有效起始边
                return None, None

        return AB_sequence

    # ********** 3. 生成多条AB_sequence **********
    ABsequences = []
    while True:
        new_sequence = Construct_ABsequence(Gab, matrix_a_inter, matrix_b_inter)
        if new_sequence[-1] is None:
            break
        ABsequences.append(new_sequence)

    # ********** 4. 构建中间解S0 **********
    def Construct_Intermediate(ABsequences, distance_matrix, Nc, Nf):
        # ****** 1. 构造E_set ******
        # 随机选择第一条序列的索引
        ABseq_idx_1 = random.randrange(len(ABsequences))

        common_arcs = ABsequences[ABseq_idx_1]  # 初始化为选择的第一条序列

        # ****** 2. 找到有共同结点的第二条序列 *******
        # 如果ABsequences有超过1条序列
        if len(ABsequences) > 1:
            set1 = {item for tup in ABsequences[ABseq_idx_1] for item in tup}
            for i, seq in enumerate(ABsequences):
                if i != ABseq_idx_1 and {item for tup in seq for item in tup} & set1:
                    # 与ABseq_idx_1有共同结点的ABsequence
                    common_arcs = common_arcs + ABsequences[i]

        E_set = build_adjacency_matrix(Nc + Nf, common_arcs)

        # ****** 3. 构建中间解的邻接矩阵（未修正子环和非哈密顿环）******
        # 在 父A不在E_set的边 和 父B在E_set的边
        intermediate_matrix = ((matrix_a == 1) & (E_set == 0)) | ((matrix_b == 1) & (E_set == 1))
        # .astype(int)将布尔值转为整型
        intermediate_matrix_inter = copy.deepcopy(intermediate_matrix.astype(int))

        # ****** 4. 将中间解从邻接矩阵转化为列表，并修复子环和非哈密顿环 ******
        def AdjacencyMatrix_to_Solution(Nc, Nf, intermediate_matrix_inter):
            # 中间解
            routes = []

            # 2-opt* 修复子环
            def two_opt_RepairSubtour(sub_tour, route):
                # 子环的2-opt*位置
                index_sub = random.randrange(1, len(sub_tour))
                # 正常路径的2-opt*位置
                index_tour = random.randrange(1, len(route))

                # 删除弧(index_i - 1, index_i)
                sub_tour = sub_tour[index_sub:] + sub_tour[1: index_sub]

                return route[:index_tour] + sub_tour + route[index_tour:]

            # 将中间解从邻接矩阵转化为列表
            while True:
                # 构建路径
                path = []
                # 寻找出发仓库
                useful_depot = np.where(intermediate_matrix_inter[Nc:Nc + Nf, :Nc] > 0)[0]

                # **** 1. 判断是否还存在路径 ****
                if np.count_nonzero(intermediate_matrix_inter) == 0:  # 邻接矩阵元素全为0，所有路径构建完毕
                    break

                # **** 2. 邻接矩阵元素不全为0.判断有无出发仓库 ****
                elif useful_depot.size == 0:  # 无出发仓库，有子环
                    # ** 1. 初始两个客户点 **
                    start_tail = np.where(intermediate_matrix_inter[:Nc, :Nc] > 0)[0][0]
                    tail = start_tail
                    head = np.where(intermediate_matrix_inter[start_tail] > 0)[0][0]
                    path.extend([start_tail, head])

                    # ** 2. 构建出子环 **
                    while True:
                        intermediate_matrix_inter[tail][head] = 0

                        if head == start_tail:  # 子环构成回路
                            break
                        else:
                            tail = head
                            head = np.where(intermediate_matrix_inter[tail] > 0)[0][0]
                            path.append(head)

                    # ** 3. 将子环加入正常路径 **
                    route_index = random.randrange(len(routes))
                    routes[route_index] = two_opt_RepairSubtour(path, routes[route_index])

                    # ** 4. 可能不存在一个子环，继续循环 **
                    continue

                # **** 3. 可以构建从仓库出发的路径 ****
                else:
                    # ** 1. 选择出发仓库 **
                    start_depot = Nc + random.choice(useful_depot)
                    # 第一个客户
                    first_customer = np.where(intermediate_matrix_inter[start_depot] > 0)[0][0]
                    # 加入路径
                    path.extend([start_depot, first_customer])
                    tail, head = start_depot, first_customer

                    # ** 2. 构建一条路径（非子环）**
                    while True:
                        intermediate_matrix_inter[tail][head] = 0

                        # 寻找下一个结点
                        tail = head
                        head = np.where(intermediate_matrix_inter[tail] > 0)[0][0]

                        # ** 3. 哈密顿环判定并消除 **
                        if head >= Nc and head == start_depot:  # 路径构造完毕，为哈密顿环
                            intermediate_matrix_inter[tail][head] = 0
                            path.append(head)
                            break
                        elif head >= Nc and head != start_depot:  # 路径构造完毕，为非哈密顿环
                            # 从两个仓库中选择离第一个客户更近的
                            if distance_matrix[start_depot][first_customer] < distance_matrix[head][first_customer]:
                                intermediate_matrix_inter[tail][head] = 0  # 非哈密顿环终边被转换，需去除
                                path.append(start_depot)  # 出发仓库更近
                                break
                            else:
                                path[0] = head  # 终点仓库更近
                                intermediate_matrix_inter[tail][head] = 0  # 非哈密顿环终边被转换，需去除
                                path.append(head)
                                break
                        else:
                            path.append(head)

                routes.append(path)

            return routes  # 返回函数AdjacencyMatix_to_Solution，将中间解从邻接矩阵变为列表
        # 将邻接矩阵解转换，此时为列表型的中间解
        return AdjacencyMatrix_to_Solution(Nc, Nf, intermediate_matrix_inter)
    # 返回列表型的中间解
    return Construct_Intermediate(ABsequences, distance_matrix, Nc, Nf)



# -------------------- 3. 约束违背修复(仓库数量、车辆运载能力) --------------------
# ****** 1. 修复仓库约束 ******
def repair_violate_depot(routes, frequent_depot, distance_matrix, Nc, Nf, Nd):
    """修复仓库使用数量超过限制的问题"""

    def repair_routes(depot_list, routes_to_fix):
        """使用指定仓库列表修复路径"""
        # 获取启用仓库到客户的距离
        depot_to_customer = distance_matrix[depot_list, :Nc]

        for i, route in enumerate(routes_to_fix):
            if route[0] not in depot_list:  # 路径仓库已被废弃
                # 选择离第一个客户最近的启用仓库
                first_customer = route[1]
                distances = depot_to_customer[:, first_customer]
                nearest_depot = depot_list[np.argmin(distances)]
                routes_to_fix[i][0] = routes_to_fix[i][-1] = nearest_depot
        return routes_to_fix

    # 随机选择修复策略
    if random.random() < 0.5:
        # 计算被使用的仓库的数量
        used_depots_num = sum(1 for d in range(Nc, Nc + Nf) if frequent_depot[d] > 0)
        # 存在Nd个使用频率高的仓库
        if used_depots_num >= Nd:
            # 选择使用频率最高的Nd个仓库
            top_depots = sorted(frequent_depot, key=frequent_depot.get, reverse=True)[:Nd]
            return repair_routes(top_depots, routes)
        else:
            # 频率高的仓库不足Nd个 → 缺少的仓库随机选择
            top_depots = sorted(frequent_depot, key=frequent_depot.get, reverse=True)[:used_depots_num]
            unused_depots = [d for d in range(Nc, Nc + Nf) if d not in top_depots]
            top_depots += random.sample(unused_depots, k=(Nd - used_depots_num))
            return repair_routes(top_depots, routes)
    else:
        # 随机选择Nd个仓库
        random_depots = random.sample(range(Nc, Nc + Nf), Nd)
        return repair_routes(random_depots, routes)


# ****** 2. 修复路径容量约束 ******
def repair_vehicle_capacity(routes, distance_matrix, cus_demand, vehicle_capacity):
    """修复车辆容量约束问题"""
    penalty = 10000
    best_routes = []
    best_obj = float('inf')

    # 2-opt*邻域搜索
    for i in range(len(routes) - 1):
        route_a = routes[i]
        for a in range(len(route_a) - 1):  # 从第一个客户开始

            for j in range(i + 1, len(routes)):
                route_b = routes[j]
                for b in range(len(route_b) - 1):  # 从第一个客户开始

                    # 交换路径尾部
                    new_routes = [r[:] for r in routes]  # 深拷贝
                    new_routes[i] = route_a[:a + 1] + route_b[b + 1:-1] + [route_a[-1]]
                    new_routes[j] = route_b[:b + 1] + route_a[a + 1:-1] + [route_b[-1]]

                    # 计算目标值和总超载量
                    obj, overload = ObjectiveFunction_Penalty(
                        new_routes, distance_matrix, cus_demand, vehicle_capacity, penalty)

                    if overload == 0:  # 为可行解
                        return new_routes

                    if obj < best_obj:
                        best_routes, best_obj = new_routes, obj

    return best_routes


def Repair(paths, depot_frequency, distance_matrix, cus_demand, capacity, Nc, Nf, Nd):
    """检查路径的约束违背情况，并修复"""
    # 检查并修复仓库约束
    depots_used = {path[0] for path in paths}
    if len(depots_used) > Nd:
        paths = repair_violate_depot(paths, depot_frequency, distance_matrix, Nc, Nf, Nd)
        print('*'*5 + f'修复了仓库数量约束' + '*'*5)

    # 检查并修复容量约束
    overloaded = any(
        sum(cus_demand[customer] for customer in path[1:-1]) > capacity for path in paths)
    if overloaded:
        paths = repair_vehicle_capacity(paths, distance_matrix, cus_demand, capacity)
        print('*'*5 + f'修复了车辆运载约束' + '*'*5)

    return paths



# -------------------- 4. 目标函数值计算(可行解 + 不可行解) --------------------
def ObjectiveFunction(routes, distance_matrix):
    """计算所有路径的总行驶距离"""
    total_latency = 0
    for route in routes:
        route_latency = 0
        for city1, city2 in zip(route[:-1], route[1:-1]):
            route_latency += distance_matrix[city1, city2]
            total_latency += route_latency
    return total_latency


def ObjectiveFunction_Penalty(routes, distance_matrix, cus_demand, vehicle_capacity, beta):
    """计算带容量惩罚的总成本"""
    total_latency = ObjectiveFunction(routes, distance_matrix)
    total_overload = 0

    for route in routes:
        route_demand = sum(cus_demand[city] for city in route[1:-1])
        overload = route_demand - vehicle_capacity
        if overload > 0:
            total_overload += overload

    return total_latency + total_overload * beta, total_overload



# -------------------- 5. 种群变异算子(仓库交换算子 + 路径弹出算子) --------------------
def Mutation(routes, mutation_probability, mutation_length, Nc, Nf):
    """变异：仓库交换算子 + 路径弹出算子"""
    if random.random() < mutation_probability:  # 以mp的概率进行变异
        print('*'*5 + f'发生了变异' + '*'*5)
        for _ in range(mutation_length):    # 将两个变异算子同时执行ml次
            # ******** 1. 仓库交换算子 ********
            depot_used = {depot: [] for depot in range(Nc, Nc + Nf)}
            # 记录仓库使用情况: 某仓库在某路径被使用
            for i, route in enumerate(routes):
                depot_used[route[0]].append(i)

            # 选择要交换的仓库
            opened = [k for k, v in depot_used.items() if v]
            closed = [k for k, v in depot_used.items() if not v]
            opened_close, closed_open = random.choice(opened), random.choice(closed)

            # 交换仓库
            for i in depot_used[opened_close]:
                routes[i][0] = routes[i][-1] = closed_open
                depot_used[closed_open].append(i)  # 废弃 → 启用 状态更新

            # 将被替换的仓库使用情况重置
            depot_used[opened_close] = []

            # ******** 2. 路径弹出算子 ********
            num_paths = min(len(routes), 3)  # 避免路径数少于3
            change_paths = random.sample(range(len(routes)), num_paths)  # 选择进行交换的三条路径
            customer_indices = [random.randrange(1, len(routes[p]) - 1) for p in change_paths]

            # 执行交换(逆时针循环)
            customer = [routes[change_paths[i]][customer_indices[i]] for i in range(num_paths)]
            for i in range(num_paths):
                routes[change_paths[i]][customer_indices[i]] = customer[(i + 1) % num_paths]

    return routes



# -------------------- 6. delta最近邻 + 七个邻域算子 --------------------
def delta_neighbors(Nc, cost_matrix, delta):
    """
    寻找每个客户的最近 delta 个顶点

    参数：
    Nc: 客户数量
    cost_matrix: 距离矩阵
    delta: 最近邻的个数

    返回：
    客户最近顶点的字典
    """
    n = cost_matrix.shape[0]
    customers = list(range(Nc))  # 限制建立最近邻的范围（客户集合，不含仓库）
    vertices = list(range(n))  # 允许作为邻居的点集合（仓库 + 客户）

    Vset = set(vertices)  # 用于检查某个点是否在允许的邻居集合中
    N = {}  # 设定字典，放置客户邻域集合
    for u in customers:
        order = np.argsort(cost_matrix[u])  # 从近到远，返回一个索引数组
        neigh = [v for v in order if v != u and v in Vset]
        N[u] = neigh[:delta]
    return N


# ******** Relocate算子(将一个客户放到同/不同路径的近邻位置)
def relocate_delta(routes, N):
    # 随机选择两条路径(r1要求长度大于3，能有客户点被拿出)
    r1_idx = random.choice([idx for idx in range(len(routes)) if len(routes[idx]) >= 4])
    # 随机选择r2_idx，允许选择与r1_idx相同的路径
    r2_idx = random.choice(range(len(routes)))

    new_routes = copy.deepcopy(routes)

    i = random.randrange(1, len(routes[r1_idx]) - 1)
    client = routes[r1_idx][i]
    r2 = routes[r2_idx]

    cand_pos = []
    if r1_idx == r2_idx:
        valid_positions = [x for x in range(1, len(r2)) if x not in {i, i + 1}]

        for pos in valid_positions:
            if r2[pos - 1] in N.get(client, []) or r2[pos] in N.get(client, []):
                cand_pos.append(pos)  # pos + 1 是插入的位置

        if not cand_pos:
            cand_pos = valid_positions
    else:
        # 遍历 r2 中的每个位置，查找候选插入位置
        for pos in range(1, len(r2)):
            # 检查当前邻接位置的客户是否属于 client 的邻接客户
            if r2[pos - 1] in N.get(client, []) or r2[pos] in N.get(client, []):
                cand_pos.append(pos)

        # 如果没有候选位置，默认选择
        if not cand_pos:
            cand_pos = list(range(1, len(r2)))

    new_routes[r1_idx].pop(i)
    j = random.choice(cand_pos)

    # 修正插入位置
    if r1_idx == r2_idx and i < j:
        j -= 1

    new_routes[r2_idx].insert(j, client)

    return new_routes


# ******** SWAP算子(交换两个结点的位置) ********
def swap_delta(routes, N):
    r1_idx, r2_idx = random.choices(range(len(routes)), k=2)

    new_routes = copy.deepcopy(routes)

    # 从第一条路径随机选择一个结点(仓库/客户)
    i = random.randrange(len(routes[r1_idx]))

    # 当结点为仓库 → 仅交换路径的仓库
    if i in (0, len(routes[r1_idx]) - 1):
        new_routes[r1_idx][0], new_routes[r2_idx][0] = new_routes[r2_idx][0], new_routes[r1_idx][0]
        new_routes[r1_idx][-1], new_routes[r2_idx][-1] = new_routes[r2_idx][-1], new_routes[r1_idx][-1]
        return routes

    # 当结点为客户
    client = routes[r1_idx][i]

    if r1_idx == r2_idx:
        # 路径客户数量不符合要求
        if len(routes[r1_idx]) <= 3:
            avail_routes_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4]
            # 路径的客户数超过两个
            if avail_routes_idx:
                r1_idx = r2_idx = random.choice(avail_routes_idx)
                client = routes[r1_idx][i]
                valid_positions = [x for x in range(1, len(routes[r1_idx]) - 1) if x != i]
                # 记录临近客户的交换位置
                cand_pos = [pos for pos in valid_positions if
                            routes[r2_idx][pos - 1] in N.get(client, []) or
                            routes[r2_idx][pos + 1] in N.get(client,[])]
            else:  # 无路径的客户数超过两个
                r1_idx, r2_idx = random.sample(range(len(routes)), k=2)
                cand_pos = list(range(1, len(routes[r2_idx]) - 1))
        else:  # 路径客户数量符合要求
            valid_positions = [x for x in range(1, len(routes[r1_idx]) - 1) if x != i]
            # 记录临近客户的交换位置
            cand_pos = [pos for pos in valid_positions if
                        routes[r2_idx][pos - 1] in N.get(client, []) or routes[r2_idx][pos + 1] in N.get(client, [])]

        if not cand_pos:
                cand_pos = [x for x in range(1, len(routes[r1_idx]) - 1) if x != i]
    else:
        cand_pos = list(range(1, len(routes[r2_idx]) - 1))

    # 执行swap操作
    j = random.choice(cand_pos)

    new_routes[r1_idx][i], new_routes[r2_idx][j] = new_routes[r2_idx][j], new_routes[r1_idx][i]
    return new_routes


# ******** 2-opt/2-opt*算子(路径内一段客户反转/两个路径之间的客户交换) ********
def two_opt_delta(routes, N):
    # 查找所有长度大于等于4的路径
    valid_routes = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4]

    if len(valid_routes) < 2:
        # 如果没有两条符合条件的路径，则直接返回原路径
        return routes

    # 随机选择两条路径（可能相同）
    r1_idx, r2_idx = random.choices(valid_routes, k=2)
    new_routes = copy.deepcopy(routes)

    def reverse_route_segment(route, i, j):
        """反转路径中i到j之间的客户"""
        route[i:j + 1] = route[i:j + 1][::-1]

    def swap_customers_in_paths(r1, r2, i, j):
        """交换路径中的客户"""
        tail1 = r1[i + 1:-1]
        tail2 = r2[j + 1:-1]
        new_r1 = r1[:i + 1] + tail2 + [r1[0]]
        new_r2 = r2[:j + 1] + tail1 + [r2[0]]
        return new_r1, new_r2

    # 处理路径相同的情况
    if r1_idx == r2_idx:
        if len(routes[r1_idx]) <= 3:
            # 路径长度小于等于3时，选择其他路径交换
            avail_routes_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4]
            if avail_routes_idx:
                r1_idx = random.choice(avail_routes_idx)
                route = new_routes[r1_idx]
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                reverse_route_segment(route, i, j)
            else:
                r2_idx = random.choice([u for u in range(len(routes)) if u != r1_idx])
                r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
                i = random.randint(1, len(r1) - 2)
                client = r1[i]
                cand_pos = [pos for pos in range(1, len(r2) - 1) if r2[pos] in N.get(client, [])]
                j = random.choice(cand_pos) if cand_pos else random.randint(1, len(r2) - 2)
                new_routes[r1_idx], new_routes[r2_idx] = swap_customers_in_paths(r1, r2, i, j)
        else:
            route = new_routes[r1_idx]
            i, j = sorted(random.sample(range(1, len(route) - 1), 2))
            reverse_route_segment(route, i, j)
    else:
        # 两条路径分别交换
        if len(routes[r1_idx]) == 3 and len(routes[r2_idx]) == 3:
            r2_idx = random.sample([u for u in range(len(routes)) if u != r1_idx], k=1)[0]

        r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
        i = random.randint(1, len(r1) - 2)
        client = r1[i]
        cand_pos = [pos for pos in range(1, len(r2) - 1) if r2[pos] in N.get(client, [])]
        j = random.choice(cand_pos) if cand_pos else random.randint(1, len(r2) - 2)
        new_routes[r1_idx], new_routes[r2_idx] = swap_customers_in_paths(r1, r2, i, j)

    return new_routes



# ******** 2-relocate算子(将一段弧整体移动到相同或不同的路径) ********
def two_relocate_delta(routes, N):
    # 随机选择两条路径(r1要求长度大于4，能有弧被拿出)
    r1_idx = random.choice([idx for idx in range(len(routes)) if len(routes[idx]) >= 5])
    # 随机选择r2_idx，允许选择与r1_idx相同的路径
    r2_idx = random.choice(range(len(routes)))

    new_routes = copy.deepcopy(routes)

    # 随机选择弧
    i = random.randrange(1, len(routes[r1_idx]) - 2)
    arc = new_routes[r1_idx][i:i + 2]
    s0, s1 = set(N.get(arc[0], [])), set(N.get(arc[1], []))

    if r1_idx == r2_idx:
        # 计算候选插入位置
        valid_positions = [pos for pos in range(1, len(routes[r1_idx])) if pos not in {i, i + 1, i + 2}]
        # 路径相同的情况下，寻找符合条件的插入位置
        cand_pos = [pos for pos in valid_positions if
                    routes[r2_idx][pos - 1] in s0 or routes[r2_idx][pos] in s1]  # 检查邻近客户
        if not cand_pos:
            # 不存在邻居 → 随机选择一个不同于原位置的插入位置
            cand_pos = valid_positions
    else:
        # 路径不同的情况下，寻找符合条件的插入位置
        cand_pos = [pos for pos in range(1, len(routes[r2_idx]))
                    if routes[r2_idx][pos - 1] in s0 or routes[r2_idx][pos] in s1]

        if not cand_pos:
            # 不存在邻居 → 随机位置
            cand_pos = [x for x in range(1, len(routes[r2_idx]))]

    # 随机选择一个插入位置
    j = random.choice(cand_pos)
    # 确保删除位置正确
    if r1_idx == r2_idx and j < i:
        i += 2

    # 执行 relocate
    new_routes[r2_idx] = new_routes[r2_idx][:j] + arc + new_routes[r2_idx][j:]
    new_routes[r1_idx] = new_routes[r1_idx][:i] + new_routes[r1_idx][i + 2:]

    return new_routes


# ******** Node-Arc swap算子(在相同/不同路径交换单个客户与一个弧) ********
def NodeArc_swap_delta(routes, N):
    # 随机选择两条路径（可能相同）
    r1_idx, r2_idx = random.choices(range(len(routes)), k=2)
    new_routes = copy.deepcopy(routes)

    # 随机选择客户(不包括起始仓库) → 预留弧位置
    i = random.randrange(1, len(routes[r1_idx]) - 1)
    client = routes[r1_idx][i]  # 被交换的客户
    u_neighbors = N.get(client, [])  # 获取客户的近邻

    # 获取有效的插入位置，避免原位置，并确保与近邻相关
    def get_valid_positions(r_idx, excluded_idx):
        valid_positions = [x for x in range(1, len(routes[r_idx]) - 2) if x not in excluded_idx]
        return [pos for pos in valid_positions if routes[r_idx][pos - 1] in u_neighbors]

    # 判断路径相同与不同的情况
    if r1_idx == r2_idx:
        # 路径相同且长度小于等于4时，随机选择符合条件的路径
        if len(routes[r1_idx]) <= 4:
            avail_routes_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 5]
            if avail_routes_idx:
                r1_idx = r2_idx = random.choice(avail_routes_idx)
                client = routes[r1_idx][i]

                # 特殊情况，选择不同位置
                if len(routes[r1_idx]) == 5 and i == 2:
                    i = random.choice([1, 3])
                    client = routes[r1_idx][i]

                # 查找插入位置
                cand_pos = get_valid_positions(r2_idx, {i - 1, i})
                if not cand_pos:
                    cand_pos = [x for x in range(1, len(routes[r1_idx]) - 2) if x not in {i - 1, i}]
            else:
                avail_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4 and idx != r1_idx]
                if avail_idx:
                    r2_idx = random.choice(avail_idx)
                else:
                    r2_idx = random.choice([idx for idx in range(len(routes)) if idx != r1_idx])
                    r1_idx, r2_idx = r2_idx, r1_idx
                    i = random.randrange(1, len(routes[r1_idx]) - 1)
                    client = routes[r1_idx][i]  # 被交换的客户
                    u_neighbors = N.get(client, [])  # 获取客户的近邻
                cand_pos = get_valid_positions(r2_idx, {})
                if not cand_pos:
                    cand_pos = [x for x in range(1, len(routes[r2_idx]) - 2)]
        else:
            if len(routes[r1_idx]) == 5 and i == 2:
                i = random.choice([1, 3])
                client = routes[r1_idx][i]
            cand_pos = get_valid_positions(r2_idx, {i - 1, i})
            if not cand_pos:
                cand_pos = [x for x in range(1, len(routes[r1_idx]) - 2) if x not in {i - 1, i}]
    else:
        # 第二条路径必须有两个客户
        if len(routes[r2_idx]) <= 3:
            avail_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4 and idx != r1_idx]
            if avail_idx:
                r2_idx = random.choice(avail_idx)
            else:
                r2_idx = random.choice([idx for idx in range(len(routes)) if idx != r1_idx])
                r1_idx, r2_idx = r2_idx, r1_idx
                i = random.randrange(1, len(routes[r1_idx]) - 1)
                client = routes[r1_idx][i]  # 被交换的客户
                u_neighbors = N.get(client, [])  # 获取客户的近邻
        cand_pos = get_valid_positions(r2_idx, {})
        if not cand_pos:
            cand_pos = [x for x in range(1, len(routes[r2_idx]) - 2)]

    # 获取被交换的弧
    j = random.choice(cand_pos)
    arc = new_routes[r2_idx][j:j + 2]

    if r1_idx == r2_idx and i < j:
        j += 1

    # 交换操作：把客户 client 移到目标路径 r2 中的插入位置 j
    new_routes[r1_idx] = new_routes[r1_idx][:i] + arc + new_routes[r1_idx][i + 1:]
    new_routes[r2_idx] = new_routes[r2_idx][:j] + [client] + new_routes[r2_idx][j + 2:]

    return new_routes



# ******** Arc-Arc swap算子(在相同/不同路径交换两个弧) ********
def ArcArc_swap_delta(routes, N):
    # 随机选择两条路径(至少有两个客户)
    valid_routes = [idx for idx in range(len(routes)) if len(routes[idx]) >= 4]
    if len(valid_routes) < 2:
        return routes  # 没有足够的路径可以交换，直接返回原路径

    r1_idx, r2_idx = random.choices(valid_routes, k=2)
    new_routes = copy.deepcopy(routes)
    # 随机选择弧
    i = random.randrange(1, len(routes[r1_idx]) - 2)
    arc1 = new_routes[r1_idx][i:i + 2]
    s0, s1 = set(N.get(arc1[0], [])), set(N.get(arc1[1], []))

    def get_valid_positions(r_idx, i, s0, s1, is_same_path=True):
        """获取有效插入位置，避免原位置，并确保与近邻相关"""
        if is_same_path:
            valid_positions = [x for x in range(1, len(routes[r_idx]) - 2) if x not in {i - 1, i, i + 1}]
            return [pos for pos in valid_positions if (routes[r_idx][pos - 1] in s0 or routes[r_idx][pos + 2] in s1) and pos not in {i, i + 1}]
        else:
            valid_positions = [x for x in range(1, len(routes[r_idx]) - 2)]
            return [pos for pos in valid_positions if routes[r_idx][pos - 1] in s0 or routes[r_idx][pos + 2] in s1]

    if r1_idx == r2_idx:
        # 路径相同的处理
        if len(routes[r1_idx]) <= 5:
            avail_routes_idx = [idx for idx in range(len(routes)) if len(routes[idx]) >= 6]
            if avail_routes_idx:
                r1_idx = r2_idx = random.choice(avail_routes_idx)
                arc1 = new_routes[r1_idx][i:i + 2]
                if len(routes[r1_idx]) == 6 and i == 2:
                    i = random.choice([1, 3])
                    arc1 = routes[r1_idx][i:i + 2]
                cand_pos = get_valid_positions(r2_idx, i, s0, s1)
                if not cand_pos:
                    cand_pos = [x for x in range(1, len(routes[r1_idx]) - 2) if x not in {i - 1, i, i + 1}]
            else:
                r2_idx = random.choice([idx for idx in range(len(routes)) if idx != r1_idx])
                cand_pos = get_valid_positions(r2_idx, i, s0, s1, is_same_path=False)
                if not cand_pos:
                    cand_pos = [x for x in range(1, len(routes[r2_idx]) - 2)]
        else:
            if len(routes[r1_idx]) == 6 and i == 2:
                i = random.choice([1, 3])
                arc1 = routes[r1_idx][i:i + 2]
            cand_pos = get_valid_positions(r2_idx, i, s0, s1)
            if not cand_pos:
                cand_pos = [x for x in range(1, len(routes[r2_idx]) - 2) if x not in {i - 1, i, i + 1}]
    else:
        # 路径不同的处理
        cand_pos = get_valid_positions(r2_idx, i, s0, s1, is_same_path=False)
        if not cand_pos:
            cand_pos = [x for x in range(1, len(routes[r2_idx]) - 2)]

    # 获取被交换的弧
    j = random.choice(cand_pos)
    arc2 = new_routes[r2_idx][j:j + 2]

    # 交换弧
    new_routes[r1_idx] = new_routes[r1_idx][:i] + arc2 + new_routes[r1_idx][i + 2:]
    new_routes[r2_idx] = new_routes[r2_idx][:j] + arc1 + new_routes[r2_idx][j + 2:]

    return new_routes




# ******** swap*算子(极角扇形+最佳插入位置) ********
def swap_star(routes, coord_x, coord_y, cost_matrix):

    # ******** 1. 选择平面直角坐标系原点 → 路径最多的仓库/相等时随机 ********
    # 计算仓库使用次数，随机选择使用次数最高的仓库
    depot_used = Counter(route[0] for route in routes)  # 记录仓库使用次数
    max_frequency = max(depot_used.values())
    frequent_depots = [depot for depot, frequency in depot_used.items() if frequency == max_frequency]
    site = random.choice(frequent_depots)


    # ******** 2. 计算每个客户的角度 ********
    site_x, site_y = coord_x[site], coord_y[site]  # 原点坐标
    # 使用列表推导式计算所有点的角度，并确保在[0, 2π)范围内
    customer_angle = np.array([
        math.atan2(cy - site_y, cx - site_x) % (2 * math.pi)
        for cx, cy in zip(coord_x, coord_y)
    ])

    # ******** 3. 记录每条路径的扇面 ********
    route_sector = []
    for route in routes:
        route_angles = customer_angle[route]  # 向量化获取角度
        # 移除原点仓库路径的仓库
        if route[0] == site:
            route_angles = route_angles[1:-1]
        start = min(route_angles)
        end = max(route_angles)
        route_sector.append((start, end))  # 记录路径扇形面起始、终止的角度

    # ******** 4. 记录每条路径的临近路径(避免重复计算) ********
    near_routes = {route_index: [] for route_index in range(len(routes))}
    for i in range(len(routes) - 1):
        for j in range(i + 1, len(routes)):
            # 路径的极角扇形相交 → 记录
            if route_sector[j][0] <= route_sector[i][1] and route_sector[i][0] <= route_sector[j][1]:
                near_routes[i].append(j)
                near_routes[j].append(i)

    # ******* 5. 确定交换的两条路径和对应客户，寻找最佳插入位置 ********
    new_routes = copy.deepcopy(routes)
    # 确定路径
    avail_route = [k for k, v in near_routes.items() if v]
    if avail_route:
        r1_idx = random.choice(avail_route)
        r2_idx = random.choice(near_routes[r1_idx])
    else:
        r1_idx, r2_idx = random.sample(range(len(routes)), k=2)
    route_one, route_two = new_routes[r1_idx], new_routes[r2_idx]

    # 确定客户
    cus1_idx = random.randrange(1, len(route_one) - 1)
    cus2_idx = random.randrange(1, len(route_two) - 1)
    customer1, customer2 = route_one[cus1_idx], route_two[cus2_idx]

    # **** 三个最佳插入位置 ****
    # 计算客户1插入路径2的三个最佳位置的花销
    cost_c1_r2 = sorted([
        (a, b, cost_matrix[a][customer1] + cost_matrix[customer1][b] - cost_matrix[a][b])
        for a, b in zip(route_two, route_two[1:])
    ], key=lambda x: x[2])[:3]
    # 计算客户2插入路径1的三个最佳位置的花销
    cost_c2_r1 = sorted([
        (a, b, cost_matrix[a][customer2] + cost_matrix[customer2][b] - cost_matrix[a][b])
        for a, b in zip(route_one, route_one[1:])
    ], key=lambda x: x[2])[:3]

    # ******** 6. 确定交换位置 ********
    def find_best_site(site_cost, c1, c2_idx, r2):
        # 插入位置的前驱和后继
        pred, succe = r2[c2_idx - 1], r2[c2_idx + 1]
        ori_cost = cost_matrix[pred][c1] + cost_matrix[c1][succe] - cost_matrix[pred][succe]
        # 最佳的三个插入位置
        for i, j, c in site_cost:
            if i != r2[c2_idx] and j != r2[c2_idx] and c < ori_cost:
                # 比较最佳位置花销和被更换客户位置花销
                    return (i, j)

        return (pred, succe)

    c2_r1_site = find_best_site(cost_c2_r1, customer2, cus1_idx, route_one)
    c1_r2_site = find_best_site(cost_c1_r2, customer1, cus2_idx, route_two)

    # ******** 7. 对路径执行交换操作 ********
    def change_customer(c1, c2, r1, s1):
        r1.remove(c1)
        idx = r1.index(s1[0])  # 找到插入位置
        r1.insert(idx + 1, c2)
        return r1

    route_one = change_customer(customer1, customer2, route_one, c2_r1_site)
    route_two = change_customer(customer2, customer1, route_two, c1_r2_site)
    # route_one, route_two是对原列表中对象的引用，这里修改已经生效
    return new_routes



# -------------------- 7. 初始化Q表和R矩阵 + RL-SOVND + 更新最优邻域解的仓库使用频率 --------------------
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

            # 应对初始情况下的判断结构
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



# -------------------- 8. 根据海明距离更新种群 --------------------
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


# -------------------- 9. 解停滞时的半数种群替换 --------------------
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
    append_idx = range(choose_length)

    # 3. 补足种群 (自适应存储器/随机解生成方法)
    count_store = 0  # 记录从自适应存储器取用解的次数
    count = 0  # 在新种群中添加解的数量
    while count < math.ceil(pop_length / 2):  # 限制只放原种群一半数量的新种群
        if count_store < choose_length and random.random() < 0.5:
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



def main(file_name):
    # 读取CSV文件
    def read_from_csv(filename='vrp_data.csv'):
        """从CSV文件读取VRP数据"""
        data = {
            'parameters': {},
            'x_coords': [],
            'y_coords': [],
            'demand': [],
            'distance': None
        }

        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)

            # 找到各部分的行索引
            coord_start = -1
            demand_start = -1
            distance_start = -1

            for i, line in enumerate(lines):
                if not line:  # 跳过空行
                    continue
                if line[0] == 'Coordinates':
                    coord_start = i
                elif line[0] == 'Demands':
                    demand_start = i
                elif line[0] == 'Distance_Matrix':
                    distance_start = i

            # 1. 读取参数部分 (在Coordinates之前的行)
            for i in range(coord_start):
                if lines[i] and len(lines[i]) >= 2:
                    if lines[i][0] != 'Parameter':
                        key = lines[i][0]
                        try:
                            value = int(lines[i][1])
                        except ValueError:
                            value = float(lines[i][1]) if '.' in lines[i][1] else lines[i][1]
                        data['parameters'][key] = value

            # 2. 读取坐标数据
            if coord_start != -1:
                # 跳过表头行 (Coordinates, Point_Type,Index,X,Y, 以及可能的空行)
                i = coord_start + 2  # 跳过'Coordinates'行和'Point_Type,Index,X,Y'行
                while i < len(lines) and lines[i]:
                    if len(lines[i]) >= 4 and lines[i][0] in ['Customer', 'Depot']:
                        try:
                            x = float(lines[i][2])  # 修改为float
                            y = float(lines[i][3])  # 修改为float
                            data['x_coords'].append(x)
                            data['y_coords'].append(y)
                        except (ValueError, IndexError) as e:
                            print(f"解析坐标错误 第{i + 1}行: {lines[i]}, 错误: {e}")
                    i += 1

            # 3. 读取需求数据
            if demand_start != -1:
                i = demand_start + 2  # 跳过'Demands'行和'Customer_Index,Demand'行
                while i < len(lines) and lines[i]:
                    if len(lines[i]) >= 2:
                        try:
                            demand_val = int(lines[i][1])
                            data['demand'].append(demand_val)
                        except (ValueError, IndexError) as e:
                            print(f"解析需求错误 第{i + 1}行: {lines[i]}, 错误: {e}")
                    i += 1

            # 4. 读取距离矩阵
            if distance_start != -1:
                i = distance_start + 2  # 跳过'Distance_Matrix'行和表头行
                matrix_rows = []

                # 确定矩阵大小
                # 从表头行读取节点数量
                header_row = lines[distance_start + 1]
                n = len(header_row) - 1  # 减去第一列'From/To'

                # 读取矩阵数据
                row_count = 0
                while i < len(lines) and lines[i] and lines[i][0].startswith('Node_'):
                    if row_count >= n:
                        break

                    row_data = lines[i][1:1 + n]  # 跳过第一列节点名
                    try:
                        row_values = [int(val) if val != '' else 0 for val in row_data]
                        matrix_rows.append(row_values)
                    except ValueError as e:
                        print(f"解析距离矩阵行错误 第{i + 1}行: {lines[i]}, 错误: {e}")

                    row_count += 1
                    i += 1

                if matrix_rows:
                    data['distance'] = np.array(matrix_rows)

        return data

    instance_data = read_from_csv(file_name)

    """主执行函数"""
    start_time = time.time()
    print("开始执行RLHEA方法运算...")

    # 参数设置
    Nv = instance_data['parameters']['Nv']  # 车辆数
    Nf = instance_data['parameters']['Nf']  # 仓库数
    Nc = instance_data['parameters']['Nc']  # 客户数
    Nd = instance_data['parameters']['Nd']  # 随机生成开放的仓库数
    capacity = instance_data['parameters']['capacity']  # 车辆运载能力上限
    distance = instance_data['distance']
    demand = instance_data['demand']
    x_coord = instance_data['x_coords']
    y_coord = instance_data['y_coords']

    Ir = 10  # 迭代停滞上限
    max_generation = 60  # 迭代次数上限

    population_scale = 20  # 种群规模
    mp = 0.1  # 变异概率
    ml = 2  # 变异次数
    epsilon = 0.7  # epsilon贪心策略概率
    alpha = 0.2  # 学习率
    gamma = 0.85  # 折扣率(Q值更新)
    delta = 8                                                                                                                                                                                                                            # 最近邻数量
    xi = 0.95  # 折扣系数(奖励值更新)
    up = 2  # beta更新滑动窗口
    psi = 0.55  # 适应度函数计算参数
    adaptive_store_length = 30  # 自适应记忆存储器大小

    # 邻域算子
    neighbor_structures = {
        "relocate": (relocate_delta, 0),
        "swap": (swap_delta, 1),
        "two_opt": (two_opt_delta, 2),
        "two_relocate": (two_relocate_delta, 3),
        "NodeArc_swap": (NodeArc_swap_delta, 4),
        "ArcArc_swap": (ArcArc_swap_delta, 5),
        "swap_star": (swap_star, 6),
    }
    neighbor_delta = delta_neighbors(Nc, distance, delta)
    depot_frequency = {depot: 0 for depot in range(Nc, Nc + Nf)}  # 高质量解中仓库的使用频率 (字典)
    adaptive_store = []  # 自适应记忆存储器(存放最近adaptive_stroe_length个由RL-SOVND找出的局部最优解)


    In = 0  # 初始化迭代连续计数器
    Ir_count = 0  # 进度停滞调整次数
    # 建立初始化的Q表和奖赏矩阵
    Q_table, R_matrix = build_Qtable_Reward(len(neighbor_structures), len(neighbor_structures))

    # 种群初始化
    population = []
    population_obj = []
    count = 0
    while count < population_scale:
        print(f'第{count}个初始化种群')
        # 随机选择路径生成方法
        if random.random() < 0.5:
            routes = initial_greed_method(Nf, Nd, Nc, Nv, distance)
        else:
            routes = initial_random_method(Nf, Nd, Nc, Nv)
        print(f'初始化解为{routes}')

        a, b = validate_numbers_in_range(routes)
        if not a or not b:
            return None, None

        # 使用RL-SOVND改进初始化的解
        routes, obj, Q_table, R_matrix = RL_SOVND(routes, Q_table, R_matrix, neighbor_structures, neighbor_delta, distance, demand, x_coord, y_coord,
            float('inf'), Nd, capacity, up, epsilon, alpha, gamma, xi)

        if routes == None:
            return None, None

        # 在无相同解(dis_apop != 0) → 添加进种群
        if dist_apop(routes, population, Nc, Nf):
            population.append(routes)
            population_obj.append(obj)
            count += 1

    print(f'初始化的种群为：')
    print(population)

    # 记录当前最优可行解
    best_obj = min(population_obj)
    best_routes = population[np.argmin(population_obj)]

    print(f'初始的最优目标函数值为{best_obj}')
    print(f'初始的最优解为{best_routes}')

    # 迭代
    for x in range(max_generation):
        print("*"*20 + f'第{x}次迭代' + "*"*20)

        # 每次迭代选择三个种群作为父群
        Sa_idx, Sb_idx, Sc_idx = random.sample(range(len(population)), 3)
        Sa, Sb, Sc = population[Sa_idx], population[Sb_idx], population[Sc_idx]
        print(f'选出的父群为：')
        print(Sa)
        print(Sb)
        print(Sc)

        # 使用多亲本边装配交叉生成子代
        So = MPEAX_Offspring(Nf, Nc, Sa, Sb, distance)
        print(f'产生的中间解为{So}')

        # 如果子代与第三个亲本相同，重新选择Sc
        if not dist_ab(So, Sc, Nc, Nf):
            print(f'中间解与Sc相同 → 更换Sc')
            available_indices = [i for i in range(len(population))
                                 if i not in (Sa_idx, Sb_idx, Sc_idx) and
                                 dist_ab(So, population[i], Nc, Nf)]
            Sc = population[random.choice(available_indices)]
            print(f'新的Sc为{Sc}')

        a, b = validate_numbers_in_range(So)
        if not a or  not b:
            return None, None

        S = MPEAX_Offspring(Nf, Nc, So, Sc, distance)
        print(f'产生的子代为{S}')

        a, b = validate_numbers_in_range(S)
        if not a or not b:
            return None, None

        # 判断是否为不可行解(仓库数量/路径负载) → 修复
        S = Repair(S, depot_frequency, distance, demand, capacity, Nc, Nf, Nd)

        # 路径变异
        S = Mutation(S, mp, ml, Nc, Nf)

        a, b = validate_numbers_in_range(S)
        if not a or not b:
            return None, None

        # 计算邻域解
        Sl, Sl_obj, Q_table, R_matrix = RL_SOVND(S, Q_table, R_matrix, neighbor_structures, neighbor_delta,
                          distance, demand, x_coord, y_coord, best_obj, Nd, capacity, up, epsilon, alpha, gamma, xi)
        if Sl == None:
            return None, None
        print(f'邻域解目标函数值为{Sl_obj}')
        print(f'邻域解为{Sl}')

        # 使用频率更新
        depot_frequency = update_frequency(Sl, depot_frequency)

        # 更新自适应存储器M
        current_adaptive_length = len(adaptive_store)
        if current_adaptive_length < adaptive_store_length:
            adaptive_store.append(Sl)
            current_adaptive_length += 1
        else:
            adaptive_store.pop(0)
            adaptive_store.append(Sl)

        print(f'当前自适应存储器存放数量为{current_adaptive_length}')

        # 最优解更新
        if Sl_obj < best_obj: # type: ignore
            best_routes = Sl
            best_obj = Sl_obj
            In = 0
            print('*'*10 + f'更新了全局最优解' + '*'*10)
            print(f'新的全局最优解为: {best_obj}')
        else:
            In += 1

        # 种群更新
        population = population_update(Sl, population, distance, Nf, Nc, psi)

        # 迭代停滞
        if In > Ir:
            print('*'*5 + f'迭代停滞，种群替换' + '*'*5)
            population, Q_table, R_matrix = ReplacePopulation(population, population_scale, adaptive_store, adaptive_store_length, distance, Nf, Nd, Nc, Nv,
                      Q_table, R_matrix, neighbor_structures, neighbor_delta, demand, x_coord, y_coord, best_obj, capacity, up, epsilon, alpha, gamma, xi)
            Ir_count += 1
            In= 0

    end_time = time.time()
    print(f'仓库使用频率排行：{depot_frequency}')
    depots_used = {path[0] for path in best_routes}
    print(f'仓库使用情况：{depots_used}')
    print(f'种群替换次数为：{Ir_count}')
    print(f'运行时间为: {end_time - start_time}')
    return best_routes, best_obj


if __name__ == "__main__":
    '''
    file = '10-3-3'
    routes, routes_obj = main(file)
    print(routes_obj)
    print(f'最优解为{routes}')
    '''
    file = '10-3-3'
    solution, obj = [], []
    for i in range(5):
        print('*'*25 + f'第{i}次运行' + '*'*25)
        routes, routes_obj = main(file)
        solution.append(routes)
        obj.append(routes_obj)
    for i in range(5):
        print(f'第{i}次运行最优目标函数值为：')
        print(obj[i])
        print(f'最优解为{solution[i]}')