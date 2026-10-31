import random
import copy
import numpy as np


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

            # 检查 1.回到起点（初始弧到仓库） 2.尾结点相同（回到起点） 3.头结点相同
            if (arc_num >= 2) and (tail == start_tail or head == start_head):
                break

            # 选择下一条边
            tail, head, depot_tag = select_next_edge(tail, head, depot_tag, matrix_a_inter, matrix_b_inter,
                                                     current_from_a, Nc, Nf)
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
