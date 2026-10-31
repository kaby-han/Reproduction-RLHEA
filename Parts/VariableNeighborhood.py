from collections import Counter
import math
import numpy as np
import random
import copy


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
    customers = list(range(Nc))  # 限制只为哪些 u 建邻域（客户集合，不含仓库）
    vertices = list(range(n))  # 允许作为邻居的点集合（仓库+客户）

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
    # 计算客户1插入路径2的三个最佳位置
    cost_c1_r2 = sorted([
        (a, b, cost_matrix[a][customer1] + cost_matrix[customer1][b] - cost_matrix[a][b])
        for a, b in zip(route_two, route_two[1:])
    ], key=lambda x: x[2])[:3]
    # 计算客户2插入路径1的三个最佳位置
    cost_c2_r1 = sorted([
        (a, b, cost_matrix[a][customer2] + cost_matrix[customer2][b] - cost_matrix[a][b])
        for a, b in zip(route_one, route_one[1:])
    ], key=lambda x: x[2])[:3]

    # ******** 6. 确定交换位置 ********
    def find_best_site(site_cost, c1, c2_idx, r2):
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
