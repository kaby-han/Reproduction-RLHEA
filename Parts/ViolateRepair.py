import numpy as np
import random

from ObjFunctionAndPenalty import ObjectiveFunction_Penalty

Nv = 5
Nf = 5
Nc = 20
depot_frequency = {depot: 0 for depot in range(Nc, Nc + Nf)}  # 高质量解中仓库的使用频率 (字典)


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