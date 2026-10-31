import random
import numpy as np
from Test.PopulationManagement.pop_update import dist_ab, dist_apop
from data import distance

Nv = 5
Nf = 5
Nc = 20

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
