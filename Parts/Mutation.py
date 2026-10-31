import random

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