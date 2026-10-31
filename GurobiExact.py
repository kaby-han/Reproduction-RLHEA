from gurobipy import *
import numpy as np
import csv

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

file_name = '10-3-3'
instance_data = read_from_csv(file_name)

# 参数设置
Nv = instance_data['parameters']['Nv']  # 车辆数
Nf = instance_data['parameters']['Nf']  # 仓库数
Nc = instance_data['parameters']['Nc']  # 客户数
Nd = instance_data['parameters']['Nd']  # 开放的仓库数
capacity = instance_data['parameters']['capacity']  # 车辆运载能力上限
distance = instance_data['distance']  # 开销矩阵
demand = instance_data['demand']  # 客户需求

N = Nf + Nd + Nf  # 总节点数复制车场

# ****** 1. 创建模型 ******
model = Model('LLRP')
model.Params.TimeLimit = 180

# ****** 2. 创建变量 ******
# 定义变量
t = {}  # 车辆抵达客户的时间
x = {}  # 车辆执行弧的0-1变量
f = {}  # 客户由仓库供应的0-1变量
z = {}  # 仓库启用的0-1变量
big_M = 100000

for k in range(Nv):
    for i in range(Nc + Nf):
        # 变量1：车辆k抵达客户i的时间
        t[k, i] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="t_%d_%d" % (k, i))
        for j in range(Nc + Nf):
            # 变量2：车辆k执行弧(i, j)
            if i != j:
                # 以下两个是排除车场到车场的变量
                if i in range(Nc):
                        x[k, i, j] = model.addVar(vtype=GRB.BINARY, name="x_%d_%d_%d" % (k, i, j))
                else:
                    if j in range(Nc):
                        x[k, i, j] = model.addVar(vtype=GRB.BINARY, name="x_%d_%d_%d" % (k, i, j))


for g in range(Nc, Nc + Nf):
    # 变量3：仓库g启用
    z[g] = model.addVar(vtype=GRB.BINARY, name="z_%d" % (g))
    for i in range(Nc):
        # 变量4：仓库g供应客户i
        f[g, i] = model.addVar(vtype=GRB.BINARY, name="f_%d_%d" % (g, i))


# ****** 3. 创建目标函数 ******
obj = 0
for k in range(Nv):
    for i in range(Nc):
        obj += t[k, i]
model.setObjective(obj, GRB.MINIMIZE)


# ****** 4. 创建约束 ******
# 约束1：流平衡约束
for k in range(Nv):
    for i in range(Nc + Nf):
        lh = LinExpr()
        rh = LinExpr()
        for j in range(Nc + Nf):
            if i != j and not (i >= Nc and j >= Nc):
                rh.addTerms(1, x[k, i, j])
        for j in range(Nc + Nf):
            if i != j and not (i >= Nc and j >= Nc):
                lh.addTerms(1, x[k, j, i])
        model.addConstr(lh - rh == 0, "PointFlowConstr_%d_%d" % (k, i))


# 约束2：每个客户都必须被有且只有一辆车服务
for i in range(Nc):
    expr = LinExpr()
    for k in range(Nv):
        for j in range(Nc + Nf):
            if i != j:
                expr.addTerms(1, x[k, i, j])
    model.addConstr(expr == 1, name="CustomerServed_%d" % (i))


# 约束3：每个客户都必须被有且只有一个仓库服务
for i in range(Nc):
    expr = LinExpr()
    for g in range(Nc, Nc + Nf):
        expr.addTerms(1, f[g, i])
    model.addConstr(expr == 1, name="DepotServe_%d" % (i))


# 约束4：路径负载约束
for k in range(Nv):
    lh = LinExpr()
    for i in range(Nc + Nf):
        for j in range(Nc):
            if i != j:
                lh.addTerms(demand[j], x[k, i, j])
    model.addConstr(lh <= capacity, "Capacity_%d" % k)


# 约束5：路径由仓库出发
for k in range(Nv):
    expr = LinExpr()
    for g in range(Nc, Nc + Nf):
        for i in range(Nc):
            expr.addTerms(1, x[k, g, i])
    model.addConstr(expr == 1, name="CustomerInRoute_FromDepot_%d" % k)


# 约束6：与约束5一同决定路径与客户关系
for i in range(Nc):
    for k in range(Nv):
        for g in range(Nc, Nc + Nf):
            expr = LinExpr()
            for u in range(Nc):
                expr.addTerms(1, x[k, g, u])
            for u in range(Nc + Nf):
                if u != i:
                    expr.addTerms(1, x[k, u, i])
            model.addConstr(expr <= 1 + f[g, i], name="CustomerInRoute_FromDepot_%d_%d_%d" % (i, k, g))


# 约束7：仓库启用后才能供应客户
for g in range(Nc, Nc + Nf):
    expr = LinExpr()
    for i in range(Nc):
        expr.addTerms(1, f[g, i])
    model.addConstr(expr <= big_M * z[g], name="DepotOpen_%d" % g)


# 约束8：仓库启用数小于仓库总数
expr = LinExpr()
for g in range(Nc, Nc + Nf):
    expr.addTerms(1, z[g])
model.addConstr(expr <= Nd, name="DepotNumber")


# 约束9：时间推导式
for k in range(Nv):
    for i in range(Nc + Nf):
        for j in range(Nc):
            if i != j:
                # 时间连续性约束
                model.addConstr(
                    t[k, j] >= t[k, i] + distance[i, j] - (1 - x[k, i, j]) * big_M,
                    name="TimeContinuity_%d_%d_%d" % (k, i, j)
                )

# 求解模型
model.optimize()


# 打印求解结果
def print_solution(model, Nv, Nc, Nf, x, z, f, t, distance, demand):
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        print("=" * 60)
        print("求解状态:", model.status)
        print(f"目标函数值: {model.objVal:.2f}")
        print(f"求解时间: {model.Runtime:.2f}秒")
        print(f"最优间隙: {model.MIPGap:.4%}" if model.MIPGap > 0 else "最优间隙: 0.00%")
        print("=" * 60)

        # 打印启用的仓库
        enabled_depots = []
        for g in range(Nc, Nc + Nf):
            if z[g].X > 0.5:
                enabled_depots.append(g)

        print(f"\n启用的仓库 ({len(enabled_depots)}/{Nf}):")
        for g in enabled_depots:
            # 获取该仓库服务的客户
            served_customers = []
            for i in range(Nc):
                if f[g, i].X > 0.5:
                    served_customers.append(i)
            print(f"  仓库 {g - Nc} (索引{g}): 服务{served_customers}")

        print("\n" + "=" * 60)

        # 打印每辆车的路径
        print("\n车辆路径详情:")
        total_distance = 0
        total_customers_served = 0

        for k in range(Nv):
            # 找出车辆k的所有边
            arcs = []
            for i in range(Nc + Nf):
                for j in range(Nc + Nf):
                    if i != j and (k, i, j) in x and x[k, i, j].X > 0.5:
                        arcs.append((i, j))

            if not arcs:
                print(f"\n车辆 {k}: 未使用")
                continue

            # 重建路径
            # 找到起始仓库
            start_depot = None
            for i, j in arcs:
                if i >= Nc:  # 如果起点是仓库
                    start_depot = i
                    break

            if start_depot is None:
                print(f"\n车辆 {k}: 路径不完整")
                continue

            # 重建路径
            path = []
            current_node = start_depot
            visited = set()

            while True:
                path.append(current_node)
                visited.add(current_node)

                # 找到当前节点的下一个节点
                next_node = None
                for i, j in arcs:
                    if i == current_node and j not in visited:
                        next_node = j
                        break

                if next_node is None:
                    break

                current_node = next_node

            # 计算路径长度和负载
            path_distance = 0
            path_demand = 0
            path_customers = []

            for i in range(len(path) - 1):
                from_node = path[i]
                to_node = path[i + 1]
                path_distance += distance[from_node, to_node]

                if to_node < Nc:  # 如果是客户节点
                    path_demand += demand[to_node]
                    path_customers.append(to_node)

            # 打印车辆路径
            print(f"\n车辆 {k}:")
            print(f"  起始仓库: {start_depot - Nc if start_depot >= Nc else start_depot}")
            print(f"  路径: {' -> '.join([f'D{node - Nc}' if node >= Nc else f'C{node}' for node in path])}")
            print(f"  服务客户: {sorted(path_customers)} ({len(path_customers)}个)")
            print(f"  总需求: {path_demand}/{capacity}")
            print(f"  路径距离: {path_distance:.2f}")

            # 打印客户服务时间
            if path_customers:
                print("  客户服务时间:")
                for customer in path_customers:
                    time_val = t[k, customer].X if (k, customer) in t else 0
                    print(f"    客户{customer}: 时间 {time_val:.2f}")

            total_distance += path_distance
            total_customers_served += len(path_customers)

        print("\n" + "=" * 60)
        print("汇总信息:")
        print(f"  启用的仓库数: {len(enabled_depots)}/{Nf}")
        print(f"  使用的车辆数: {sum(1 for k in range(Nv) if any((k, i, j) in x and x[k, i, j].X > 0.5 for i in range(Nc + Nf) for j in range(Nc + Nf) if i != j))}/{Nv}")
        print(f"  服务的客户数: {total_customers_served}/{Nc}")
        print(f"  总服务时间: {model.objVal:.2f}")

        # 检查约束满足情况
        print("\n约束检查:")
        # 检查每个客户是否都被服务
        customers_served = [0] * Nc
        for k in range(Nv):
            for i in range(Nc):
                for j in range(Nc + Nf):
                    if i != j and (k, i, j) in x and x[k, i, j].X > 0.5:
                        customers_served[i] += 1

        all_customers_served = all(count == 1 for count in customers_served)
        print(f"  所有客户都被服务: {'是' if all_customers_served else '否'}")

        # 检查车辆容量约束
        capacity_violations = 0
        for k in range(Nv):
            vehicle_demand = 0
            for i in range(Nc + Nf):
                for j in range(Nc):
                    if i != j and (k, i, j) in x and x[k, i, j].X > 0.5:
                        vehicle_demand += demand[j]
            if vehicle_demand > capacity + 1e-6:  # 添加容差
                capacity_violations += 1

        print(f"  容量约束违反: {capacity_violations}辆车")

    else:
        print("=" * 60)
        print("未找到可行解")
        print(f"求解状态: {model.status}")
        print(f"求解时间: {model.Runtime:.2f}秒")
        print("=" * 60)


# 调用打印函数
print_solution(model, Nv, Nc, Nf, x, z, f, t, distance, demand)