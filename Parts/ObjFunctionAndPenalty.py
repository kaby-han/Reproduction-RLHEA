def ObjectiveFunction(routes, distance_matrix):
    """计算所有路径的总行驶距离"""
    total_distance = 0
    for route in routes:
        total_distance += sum(distance_matrix[city1][city2]
                              for city1, city2 in zip(route, route[1:]))
    return total_distance


def ObjectiveFunction_Penalty(routes, distance_matrix, cus_demand, vehicle_capacity, beta):
    """计算带容量惩罚的总成本"""
    total_distance = ObjectiveFunction(routes, distance_matrix)
    total_overload = 0

    for route in routes:
        route_demand = sum(cus_demand[city] for city in route[1:-1])
        overload = route_demand - vehicle_capacity
        if overload > 0:
            total_overload += overload

    return total_distance + total_overload * beta, total_overload