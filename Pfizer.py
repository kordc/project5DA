import matplotlib.pyplot as plt
import numpy as np
from pulp import *
from tqdm import tqdm

# distance matrix
D = [
    [16.160, 24.080, 24.320, 21.120],
    [19.000, 26.470, 27.240, 17.330],
    [25.290, 32.490, 33.420, 12.250],
    [0.000, 7.930, 8.310, 36.120],
    [3.070, 6.440, 7.560, 37.360],
    [1.220, 7.510, 8.190, 36.290],
    [2.800, 10.310, 10.950, 33.500],
    [2.870, 5.070, 5.670, 38.800],
    [3.800, 8.010, 7.410, 38.160],
    [12.350, 4.520, 4.350, 48.270],
    [11.110, 3.480, 2.970, 47.140],
    [21.990, 22.020, 24.070, 39.860],
    [8.820, 3.300, 5.360, 43.310],
    [7.930, 0.000, 2.070, 43.750],
    [9.340, 2.250, 1.110, 45.430],
    [8.310, 2.070, 0.000, 44.430],
    [7.310, 2.440, 1.110, 43.430],
    [7.550, 0.750, 1.530, 43.520],
    [11.130, 18.410, 19.260, 25.400],
    [17.490, 23.440, 24.760, 23.210],
    [11.030, 18.930, 19.280, 25.430],
    [36.120, 43.750, 44.430, 0.000]
]

# labor intensity
P = [0.1609, 0.1164, 0.1026, 0.1516, 0.0939, 0.1320, 0.0687, 0.0930, 0.2116, 0.2529, 0.0868, 0.0828, 0.0975, 0.8177,
     0.4115, 0.3795, 0.0710, 0.0427, 0.1043, 0.0997, 0.1698, 0.2531]

# current assignment
A = [
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
]

# current locations of representatives
L = [4, 14, 16, 22]


# calculations
def get_results(limitf2: float | int) -> tuple[float, float, list[list[float]]]:
    regions_number = len(D)
    representatives_number = len(L)

    solver = LpProblem("Pfizer", LpMinimize)
    # get binary variables
    x = [[LpVariable(f'x_{repr}_{reg}', cat="Binary") for reg in range(
        regions_number)] for repr in range(representatives_number)]

    # Intensity should have sum [0.9, 1.1]
    for vars in x:
        total_intensity = lpSum(
            [var * intensity for var, intensity in zip(vars, P)])
        solver += total_intensity >= 0.9
        solver += total_intensity <= 1.1

    # each location should have one representative
    for region in range(regions_number):
        solver += lpSum([vars[region] for vars in x]) == 1

    # F2 constraint
    total_cost_f2 = 0
    A_arr = np.array(A).T
    for i, vars in enumerate(x):
        new_idx = np.where(A_arr[i] == 0)[0]
        total_cost_f2 += lpSum([vars[j] * P[j] for j in new_idx])
    solver += total_cost_f2 <= limitf2

    # F1 constraint
    total_cost_f1 = 0
    for i, vars in enumerate(x):
        total_cost_f1 += lpSum(var * distance for var,
                               distance in zip(vars, np.array(D).T[i]))
    # solver to minimize F1
    solver.setObjective(total_cost_f1)

    # solve the problem
    solver.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[solver.status] != "Optimal":
        return -1, limitf2, None

    return solver.objective.value(), total_cost_f2.value(), np.array([[var.value() for var in row] for row in x])


if __name__ == '__main__':
    print(f"Current f1 score: {(np.array(A) * np.array(D)).sum()}")
    print(f"Minimal f1 possible if everything allowed: {np.array(D).min(axis=1).sum()}")
    MAX_LABOR_CHANGE = sum(P)
    results = [get_results(i) for i in tqdm(np.linspace(0.2, MAX_LABOR_CHANGE, 1000))]
    points = np.array([(round(x, 3), round(y, 3)) for x, y, _ in results])
    solutions = [s for _, __, s in results]

    point_to_solution = {tuple(points[i]): solutions[i]
                         for i in range(len(solutions))}


    points = points[points[:, 0] >= 0].T
    pareto_points = set()
    pareto_points = {tuple(point) for point in points.T if not any(
        (points[0] <= point[0]) & (points[1] < point[1]))}

    pareto_points = sorted(pareto_points)

    print("Pareto points:", len(pareto_points))
    for point in pareto_points:
        print(f"\n\nSolution for point {point}:")
        print(point_to_solution[point])

    x, y = zip(*pareto_points)
    
    plt.scatter(x, y)
    plt.xlabel('f1 cost')
    plt.ylabel('f2 cost')
    # plt.show()
    plt.savefig('Project5_1.pdf')
