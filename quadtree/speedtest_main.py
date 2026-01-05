from main import Quadtree, Boundary, Point
import numpy as np
import pandas as pd

def brute_force(points, query_boundary):
    result = []
    for p in points:
        if (query_boundary.left_down.x <= p.x <= query_boundary.right_up.x and
            query_boundary.left_down.y <= p.y <= query_boundary.right_up.y):
            result.append(p)
    return result

def generate_random_points(boundary: Boundary, n: int) -> list[Point]:
    points = []
    for _ in range(n):
        x = np.random.uniform(boundary.left_down.x, boundary.right_up.x)
        y = np.random.uniform(boundary.left_down.y, boundary.right_up.y)
        points.append(Point(x, y))
    return points

def generate_collinear_points(boundary: Boundary, n: int) -> list[Point]:
    x1, y1 = boundary.left_down.x, boundary.left_down.y
    x2, y2 = boundary.right_up.x, boundary.right_up.y

    t_values = np.random.uniform(0, 1, n)
    points = [Point(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in t_values]
    return points

def generate_gesture_points(boundary: Boundary, n: int) -> list[Point]:
    width = boundary.right_up.x - boundary.left_down.x
    height = boundary.right_up.y - boundary.left_down.y

    # mniejszy = gęściej
    margin = 0.06 * min(width, height)

    centers = [
        Point(
            np.random.uniform(boundary.left_down.x, boundary.right_up.x),
            np.random.uniform(boundary.left_down.y, boundary.right_up.y),
        )
        for _ in range(3)
    ]

    base = n // 3
    counts = [base, base, n - 2 * base]

    points: list[Point] = []
    for c, k in zip(centers, counts):
        for _ in range(k):
            x = c.x + np.random.uniform(-margin, margin)
            y = c.y + np.random.uniform(-margin, margin)

            x = float(np.clip(x, boundary.left_down.x, boundary.right_up.x))
            y = float(np.clip(y, boundary.left_down.y, boundary.right_up.y))

            points.append(Point(x, y))

    return points

def generate_rectangle_points(boundary: Boundary, n: int) -> list[Point]:
    a = (boundary.left_down.x, boundary.left_down.y)
    b = (boundary.right_up.x, boundary.left_down.y)
    c = (boundary.right_up.x, boundary.right_up.y)
    d = (boundary.left_down.x, boundary.right_up.y)
    points = []
    for _ in range(n):
        side = np.random.randint(0, 4)
        t = np.random.uniform(0, 1)
        if side == 0:
            x = a[0] + t * (b[0] - a[0])
            y = a[1]
        elif side == 1:
            x = b[0]
            y = b[1] + t * (c[1] - b[1])
        elif side == 2:
            x = c[0] + t * (d[0] - c[0])
            y = c[1]
        else:
            x = d[0]
            y = d[1] + t * (a[1] - d[1])
        points.append(Point(x, y))

    return points

def generate_one_edge_points(boundary: Boundary, n: int) -> list[Point]:
    a = (boundary.left_down.x, boundary.left_down.y)
    b = (boundary.right_up.x, boundary.left_down.y)
    points = []
    for _ in range(n):
        t = np.random.uniform(0, 1)
        x = a[0] + t * (b[0] - a[0])
        y = a[1]
        points.append(Point(x, y))
    return points

def run_speed_test():
    num_points = [1000, 5000, 10000, 50000, 100000]

    boundary = Boundary(Point(0, 0), Point(10000, 0), Point(10000, 10000), Point(0, 10000))
    query_boundary = Boundary(Point(2500, 2500), Point(7500, 2500), Point(7500, 7500), Point(2500, 7500))

    funcs = {
        "Random": generate_random_points,
        "Collinear": generate_collinear_points,
        "Gesture-like": generate_gesture_points,
        "Rectangle edges": generate_rectangle_points,
        "One edge": generate_one_edge_points,
    }

    df = pd.DataFrame(columns=["Algorithm", "Data Size", "Points functions", "Time", "Points"])

    x = 0
    max_x = 25 * 2

    for func_name, func in funcs.items():
        for n in num_points:
            points = func(boundary, n)
            quadtree = Quadtree(boundary, points)

            import time
            start_time = time.time()
            quadtree_results = quadtree.search(query_boundary)
            elapsed_time = time.time() - start_time

            start_time = time.time()
            brute_force_results = brute_force(points, query_boundary)
            brute_force_time = time.time() - start_time

            assert set(quadtree_results) == set(brute_force_results)

            df.loc[x] = {
                "Algorithm": "Quadtree",
                "Data Size": n,
                "Points functions": func_name,
                "Time": elapsed_time,
                "Points": len(quadtree_results),
            }

            print(f"[{x + 1}/{max_x}] Completed with quadtree: {func_name} with {n} points.")
            x += 1

            df.loc[x] = {
                "Algorithm": "Brute-force",
                "Data Size": n,
                "Points functions": func_name,
                "Time": brute_force_time,
                "Points": len(brute_force_results),
            }

            print(f"[{x + 1}/{max_x}] Completed with brute-force: {func_name} with {n} points.")
            x += 1

    df.to_csv("results/speed_test.csv", index=False)

def run_correctness_test():
    boundary = Boundary(Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100))
    query_boundary = Boundary(Point(25, 25), Point(75, 25), Point(75, 75), Point(25, 75))

    points = generate_random_points(boundary, 1000)
    quadtree = Quadtree(boundary, points)

    quadtree_results = quadtree.search(query_boundary)
    brute_force_results = brute_force(points, query_boundary)

    quadtree_set = set((p.x, p.y) for p in quadtree_results)
    brute_force_set = set((p.x, p.y) for p in brute_force_results)

    assert quadtree_set == brute_force_set, "Quadtree results do not match brute-force results"
    print("Correctness test passed!")

if __name__ == "__main__":
    run_correctness_test()
    run_speed_test()