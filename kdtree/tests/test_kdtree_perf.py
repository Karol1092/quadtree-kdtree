import time
import pytest
import numpy as np
from kdtree import Kdtree

def brute_force(points, query_range):
    x1, x2, y1, y2 = query_range
    return [p for p in points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]

def test_performance_vs_brute_force():
    n = 500000
    m = 500
    
    points = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) for _ in range(n)]
    
    queries = []
    for _ in range(m):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(x1, 1)
        y1 = np.random.uniform(0, 1)
        y2 = np.random.uniform(y1, 1)
        
        queries.append([x1, x2, y1, y2])
    
    kdtree = Kdtree(points)
    
    start = time.perf_counter()
    for q in queries:
        kdtree.search(q)
    kd_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for q in queries:
        brute_force(points, q)
    brute_time = time.perf_counter() - start
    
    print(f"kd-tree time: {kd_time:.4f}s")
    print(f"brute-force time: {brute_time:.4f}s")

    assert kd_time < brute_time

def test_scaling_trend():
    sizes = [50000, 100000, 250000, 500000, 1000000]
    m = 10000
    times = []
    
    queries = []
    for _ in range(m):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(x1, 1)
        y1 = np.random.uniform(0, 1)
        y2 = np.random.uniform(y1, 1)
        
        queries.append([x1, x2, y1, y2])
    
    for n in sizes:
        points = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) for _ in range(n)]
        kdtree = Kdtree(points)
        
        start = time.perf_counter()
        for q in queries:
            kdtree.search(q)
        times.append(time.perf_counter() - start)
    
    for i, t in enumerate(times):
        print(f"{sizes[i]} points time: {t:.4f}s")
    
    for i in range(1, len(times)):
        assert times[i - 1] <= times[i]
        
    
    