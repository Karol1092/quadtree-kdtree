from kdtree_test import Kdtree

def test_empty_tree_query():
    points = []
    kdtree = Kdtree(points)
    result = kdtree.search([0, 10, 0, 10])
    assert result == []
    
def test_single_point_inside():
    points = [(5, 5)]
    kdtree = Kdtree(points)
    result = kdtree.search([3, 7, 3, 7])
    assert result == [(5, 5)]
    
def test_single_point_outside():
    points = [(8, 8)]
    kdtree = Kdtree(points)
    result = kdtree.search([2, 5, 2, 5])
    assert result == []

def test_query_all_points():
    points = [(5, 5), (3, 7), (6, 3), (8, 8), (9, 9), (2, 3), (10, 1)]
    kdtree = Kdtree(points)
    result = kdtree.search([0, 10, 0, 10])
    assert set(result) == set(points)
    
def test_query_some_points():
    points = [(5, 5), (3, 7), (6, 3), (8, 8), (9, 9), (2, 3), (10, 1)]
    kdtree = Kdtree(points)
    result = kdtree.search([4, 9, 2, 6])
    assert set(result) == set([(5, 5), (6, 3)])
    
def test_query_no_points():
    points = [(5, 5), (3, 7), (6, 3), (8, 8), (9, 9), (2, 3), (10, 1)]
    kdtree = Kdtree(points)
    result = kdtree.search([7, 6, 8, 7])
    assert result == []