from kdtree_test import Kdtree

def test_empty_build():
    kdtree = Kdtree([])
    assert kdtree.root is None
    
def test_single_point_build():
    kdtree = Kdtree([(5, 5)])
    root = kdtree.root
    
    assert kdtree.points == [(5, 5)]    
    assert root is not None
    assert root.point == (5, 5)
    assert root.left is None
    assert root.right is None
    assert root.axis == 0
    
def test_two_points_build():
    points = [(3, 1), (5, 5)]
    kdtree = Kdtree(points)
    root = kdtree.root
    left = root.left
    right = root.right
    subtree_points = root.get_subtree_points()
    
    assert root is not None
    
    assert left is not None and right is None
    
    assert left.left is None and left.right is None
    
    
    assert root.axis == 0
    assert left.axis == 1
    
    assert set(subtree_points) == set(kdtree.points) == set(points)
    
def test_contains_all_points():
    points = [(5, 5), (3, 7), (6, 3), (8, 8), (9, 9), (2, 3), (10, 1)]
    kdtree = Kdtree(points)
    root = kdtree.root
    subtree_points = root.get_subtree_points()
    
    assert set(kdtree.points) == set(points) == set(subtree_points)
    
def test_collinear_x_points():
    points = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    kdtree = Kdtree(points)
    root = kdtree.root
    
    subtree_points = root.get_subtree_points()
    
    assert set(subtree_points) == set(points)
    
def test_collinear_y_points():
    points = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    kdtree = Kdtree(points)
    root = kdtree.root
    
    subtree_points = root.get_subtree_points()
    
    assert set(subtree_points) == set(points)
    
def test_axis():
    points = [(5, 5), (3, 7), (6, 3), (8, 8), (9, 9), (2, 3), (10, 1)]
    kdtree = Kdtree(points)
    root = kdtree.root
    
    assert root.axis == 0
    
    if root.left:
        assert root.left.axis == 1
        
    if root.right:
        assert root.right.axis == 1
    
    if root.left.left:
        assert root.left.left.axis == 0
        
    if root.left.right:
        assert root.left.right.axis == 0
        
    if root.right.left:
        assert root.right.left.axis == 0
        
    if root.right.right:
        assert root.right.right.axis == 0
    
    