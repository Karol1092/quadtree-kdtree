# wpisać w terminalu
# source ~/anaconda3/etc/profile.d/conda.sh
# użyć python3

import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parents[0]))

from visualizer.main import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Node:
    def __init__(self, point, axis = 0, left = None, right = None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        
    def get_subtree_points(self):
        points = [self.point]

        if self.left is not None:
            points += self.left.get_subtree_points()
        if self.right is not None:
            points += self.right.get_subtree_points()
            
        return points

    def traverse(self, vis: Visualizer, xmin, xmax, ymin, ymax) -> None:
        area = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        to_remove = vis.add_polygon(area, alpha = 0.3, color="grey")
        vis.remove_figure(to_remove)
        
        if not self:
            return

        x, y = self.point
        
        if self.axis == 0:
            vis.add_line_segment(((x, ymin), (x, ymax)), color="green")
            
            if self.left:
                self.left.traverse(vis, xmin, x, ymin, ymax)
            if self.right:
                self.right.traverse(vis, x, xmax, ymin, ymax)
            
        else:
            vis.add_line_segment(((xmin, y), (xmax, y)), color="green")
            
            if self.left:
                self.left.traverse(vis, xmin, xmax, ymin, y)
            if self.right: 
                self.right.traverse(vis, xmin, xmax, y, ymax)
    
class Kdtree:
    def build_kdtree(self, points, depth = 0):
        axis = depth % self.k
        n = len(points[axis])
        
        if n < 1:
            return None
        if n == 1:
            return Node(points[axis][0], axis = depth % self.k)
        
        median = points[axis][n // 2]
        
        left, right = [[] for _ in range(self.k)], [[] for _ in range(self.k)]
        
        for i in range(self.k):
            for p in points[i]:
                if p == median:
                    continue
                if p[axis] < median[axis]:
                    left[i].append(p)
                else:
                    right[i].append(p)
        
        return Node(
            point = median,
            axis = axis,
            left = self.build_kdtree(left, depth + 1),
            right = self.build_kdtree(right, depth + 1)
        )
        
    def __init__(self, points, k = 2):
        self.points = points
        self.k = k
        
        P = [sorted(points, key = lambda x: x[i]) for i in range(k)]
        
        self.root = self.build_kdtree(P)
    
    def visualize(self, default_region = [0, 10, 0, 10]) -> Visualizer:
        vis = Visualizer()
        
        vis.add_point(self.points, color="blue")
        
        root = self.root
        
        xmin, xmax, ymin, ymax = default_region
        
        root.traverse(vis, xmin, xmax, ymin, ymax)
        
        return vis
    
    def contains(self, node_region, query_region):
        x1, x2, y1, y2 = node_region
        xmin, xmax, ymin, ymax = query_region
        return xmin <= x1 and x2 <= xmax and ymin <= y1 and y2 <= ymax
        
    def intersects(self, node_region, query_region):
        x1, x2, y1, y2 = node_region
        xmin, xmax, ymin, ymax = query_region
        return not (xmin > x2 or x1 > xmax or ymin > y2 or y1 > ymax)
       
    def _search(self, node: Node, node_region, query_region, result):
        if not node:
            return
        
        x1, x2, y1, y2 = node_region
        xmin, xmax, ymin, ymax = query_region
        x, y = node.point
        
        if xmin <= x <= xmax and ymin <= y <= ymax:
            result.append(node.point)
            
        if node.axis == 0:
            left_region = [x1, x, y1, y2]
            right_region = [x, x2, y1, y2]
        else:
            left_region = [x1, x2, y1, y]
            right_region = [x1, x2, y, y2]
                        
        if node.left:  
            if self.contains(left_region, query_region):
                result.extend(node.left.get_subtree_points())
            elif self.intersects(left_region, query_region):
                self._search(node.left, left_region, query_region, result)
        
        if node.right:
            if self.contains(right_region, query_region):
                result.extend(node.right.get_subtree_points())
            elif self.intersects(right_region, query_region):
                self._search(node.right, right_region, query_region, result)
    
    def search(self, query_region, default_region = [0, 10, 0, 10]):
        result = []
        self._search(self.root, default_region, query_region, result)
        return result
               
    def _search_vis(self, node: Node, node_region, query_region, result, vis: Visualizer):
        if not node:
            return
        
        x1, x2, y1, y2 = node_region
        xmin, xmax, ymin, ymax = query_region
        x, y = node.point
        

        to_remove = []
        
        area = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        to_remove.append(vis.add_polygon(area, alpha = 0.3, color="grey"))
        
        to_remove.append(vis.add_point(node.point, color="orange"))
        
        if node.axis == 0:
            to_remove.append(vis.add_line_segment(((x, y1), (x, y2)), color="yellow"))
        else:
            to_remove.append(vis.add_line_segment(((x1, y), (x2, y)), color="yellow"))
        
        for fig in to_remove:
            vis.remove_figure(fig)
        

        if xmin <= x <= xmax and ymin <= y <= ymax:
            result.append(node.point)
            vis.add_point(node.point, color="red")
            
        if node.axis == 0:
            left_region = [x1, x, y1, y2]
            right_region = [x, x2, y1, y2]
        else:
            left_region = [x1, x2, y1, y]
            right_region = [x1, x2, y, y2]
                        
        if node.left:  
            if self.contains(left_region, query_region):
                pts = node.left.get_subtree_points()
                result.extend(pts)
                for p in pts:
                    vis.add_point(p, color="red")
            elif self.intersects(left_region, query_region):
                self._search_vis(node.left, left_region, query_region, result, vis)
        
        if node.right:
            if self.contains(right_region, query_region):
                pts = node.right.get_subtree_points()
                result.extend(pts)
                for p in pts:
                    vis.add_point(p, color="red")
            elif self.intersects(right_region, query_region):
                self._search_vis(node.right, right_region, query_region, result, vis)
                
    def search_vis(self, query_region, node_region = [0, 10, 0, 10]):
        result = []
        vis = self.visualize(node_region)
        xmin, xmax, ymin, ymax = query_region
        vis.add_polygon(((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)), alpha = 0.5, color="grey")
        self._search_vis(self.root, node_region, query_region, result, vis)
        return vis