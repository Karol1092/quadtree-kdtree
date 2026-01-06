from visualizer.main import Visualizer

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

    def __le__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x <= other.x and self.y <= other.y

    def __ge__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x >= other.x and self.y >= other.y

    def __hash__(self):
        return hash((self.x, self.y))

class Boundary:
    def __init__(self, ld: Point, rd: Point, ru: Point, lu: Point):
        invalid = (
            not (ld.x < rd.x) or
            not (ld.y < lu.y) or
            rd.y != ld.y or
            ru.x != rd.x or
            lu.x != ld.x or
            ru.y != lu.y or
            lu.y <= ld.y
        )
        if invalid:
            raise ValueError("Invalid boundary points")

        self.left_down = ld
        self.right_down = rd
        self.right_up = ru
        self.left_up = lu

    @classmethod
    def from_values(cls, x1: float, x2: float, y1: float, y2: float):
        ld = Point(x1, y1)
        rd = Point(x2, y1)
        ru = Point(x2, y2)
        lu = Point(x1, y2)
        return cls(ld, rd, ru, lu)

    def contains_point(self, p: Point, eps: float) -> bool:
        return (
            (self.left_down.x - eps) <= p.x <= (self.right_up.x + eps)
            and (self.left_down.y - eps) <= p.y <= (self.right_up.y + eps)
        )

    def to_value_list(self) -> list[tuple[float, float]]:
        return [(self.left_down.x, self.left_down.y),
                (self.right_down.x, self.right_down.y),
                (self.right_up.x, self.right_up.y),
                (self.left_up.x, self.left_up.y)]

    def intersects(self, other, eps: float) -> bool:
        if not isinstance(other, Boundary):
            return False
        return not (
            (self.right_up.x + eps) < other.left_down.x
            or (other.right_up.x + eps) < self.left_down.x
            or (self.right_up.y + eps) < other.left_down.y
            or (other.right_up.y + eps) < self.left_down.y
        )

    def visualize(self, vis=None, color='green') -> Visualizer:
        if not vis:
            vis = Visualizer()

        vis.add_polygon(self.to_value_list(), facecolor=None ,edgecolor=color, closed=True, linewidth=2, alpha=0.2)

        return vis

    def __repr__(self):
        return f"Boundary({self.left_down}, {self.right_down}, {self.right_up}, {self.left_up})"

class Quadtree:
    def __init__(self, boundary: Boundary, points: list[Point] | None = None, capacity: int = 1, eps: float = 1e-9):
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")

        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.all_points = set()
        self.eps = eps

        self.left_down = None
        self.right_down = None
        self.right_up = None
        self.left_up = None

        if points:
            self.insert(points)

    @classmethod
    def from_points(cls, points: list[Point], capacity: int = 1, eps: float = 1e-9):
        boundary = Quadtree.get_max_boundary(points)
        return Quadtree(boundary, points, capacity, eps)

    @classmethod
    def from_float(cls, fl: list[tuple[float, float]], capacity: int = 1, eps: float = 1e-9):
        points = [Point(x, y) for x, y in fl]
        boundary = Quadtree.get_max_boundary(points)
        return Quadtree(boundary, points, capacity, eps)

    @staticmethod
    def get_max_boundary(points: list[Point]) -> Boundary:
        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        return Boundary(
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(max_x, max_y),
            Point(min_x, max_y)
        )

    def subdivide(self):
        mid_x = (self.boundary.left_down.x + self.boundary.right_up.x) / 2.0
        mid_y = (self.boundary.left_down.y + self.boundary.right_up.y) / 2.0

        self.left_down = Quadtree(
            Boundary(
                self.boundary.left_down,
                Point(mid_x, self.boundary.left_down.y),
                Point(mid_x, mid_y),
                Point(self.boundary.left_down.x, mid_y)
            ),
            capacity=self.capacity,
            eps=self.eps
        )

        self.right_down = Quadtree(
            Boundary(
                Point(mid_x, self.boundary.left_down.y),
                self.boundary.right_down,
                Point(self.boundary.right_up.x, mid_y),
                Point(mid_x, mid_y)
            ),
            capacity=self.capacity,
            eps=self.eps
        )

        self.right_up = Quadtree(
            Boundary(
                Point(mid_x, mid_y),
                Point(self.boundary.right_up.x, mid_y),
                self.boundary.right_up,
                Point(mid_x, self.boundary.left_up.y)
            ),
            capacity=self.capacity,
            eps=self.eps
        )

        self.left_up =  Quadtree(
            Boundary(
                Point(self.boundary.left_down.x, mid_y),
                Point(mid_x, mid_y),
                Point(mid_x, self.boundary.left_up.y),
                self.boundary.left_up
            ),
            capacity=self.capacity,
            eps=self.eps
        )

    def insert(self, point: list[Point] | Point) -> bool:
        if isinstance(point, list):
            return all(self.insert(p) for p in point)

        if not self.boundary.contains_point(point, self.eps):
            return False

        if point in self.all_points:
            return False

        self.all_points.add(point)

        if self.left_down is None:
            if len(self.points) < self.capacity:
                self.points.append(point)
                return True

            self.subdivide()

            for p in self.points:
                self._insert_to_all(p)

            self.points = []

        return self._insert_to_all(point)

    def _insert_to_all(self, point: Point) -> bool:
        return (self.left_down.insert(point) or self.right_down.insert(point) or
                self.right_up.insert(point) or self.left_up.insert(point))

    def get_all_points(self) -> set[Point]:
        if len(self.all_points) > 0:
            return self.all_points

        result = self.points.copy()

        if self.left_down:
            result.extend(self.left_down.get_all_points())
        if self.right_down:
            result.extend(self.right_down.get_all_points())
        if self.right_up:
            result.extend(self.right_up.get_all_points())
        if self.left_up:
            result.extend(self.left_up.get_all_points())

        return set(result)

    def search(self, boundary: Boundary) -> list[Point]:
        result = []

        if not boundary.intersects(self.boundary, self.eps):
            return result

        for p in self.points:
            if boundary.contains_point(p, self.eps):
                result.append(p)

        if self.left_down:
            result.extend(self.left_down.search(boundary))
        if self.right_down:
            result.extend(self.right_down.search(boundary))
        if self.right_up:
            result.extend(self.right_up.search(boundary))
        if self.left_up:
            result.extend(self.left_up.search(boundary))

        return result

class QuadtreeVisualizer(Visualizer):
    def __init__(self, tree: Quadtree):
        super().__init__()
        self.quadtree = tree
        self._visualize_quadtree(self.quadtree)

    def _visualize_quadtree(self, tree: Quadtree):
        self.add_point([(p.x, p.y) for p in self.quadtree.all_points], color='blue')

        def _visualize_subtree(subtree: Quadtree):
            subtree.boundary.visualize(self, color='green')

            if subtree.left_down:
                _visualize_subtree(subtree.left_down)
            if subtree.right_down:
                _visualize_subtree(subtree.right_down)
            if subtree.right_up:
                _visualize_subtree(subtree.right_up)
            if subtree.left_up:
                _visualize_subtree(subtree.left_up)

        _visualize_subtree(tree)

    def visualize_search(self, boundary: Boundary, color='red') -> Visualizer:
        boundary.visualize(self, color=color)

        found_points = self.quadtree.search(boundary)
        self.add_point([(p.x, p.y) for p in found_points], color='yellow')

        return self

    def visualize_search_with_steps(self, boundary: Boundary, color='red') -> Visualizer:
        boundary.visualize(self, color=color)

        def _visualize_search_step(tree: Quadtree):
            if not boundary.intersects(tree.boundary, tree.eps):
                return

            self.add_polygon(tree.boundary.to_value_list(), edgecolor='orange', facecolor=None, closed=True, linewidth=4, alpha=0.2)

            for p in tree.points:
                if boundary.contains_point(p, tree.eps):
                    self.add_point((p.x, p.y), color='yellow')

            if tree.left_down:
                _visualize_search_step(tree.left_down)
            if tree.right_down:
                _visualize_search_step(tree.right_down)
            if tree.right_up:
                _visualize_search_step(tree.right_up)
            if tree.left_up:
                _visualize_search_step(tree.left_up)

        _visualize_search_step(self.quadtree)

        boundary.visualize(self, color=color)
        return self