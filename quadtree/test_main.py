import pytest
from quadtree import Point, Boundary, Quadtree

@pytest.fixture()
def eps() -> float:
    # Używamy takiego samego defaultu jak w Quadtree.
    return 1e-9

def pt(x: float, y: float) -> Point:
    return Point(x, y)

def as_xy(p: Point) -> tuple[float, float]:
    return p.x, p.y

def list_xy(points: list[Point]) -> list[tuple[float, float]]:
    return [as_xy(p) for p in points]

@pytest.fixture()
def base_boundary() -> Boundary:
    # Kwadrat 0..10 w obu osiach.
    return Boundary.from_values(0, 10, 0, 10)

@pytest.fixture()
def base_tree(base_boundary: Boundary, eps: float) -> Quadtree:
    return Quadtree(boundary=base_boundary, capacity=1, eps=eps)

def test_point_repr_format():
    assert repr(pt(1, 2)) == "(1, 2)"

def test_boundary_invalid_raises():
    # ld.x >= rd.x
    with pytest.raises(ValueError):
        Boundary(pt(0, 0), pt(0, 0), pt(0, 1), pt(0, 1))

    # ld.y >= lu.y
    with pytest.raises(ValueError):
        Boundary(pt(0, 0), pt(1, 0), pt(1, 0), pt(0, 0))

def test_boundary_from_values_creates_rectangle(eps: float):
    b = Boundary.from_values(0, 2, 0, 3)
    assert as_xy(b.left_down) == (0, 0)
    assert as_xy(b.right_down) == (2, 0)
    assert as_xy(b.right_up) == (2, 3)
    assert as_xy(b.left_up) == (0, 3)
    assert b.contains_point(pt(1, 1), eps)

def test_boundary_contains_point_interior_true(base_boundary: Boundary, eps: float):
    assert base_boundary.contains_point(pt(5, 5), eps)

def test_boundary_contains_point_on_edges_true(base_boundary: Boundary, eps: float):
    # Rogi
    assert base_boundary.contains_point(pt(0, 0), eps)
    assert base_boundary.contains_point(pt(10, 0), eps)
    assert base_boundary.contains_point(pt(10, 10), eps)
    assert base_boundary.contains_point(pt(0, 10), eps)

    # Krawędzie
    assert base_boundary.contains_point(pt(0, 5), eps)
    assert base_boundary.contains_point(pt(10, 5), eps)
    assert base_boundary.contains_point(pt(5, 0), eps)
    assert base_boundary.contains_point(pt(5, 10), eps)

def test_boundary_intersects_touching_edge_true(eps: float):
    # Dotyk jedną krawędzią: a.right_up.x == b.left_down.x
    a = Boundary.from_values(0, 2, 0, 2)
    b = Boundary.from_values(2, 4, 0, 2)
    assert a.intersects(b, eps) is True

def test_quadtree_insert_outside_false(base_tree: Quadtree):
    assert base_tree.insert(pt(-1, -1)) is False

def test_quadtree_insert_list_all_true(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=2, eps=eps)
    assert qt.insert([pt(1, 1), pt(2, 2)]) is True

def test_quadtree_insert_list_with_outside_makes_false(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=3, eps=eps)
    assert qt.insert([pt(1, 1), pt(99, 99), pt(2, 2)]) is False

def test_quadtree_subdivide_on_capacity(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)

    assert qt.insert(pt(1, 1)) is True
    assert qt.left_down is None  # jeszcze nie było podziału

    assert qt.insert(pt(9, 9)) is True
    assert qt.left_down is not None
    assert qt.right_down is not None
    assert qt.right_up is not None
    assert qt.left_up is not None

    assert Point(1, 1) in qt.all_points
    assert Point(9, 9) in qt.all_points

def test_search_filters_points_without_subdivide(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=10, eps=eps)
    qt.insert([pt(2, 2), pt(9, 9)])

    query = Boundary.from_values(0, 5, 0, 5)
    result = set(list_xy(qt.search(query)))
    assert (2, 2) in result
    assert (9, 9) not in result

def test_leaf_never_exceeds_capacity_after_many_inserts(base_boundary: Boundary, eps: float):
    # Wstawiamy dużo punktów; korzeń powinien się podzielić, a potem trzymać 0 punktów,
    # a liście nie powinny przekraczać capacity.
    capacity = 2
    qt = Quadtree(base_boundary, capacity=capacity, eps=eps)

    pts = [
        pt(1, 1), pt(2, 2), pt(3, 3),
        pt(7, 1), pt(8, 2), pt(9, 3),
        pt(7, 7), pt(8, 8), pt(9, 9),
        pt(1, 7), pt(2, 8), pt(3, 9),
    ]
    assert qt.insert(pts) is True

    # Po subdivide root przechowuje punkty w dzieciach.
    assert len(qt.points) <= capacity

    # Rekurencyjnie sprawdzamy każde dziecko-liść.
    def check(node: Quadtree):
        if node.left_down is None:
            assert len(node.points) <= capacity
            return
        check(node.left_down)
        check(node.right_down)
        check(node.right_up)
        check(node.left_up)

    check(qt)

def test_subdivide_assigns_points_to_correct_quadrants(base_boundary: Boundary, eps: float):
    # Wymuszamy subdivide: capacity=1, drugi insert dzieli.
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    assert qt.insert(pt(1, 1)) is True
    assert qt.insert(pt(9, 9)) is True  # subdivide

    # Wstawiamy po jednym punkcie do każdej ćwiartki.
    p_ld = pt(1, 2)  # left-down
    p_rd = pt(8, 2)  # right-down
    p_ru = pt(8, 8)  # right-up
    p_lu = pt(2, 8)  # left-up

    assert qt.insert([p_ld, p_rd, p_ru, p_lu]) is True

    assert qt.left_down is not None
    ld = set(list_xy(qt.left_down.get_all_points()))
    rd = set(list_xy(qt.right_down.get_all_points()))
    ru = set(list_xy(qt.right_up.get_all_points()))
    lu = set(list_xy(qt.left_up.get_all_points()))

    assert (1, 2) in ld
    assert (8, 2) in rd
    assert (8, 8) in ru
    assert (2, 8) in lu

    # Dodatkowo upewniamy się, że te punkty nie "wyciekły" do innych kwadratów.
    assert (1, 2) not in (rd | ru | lu)
    assert (8, 2) not in (ld | ru | lu)
    assert (8, 8) not in (ld | rd | lu)
    assert (2, 8) not in (ld | rd | ru)


def test_contains_point_eps_threshold():
    b = Boundary.from_values(0, 1, 0, 1)

    eps_small = 1e-9
    eps_big = 1e-6

    # Punkt minimalnie poza prawą krawędzią
    p = pt(1 + 5e-7, 0.5)

    # Dla małego eps powinien być poza
    assert b.contains_point(p, eps_small) is False
    # Dla większego eps powinien wejść
    assert b.contains_point(p, eps_big) is True

def test_intersects_eps_gap():
    a = Boundary.from_values(0, 1, 0, 1)

    # b jest po prawej stronie z małą przerwą
    gap = 5e-7
    b = Boundary(
        pt(1 + gap, 0),
        pt(2 + gap, 0),
        pt(2 + gap, 1),
        pt(1 + gap, 1),
    )

    assert a.intersects(b, eps=1e-9) is False
    assert a.intersects(b, eps=1e-6) is True

def test_quadtree_insert_respects_eps_for_boundary():
    boundary = Boundary.from_values(0, 1, 0, 1)
    eps_big = 1e-6
    qt = Quadtree(boundary, capacity=5, eps=eps_big)

    # Minimalnie poza — powinno wejść
    assert qt.insert(pt(1 + 5e-7, 0.5)) is True

    # Bardziej poza — nie powinno wejść
    assert qt.insert(pt(1 + 5e-5, 0.5)) is False

def test_search_respects_eps_via_contains_point():
    boundary = Boundary.from_values(0, 1, 0, 1)
    eps_big = 1e-6
    qt = Quadtree(boundary, capacity=5, eps=eps_big)

    # Punkt minimalnie poza boundary query, ale w eps
    p = pt(1 + 5e-7, 0.5)
    assert qt.insert(p) is True

    query = Boundary.from_values(0, 1, 0, 1)
    res = set(list_xy(qt.search(query)))
    assert (p.x, p.y) in res

def test_search_respects_eps_via_intersects():
    boundary = Boundary.from_values(0, 10, 0, 10)
    eps_big = 1e-6
    qt = Quadtree(boundary, capacity=5, eps=eps_big)

    p = pt(5, 5)
    assert qt.insert(p) is True

    # Query minimalnie poza z eps
    gap = 5e-7
    query = Boundary(
        pt(0 - gap, 0 - gap),
        pt(10 + gap, 0 - gap),
        pt(10 + gap, 10 + gap),
        pt(0 - gap, 10 + gap),
    )

    res = set(list_xy(qt.search(query)))
    assert (p.x, p.y) in res

def test_from_points_empty_list_raises():
    with pytest.raises(ValueError):
        Quadtree.from_points([])

def test_from_float_empty_list_raises():
    with pytest.raises(ValueError):
        Quadtree.from_float([])

