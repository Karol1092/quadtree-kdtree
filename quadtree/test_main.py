import pytest
from main import Point, Boundary, Quadtree

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
    return Boundary.from_int(0, 10, 0, 10)

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

def test_boundary_from_int_creates_rectangle(eps: float):
    b = Boundary.from_int(0, 2, 0, 3)
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

def test_boundary_intersects_disjoint_false(eps: float):
    a = Boundary.from_int(0, 2, 0, 2)
    b = Boundary.from_int(3, 5, 3, 5)
    assert a.intersects(b, eps) is False

def test_boundary_intersects_touching_edge_true(eps: float):
    # Dotyk jedną krawędzią: a.right_up.x == b.left_down.x
    a = Boundary.from_int(0, 2, 0, 2)
    b = Boundary.from_int(2, 4, 0, 2)
    assert a.intersects(b, eps) is True

def test_boundary_intersects_containment_true(eps: float):
    a = Boundary.from_int(0, 10, 0, 10)
    b = Boundary.from_int(2, 3, 2, 3)
    assert a.intersects(b, eps) is True
    assert b.intersects(a, eps) is True

def test_quadtree_insert_outside_false(base_tree: Quadtree):
    assert base_tree.insert(pt(-1, -1)) is False

def test_quadtree_insert_inside_true(base_tree: Quadtree):
    p = pt(1, 1)
    assert base_tree.insert(p) is True

    pts = base_tree.get_all_points()
    assert (1, 1) in set(list_xy(pts))

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

    pts = set(list_xy(qt.get_all_points()))
    assert (1, 1) in pts
    assert (9, 9) in pts

def test_quadtree_insert_point_on_outer_boundary_included(base_tree: Quadtree):
    # contains_point jest inkluzywne (z eps)
    assert base_tree.insert(pt(0, 0)) is True
    assert (0, 0) in set(list_xy(base_tree.get_all_points()))

def test_quadtree_insert_point_on_midlines_occurs_once(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    assert qt.insert(pt(1, 1)) is True
    assert qt.insert(pt(9, 9)) is True  # subdivide

    # Punkt na mid_x=5
    on_mid = pt(5, 7)
    assert qt.insert(on_mid) is True

    pts = list_xy(qt.get_all_points())
    assert pts.count((5, 7)) == 1

def test_quadtree_get_all_points_empty(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    assert qt.get_all_points() == []

def test_quadtree_get_all_points_duplicates_allowed(base_boundary: Boundary, eps: float):
    # Aktualna implementacja przechowuje punkt w liście, bez deduplikacji.
    qt = Quadtree(base_boundary, capacity=10, eps=eps)
    assert qt.insert(pt(1, 1)) is True
    assert qt.insert(pt(1, 1)) is True

    pts = list_xy(qt.get_all_points())
    assert pts.count((1, 1)) == 2

def test_search_no_intersection_empty(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=10, eps=eps)
    qt.insert([pt(1, 1), pt(2, 2)])

    outside = Boundary.from_int(100, 110, 100, 110)
    assert qt.search(outside) == []

def test_search_filters_points_without_subdivide(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=10, eps=eps)
    qt.insert([pt(2, 2), pt(9, 9)])

    query = Boundary.from_int(0, 5, 0, 5)
    result = set(list_xy(qt.search(query)))
    assert (2, 2) in result
    assert (9, 9) not in result

def test_search_inclusive_edges(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=10, eps=eps)
    qt.insert(pt(5, 5))

    query = Boundary.from_int(5, 8, 5, 8)
    result = set(list_xy(qt.search(query)))
    assert (5, 5) in result

def test_boundary_contains_point_with_eps_tolerance(eps: float):
    b = Boundary.from_int(0, 1, 0, 1)
    assert b.contains_point(pt(0 + eps / 2, 0 + eps / 2), eps)
    assert b.contains_point(pt(1 + eps / 2, 1 + eps / 2), eps)
    assert b.contains_point(pt(-eps / 2, -eps / 2), eps)

def test_boundary_intersects_cross_shape_without_vertices_inside(eps: float):
    # Pionowy prostokąt przez środek
    vertical = Boundary.from_int(4, 6, 0, 10)
    # Poziomy prostokąt przez środek
    horizontal = Boundary.from_int(0, 10, 4, 6)

    assert vertical.intersects(horizontal, eps) is True
    assert horizontal.intersects(vertical, eps) is True

def test_quadtree_init_with_points_inserts_them(base_boundary: Boundary, eps: float):
    p1, p2 = pt(1, 1), pt(2, 2)
    qt = Quadtree(boundary=base_boundary, points=[p1, p2], capacity=10, eps=eps)
    assert set(list_xy(qt.get_all_points())) == {(1, 1), (2, 2)}

def test_quadtree_from_points_builds_boundary_and_keeps_all_points(eps: float):
    points = [pt(0, 0), pt(10, 10), pt(-5, 7)]
    qt = Quadtree.from_points(points, capacity=10, eps=eps)

    all_pts = set(list_xy(qt.get_all_points()))
    assert all_pts == {(0, 0), (10, 10), (-5, 7)}

    for p in points:
        assert qt.boundary.contains_point(p, eps)

def test_quadtree_from_float_builds_and_inserts(eps: float):
    qt = Quadtree.from_float([(1.5, 2.5), (-3.0, 4.0)], capacity=10, eps=eps)
    assert set(list_xy(qt.get_all_points())) == {(1.5, 2.5), (-3.0, 4.0)}

def test_quadtree_insert_points_on_midlines_and_center_no_duplicates(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    assert qt.insert(pt(1, 1)) is True
    assert qt.insert(pt(9, 9)) is True  # subdivide

    mid_x = (base_boundary.left_down.x + base_boundary.right_up.x) / 2.0
    mid_y = (base_boundary.left_down.y + base_boundary.right_up.y) / 2.0

    candidates = [pt(mid_x, 7), pt(7, mid_y), pt(mid_x, mid_y)]
    for c in candidates:
        assert qt.insert(c) is True

    pts = list_xy(qt.get_all_points())
    assert pts.count((mid_x, 7)) == 1
    assert pts.count((7, mid_y)) == 1
    assert pts.count((mid_x, mid_y)) == 1

def test_search_after_subdivide_returns_points(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    qt.insert([pt(1, 1), pt(9, 9)])  # subdivide
    qt.insert(pt(5, 5))

    query = Boundary.from_int(4, 6, 4, 6)
    res = set(list_xy(qt.search(query)))
    assert (5, 5) in res
    assert (1, 1) not in res
    assert (9, 9) not in res

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

def test_boundary_intersects_touching_corner_true(eps: float):
    # Dotyk w jednym punkcie (narożniku)
    a = Boundary.from_int(0, 2, 0, 2)
    b = Boundary.from_int(2, 4, 2, 4)
    assert a.intersects(b, eps) is True
    assert b.intersects(a, eps) is True

def test_boundary_intersects_overlapping_stripe_true(eps: float):
    # Nakładanie pasem (częściowe pokrycie)
    a = Boundary.from_int(0, 10, 0, 2)
    b = Boundary.from_int(5, 15, 1, 3)
    assert a.intersects(b, eps) is True
    assert b.intersects(a, eps) is True

def test_boundary_intersects_identical_true(eps: float):
    a = Boundary.from_int(0, 10, 0, 10)
    b = Boundary.from_int(0, 10, 0, 10)
    assert a.intersects(b, eps) is True
    assert b.intersects(a, eps) is True

def test_contains_point_eps_threshold():
    # Granica 0..1
    b = Boundary.from_int(0, 1, 0, 1)

    eps_small = 1e-9
    eps_big = 1e-6

    # Punkt minimalnie poza prawą krawędzią
    p = pt(1 + 5e-7, 0.5)

    # Dla małego eps powinien być poza
    assert b.contains_point(p, eps_small) is False
    # Dla większego eps powinien wejść
    assert b.contains_point(p, eps_big) is True

def test_intersects_eps_gap():
    a = Boundary.from_int(0, 1, 0, 1)

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
    boundary = Boundary.from_int(0, 1, 0, 1)
    eps_big = 1e-6
    qt = Quadtree(boundary, capacity=5, eps=eps_big)

    # Minimalnie poza — powinno wejść
    assert qt.insert(pt(1 + 5e-7, 0.5)) is True

    # Bardziej poza — nie powinno wejść
    assert qt.insert(pt(1 + 5e-5, 0.5)) is False

def test_search_respects_eps_via_contains_point():
    boundary = Boundary.from_int(0, 1, 0, 1)
    eps_big = 1e-6
    qt = Quadtree(boundary, capacity=5, eps=eps_big)

    # Punkt minimalnie poza boundary query, ale w eps
    p = pt(1 + 5e-7, 0.5)
    assert qt.insert(p) is True

    query = Boundary.from_int(0, 1, 0, 1)
    res = set(list_xy(qt.search(query)))
    assert (p.x, p.y) in res

def test_search_respects_eps_via_intersects():
    boundary = Boundary.from_int(0, 10, 0, 10)
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

def test_get_max_boundary_contains_extremes_exactly():
    # Bug-hunt: boundary wyliczony z punktów MUSI zawierać punkty skrajne.
    # Jeśli ktoś doda przesunięcie (+eps) tylko w jedną stronę albo pomyli min/max, to padnie.
    pts = [pt(-5, 7), pt(10, -3), pt(0, 0)]
    b = Quadtree.get_max_boundary(pts)

    assert (b.left_down.x, b.left_down.y) == (-5, -3)
    assert (b.right_up.x, b.right_up.y) == (10, 7)

    # Skrajne punkty muszą być w środku boundary nawet dla eps=0.
    for p in pts:
        assert b.contains_point(p, eps=0.0) is True


def test_subdivide_moves_parent_points_to_children(base_boundary: Boundary, eps: float):
    # Bug-hunt: po subdivide rodzic powinien wyczyścić self.points.
    qt = Quadtree(base_boundary, capacity=1, eps=eps)
    assert qt.insert(pt(1, 1)) is True
    assert qt.points  # ma 1

    assert qt.insert(pt(9, 9)) is True  # wymusza subdivide

    assert qt.left_down is not None
    assert qt.points == []

    # Oba punkty nadal osiągalne
    all_pts = set(list_xy(qt.get_all_points()))
    assert (1, 1) in all_pts
    assert (9, 9) in all_pts


def test_search_whole_boundary_returns_all_points_even_after_many_subdivides(base_boundary: Boundary, eps: float):
    qt = Quadtree(base_boundary, capacity=1, eps=eps)

    pts = [
        pt(1, 1), pt(2, 2), pt(3, 3),
        pt(9, 1), pt(8, 2), pt(7, 3),
        pt(9, 9), pt(8, 8), pt(7, 7),
        pt(1, 9), pt(2, 8), pt(3, 7),
    ]
    assert qt.insert(pts) is True

    query = base_boundary
    found = set(list_xy(qt.search(query)))
    expected = set(list_xy(pts))
    assert found == expected

