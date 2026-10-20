"""
AETHER-Q light-path selection: maximize the EDR of the HAP-HAP segment subject to the fidelity
requirement, which is equivalent to a length limit L(p) <= L_max.

  1. Dijkstra on -ln(eta): p_c, the path of maximum efficiency.
  2. Dijkstra on length: p_d, the shortest path. If p_d exceeds L_max, no path can meet F_min.
  3. If p_c exceeds L_max, LARAC (Lagrangian relaxation) runs Dijkstra on -ln(eta) + lambda * length
     and moves between p_c and p_d; every iterate within L_max is kept.
  4. Of all paths within L_max met on the way (p_c if feasible, p_d, the feasible LARAC iterates),
     the one with the largest EDR is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

import physics as P


@dataclass
class Route:
    nodes: list[int]                   # source HAP, satellites ..., destination HAP
    cost: float                        # sum of -ln(edge efficiency)
    length_km: float
    status: str                        # "served", "infeasible", "no path" (references: "within limit" /
                                       # "too long")
    selected: str = ""                 # which candidate was chosen
    edr: float = 0.0                   # [ebits/s]
    n_candidates: int = 0              # feasible candidates compared
    iterations: int = 0                # LARAC iterations
    note: str = ""
    refs: dict = field(default_factory=dict)   # "max_eta", "shortest": the two Dijkstra paths (Route)

    @property
    def eta(self) -> float:
        return float(np.exp(-self.cost))

    @property
    def n_sats(self) -> int:
        return max(len(self.nodes) - 2, 0)


def shortest_path(weights: sp.csr_matrix, src: int, dst: int):
    dist, pred = dijkstra(weights, directed=False, indices=src, return_predecessors=True)
    if not np.isfinite(dist[dst]):
        return None
    path, v = [dst], dst
    while v != src:
        v = int(pred[v])
        path.append(v)
    return path[::-1]


def path_totals(graph, path):
    """(sum of cost, sum of length) along a node path."""
    c = l = 0.0
    for u, v in zip(path[:-1], path[1:]):
        a, b = min(u, v), max(u, v)
        c += graph.cost[a, b]
        l += graph.length[a, b]
    return float(c), float(l)


def path_edr(cost: float, length_km: float) -> float:
    return P.edr(np.exp(-cost), length_km)


def select_light_path(graph, l_max_km: float, max_iter: int = 50, score=path_edr) -> Route:
    """The path of largest score(cost, length) among the candidates within l_max_km (module docstring)."""
    src, dst = graph.src_node, graph.dst_node
    p_c = shortest_path(graph.cost, src, dst)
    if p_c is None:
        return Route([], np.inf, np.inf, "no path", note="the two HAPs are not connected")
    c_c, l_c = path_totals(graph, p_c)
    p_d = shortest_path(graph.length, src, dst)
    c_d, l_d = path_totals(graph, p_d)
    refs = {name: Route(p, c, l, "within limit" if l <= l_max_km else "too long")
            for name, p, c, l in (("max_eta", p_c, c_c, l_c), ("shortest", p_d, c_d, l_d))}
    if l_d > l_max_km:
        return Route(p_d, c_d, l_d, "infeasible", refs=refs,
                     note=f"shortest path {l_d:,.0f} km > limit {l_max_km:,.0f} km")

    candidates = [(p_d, c_d, l_d, "shortest path")]
    it = 0
    if l_c <= l_max_km:
        candidates.insert(0, (p_c, c_c, l_c, "max-eta path"))
    else:
        for it in range(1, max_iter + 1):
            lam = (c_c - c_d) / (l_d - l_c)
            w = graph.cost.copy()
            w.data = graph.cost.data + lam * graph.length.data
            r = shortest_path(w, src, dst)
            c_r, l_r = path_totals(graph, r)
            if np.isclose(c_r + lam * l_r, c_c + lam * l_c, rtol=1e-12, atol=1e-12):
                break
            if l_r <= l_max_km:
                p_d, c_d, l_d = r, c_r, l_r
                candidates.append((r, c_r, l_r, f"LARAC iterate {it}"))
            else:
                p_c, c_c, l_c = r, c_r, l_r

    best, seen = None, set()
    for nodes, c, l, label in candidates:
        if tuple(nodes) in seen:
            continue
        seen.add(tuple(nodes))
        e = score(c, l)
        if best is None or e > best[0] + 1e-12 or (abs(e - best[0]) <= 1e-12 and c < best[2]):
            best = (e, nodes, c, l, label)
    e, nodes, c, l, label = best
    return Route(nodes, c, l, "served", selected=label, edr=e, n_candidates=len(seen), iterations=it, refs=refs)
