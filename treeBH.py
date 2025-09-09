"""

A Python implementation of TreeBH

Implements the TreeBH procedure from: Bogomolov, M., Peterson, C. B., Benjamini, Y., & Sabatti, C. (2021).
Hypotheses on a tree: new error rates and testing strategies. Biometrika, 108(3), 575-590.
The original package is implemented in R: https://github.com/cbpeterson/TreeBH

The logic of the TreeBH correction:
- Hypotheses are on a tree with L+1 levels (0 = top/root, L = leaves).
- I provide p-values for the *leaf* level only.
- Parent-level p-values are computed by *combining* child-level p-values
  (following Bogomolov et al., the default: Simes; also supports Fisher under independence).
- Testing proceeds top-down. At level 0, apply BH at q(0).
- In TreeBH, at level ℓ we don’t use q(ℓ) directly. Instead, we adjust it by the product of the ratios (|S_r| / |F_r|)
over all ancestor families r. Formally: q_work(ℓ) = q(ℓ) × Π_ancestors (|S_r| / |F_r|). This reflects the selective
nature of testing only within families that were selected higher up. It ensures *selective FDR* control at level ℓ.
- If we want it to be a 'regular FDR correction, we can just have everything as leaves.

In the outputs:
- 'corrected_p_value' = BH's adjusted p-value (within each tested family; i.e., the bh_qvalue).
- For internal nodes: we should report the combined p-value (p_combined) at that level, and whether that node was selected
- For leaves (or any tested family): the within-family BH adjusted p-value (corrected_p_value)


References:
- TreeBH R manual (`get_TreeBH_selections`): default test="simes".
- Paper (Biometrika 2021): Simes at L-1, and under independence you may use any
  valid global-null combiner (e.g., Simes, Fisher, Stouffer) at higher levels.

Author: RonyHirsch
"""

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Union
import numpy as np
import pandas as pd
import math, textwrap
import matplotlib.pyplot as plt
import networkx as nx


# Delimiters used between ancestor and child labels
_DELIMS = r"[.\:\;\-\u2013\u2014\|\/>,]"
_SP = r"\s+"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _ok_ancestor(s: str, min_len: int) -> bool:
    s = (s or "").strip()
    return bool(s) and (len(s) >= min_len or " " in s)


def _right_delim_positions(s: str) -> List[int]:
    return [m.start() for m in re.finditer(_DELIMS, s)]


def _parent_prefix_candidates(parent: str, min_len: int) -> List[str]:
    """
    From a parent name like 'A. B: C', generate delimiter-bounded prefix candidates:
    ['A', 'A. B', 'A. B: C'] (each normalized & without trailing delimiters/spaces).
    Keep only informative candidates (len>=min_len or contains space).
    """
    p = _norm(parent)
    if not p:
        return []
    # Full without trailing delimiters
    p_full = re.sub(rf"(?:{_SP}?{_DELIMS})+{_SP}?$", "", p)
    cands = set()
    if p_full:
        cands.add(p_full)
    # Prefixes that stop *before* each delimiter
    for idx in _right_delim_positions(p):
        pref = _norm(p[:idx])
        pref = re.sub(rf"(?:{_SP}?{_DELIMS})+{_SP}?$", "", pref)
        if pref:
            cands.add(pref)
    return [c for c in cands if _ok_ancestor(c, min_len)]


def strip_ancestors_prefix(label: str, ancestor_names: List[str], *, min_ancestor_len_for_strip: int = 5) -> str:
    """
    Remove any *leading* sequence of ancestor prefixes (root→…→parent) from label.
    For each ancestor, try all delimiter-bounded prefix candidates; strip the LONGEST
    matching candidate at the front (plus optional delim/spaces).
    """
    c = _norm(label)
    if not c or not ancestor_names:
        return c

    cand_list = []
    for anc in ancestor_names:
        for cand in _parent_prefix_candidates(anc, min_ancestor_len_for_strip):
            pat = rf"^\s*{re.escape(cand)}(?:{_SP}(?:{_DELIMS}){_SP}|{_SP}|(?:{_DELIMS}){_SP})?"
            cand_list.append((cand, re.compile(pat, flags=re.IGNORECASE)))

    changed = True
    while changed:
        changed = False
        matches = []
        for cand, rx in cand_list:
            m = rx.match(c)
            if m:
                matches.append((len(cand), m))
        if matches:
            m = max(matches, key=lambda x: x[0])[1]  # longest candidate
            new = c[m.end():].strip()
            if new:
                c = new
                changed = True
    return c

@dataclass
class TreeIndex:
    nodes_df: pd.DataFrame
    parent_map: Dict[int, Optional[int]]
    name_map: Dict[int, str]
    children: Dict[int, List[int]]
    roots: List[int]
    depth: Dict[int, int]

def load_tree_index(csv_path: str) -> TreeIndex:
    """Read CSV once and build lightweight indices reused by both functions."""
    df = pd.read_csv(csv_path)
    cols_lower = {c.lower(): c for c in df.columns}
    id_col = cols_lower.get("node_id", "node_id")
    name_col = cols_lower.get("analysis_name", cols_lower.get("name", "analysis_name"))
    parent_col = cols_lower.get("parent_id", "parent_id")
    p_col = cols_lower.get("p-value", cols_lower.get("p_value", "p-value"))

    # Coerce
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    if parent_col in df.columns:
        df[parent_col] = pd.to_numeric(df[parent_col], errors="coerce").astype("Int64")
    else:
        df[parent_col] = pd.Series([pd.NA]*len(df), dtype="Int64")

    nodes_df = pd.DataFrame({
        "node_id": df[id_col].astype("Int64"),
        "name": df[name_col].astype(str),
        "parent_id": df[parent_col].astype("Int64"),
        "p_value": pd.to_numeric(df.get(p_col, np.nan), errors="coerce"),
    })

    parent_map: Dict[int, Optional[int]] = {}
    name_map: Dict[int, str] = {}
    children: Dict[int, List[int]] = {}
    roots: List[int] = []

    for r in nodes_df.itertuples(index=False):
        nid = int(r.node_id)
        pid = None if pd.isna(r.parent_id) else int(r.parent_id)
        parent_map[nid] = pid
        name_map[nid] = str(r.name)
        children.setdefault(nid, [])
        if pid is None:
            roots.append(nid)
        else:
            children.setdefault(pid, []).append(nid)

    # Depth via BFS
    depth: Dict[int, int] = {int(r.node_id): -1 for r in nodes_df.itertuples(index=False)}
    for rt in roots:
        depth[rt] = 0
    q = roots[:]
    while q:
        u = q.pop(0)
        for v in children.get(u, []):
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                q.append(v)

    # Deterministic order
    for k in list(children.keys()):
        children[k].sort(key=lambda nid: (name_map[nid], nid))
    roots.sort(key=lambda nid: (name_map[nid], nid))

    return TreeIndex(nodes_df, parent_map, name_map, children, roots, depth)


def ancestor_name_chain(nid: int, parent_map: Dict[int, Optional[int]], name_map: Dict[int, str]) -> List[str]:
    """Return names from root→…→parent for `nid`."""
    chain = []
    curr = parent_map.get(nid)
    seen = set()
    while curr is not None and curr not in seen:
        seen.add(curr)
        chain.append(name_map.get(curr, str(curr)))
        curr = parent_map.get(curr)
    chain.reverse()
    return chain


def _read_nodes_edges(csv_path: str) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Parse a node list CSV and construct a simple edge list.

    Expected input schema (case-insensitive column matching):
      - node_id:     unique integer id per node (roots, internal, leaves)
      - analysis_name (alias: name): human-readable name
      - parent_id:   parent node id (NA for roots)
      - p-value:     p-value for *leaves only* (NA for internal nodes / roots)

    Returns:
    nodes_df : pd.DataFrame with columns: node_id:, name, parent_id, p_value. parent_id is NA for roots.
    edges : list of tuples, edges of the form (parent_id, child_id), only for rows with non-NA parent_id.
    """
    df = pd.read_csv(csv_path)

    # case-insensitive header resolution
    cols_lower = {c.lower(): c for c in df.columns}
    node_id_col = cols_lower.get("node_id", "node_id")
    name_col = cols_lower.get("analysis_name", cols_lower.get("name", "analysis_name"))
    parent_col = cols_lower.get("parent_id", "parent_id")
    p_col = cols_lower.get("p-value", cols_lower.get("p_value", "p-value"))

    # coerce types (nullable for ids to preserve NA)
    df[node_id_col] = pd.to_numeric(df[node_id_col], errors="coerce").astype("Int64")
    if parent_col in df.columns:
        df[parent_col] = pd.to_numeric(df[parent_col], errors="coerce").astype("Int64")
    else:
        df[parent_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")

    nodes_df = pd.DataFrame({
        "node_id": df[node_id_col].astype("Int64"),
        "name": df[name_col].astype(str),
        "parent_id": df[parent_col].astype("Int64"),
        "p_value": df[p_col],
    })

    """
    build edges only when parent_id exists: for each row r in nodes_df, if parent_id is not NA, 
    append (parent_id, node_id)
    """
    edges: List[Tuple[int, int]] = []
    for r in nodes_df.itertuples(index=False):
        if pd.notna(r.parent_id):
            edges.append((int(r.parent_id), int(r.node_id)))
    return nodes_df, edges


def _levels_by_depth(nodes_df: pd.DataFrame) -> Dict[int, int]:
    """
    Compute the depth of each node (root depth = 0) using an iterative traversal:
    Identify roots (parent_id NA), then traverse level by level (breadth-first), assigning depth[parent]+1 to children.
    - nodes_df: DataFrame with columns [node_id, parent_id]
    - Returns: mapping between node_id and depth (0 for roots, increasing by one per edge downwards)
    """
    parent_of: Dict[int, Optional[int]] = {}
    for r in nodes_df.itertuples(index=False):
        # Convert pandas NA to None for easier Python logic
        pid = None if pd.isna(r.parent_id) else int(r.parent_id)
        parent_of[int(r.node_id)] = pid

    # initialize all depths to unknown (-1); roots will be set to 0
    depth: Dict[int, int] = {nid: -1 for nid in parent_of}

    # find roots: nodes with parent None
    roots = [nid for nid, pid in parent_of.items() if pid is None]

    # queue for BFS, start with roots at depth 0
    queue: List[int] = []
    for r in roots:
        depth[r] = 0
        queue.append(r)

    """
    Iterative BFS (breadth-first search). 
    BFS is widely used to compute shortest-path distances in unweighted graphs, 
    on a tree/forest, “distance to root” is exactly the depth.
    What we do: BFS over the implicit parent-children relation to build children index and avoid repeated scans
    Put all roots in a queue with depth[root] = 0.
    Repeatedly pop a node u; for each child v set depth[v] = depth[u] + 1 the first time we see it 
    and push v on the queue.
    """
    children_of: Dict[int, List[int]] = {nid: [] for nid in parent_of}
    for nid, pid in parent_of.items():
        if pid is not None:
            children_of[pid].append(nid)

    while queue:
        u = queue.pop(0)  # pop the first item
        du = depth[u]  # current node's depth
        for v in children_of.get(u, []):
            if depth[v] == -1:  # if not yet visited
                depth[v] = du + 1  # child's depth = parent depth + 1
                queue.append(v)  # schedule child

    return depth


def _simple_layered_positions(nodes_df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """
    A legible layered (x,y) layout without Graphviz.
    - y = -depth from the root
    - x locations are evenly spaced per layer (centered)
    """
    depth = _levels_by_depth(nodes_df)
    layers: Dict[int, List[int]] = {}
    for u, d in depth.items():
        layers.setdefault(d, []).append(u)
    for d in layers:
        layers[d] = sorted(layers[d])

    pos: Dict[int, Tuple[float, float]] = {}
    for d, nodes in layers.items():
        xs = [0.0] if len(nodes) == 1 else list(np.linspace(-1.0, 1.0, len(nodes)))
        for x, u in zip(xs, nodes):
            pos[u] = (x, -d)
    return pos


def plot_treebh(
    csv_path: str,
    q: float = 0.05,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
    color_sig: str = "#354f52",
    color_nonsig: str = "#cad2c5",
    node_size: int = 900,
    font_size: int = 9,
    show: bool = False,
    # Legibility knobs
    leaf_max_chars: int = 18,
    show_internal_p_only: bool = True,
    annotate_internal_side: bool = True,
    side_pad: float = 0.15,
    wrap_side_labels: int = 28,
    draw_leaf_names_right: bool = True,
    # Shared stripping knobs
    strip_ancestor_prefix: bool = True,
    min_ancestor_len_for_strip: int = 5,
):
    import math, textwrap
    import matplotlib.pyplot as plt
    import networkx as nx

    # 1) Shared tree index
    ti = load_tree_index(csv_path)

    # 2) Run TreeBH (unchanged core)
    td = build_groups_from_edge_csv(csv_path)
    L = td.groups.shape[1]
    res = get_treebh_selections(td.leaves_df, td.groups, q=[q] * L)

    # Per-level p-values, mapped to node ids by depth
    level_tables = [tbl.set_index("group_id")["p_combined"].to_dict() for tbl in res.group_pvalues]

    # node p-values per node id (leaf: original p; internal: combined)
    node_p: Dict[int, float] = {}
    for r in ti.nodes_df.itertuples(index=False):
        nid = int(r.node_id)
        dep = ti.depth[nid]
        if pd.notna(r.p_value):
            node_p[nid] = float(r.p_value)
        else:
            node_p[nid] = float(level_tables[dep].get(nid, np.nan))

    # selected flags at level
    selected_nodes: set[Tuple[int, int]] = set()
    for d, tbl in enumerate(res.per_level_tables):
        if tbl is None or tbl.empty:
            continue
        if {"bh_qvalue", "q_work"}.issubset(tbl.columns):
            for row in tbl.itertuples(index=False):
                gid = int(getattr(row, "group_id"))
                if pd.notna(getattr(row, "bh_qvalue")) and pd.notna(getattr(row, "q_work")) \
                   and float(getattr(row, "bh_qvalue")) <= float(getattr(row, "q_work")):
                    selected_nodes.add((d, gid))
        elif "rejected" in tbl.columns:
            for row in tbl.itertuples(index=False):
                if bool(getattr(row, "rejected")):
                    selected_nodes.add((d, int(getattr(row, "group_id"))))

    # 3) Build graph, using the SAME stripping for labels
    def _abbr(s: str, n: int) -> str:
        return s if len(s) <= n else (s[:max(0, n-1)] + "…")

    def _wrap(s: str, w: int) -> str:
        import textwrap
        return "\n".join(textwrap.wrap(s, width=w, break_long_words=False, replace_whitespace=False)) if w else s

    G = nx.DiGraph()
    for r in ti.nodes_df.itertuples(index=False):
        nid = int(r.node_id)
        dep = ti.depth[nid]
        is_leaf = pd.notna(r.p_value)
        full_name = ti.name_map[nid]
        anc_names = ancestor_name_chain(nid, ti.parent_map, ti.name_map)
        stripped = strip_ancestors_prefix(full_name, anc_names, min_ancestor_len_for_strip=min_ancestor_len_for_strip) \
                   if strip_ancestor_prefix else full_name

        is_selected = (dep, nid) in selected_nodes
        color = color_sig if is_selected else color_nonsig
        p_show = node_p[nid]

        # minimal in-node text; leaf name drawn outside
        if is_leaf:
            node_label = (f"p={p_show:.3g}" if math.isfinite(p_show) else "p=NA") if draw_leaf_names_right \
                         else (_abbr(stripped, leaf_max_chars) + (f"\n(p={p_show:.3g})" if math.isfinite(p_show) else ""))
        else:
            if show_internal_p_only:
                node_label = f"p={p_show:.3g}" if math.isfinite(p_show) else "p=NA"
            else:
                nm = _abbr(stripped, max(leaf_max_chars, 14))
                node_label = nm + (f"\n(p={p_show:.3g})" if math.isfinite(p_show) else "")

        G.add_node(nid,
                   label=node_label,
                   stripped=stripped,
                   full_name=full_name,
                   p=p_show,
                   color=color,
                   level=dep,
                   is_leaf=is_leaf)

    # edges
    E = []
    for pid, kids in ti.children.items():
        for c in kids:
            E.append((pid, c))
    G.add_edges_from(E)

    # 4) Layout & draw
    pos = _simple_layered_positions(ti.nodes_df)  # your existing layered layout

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    nx.draw_networkx_edges(G, pos, arrows=False, edge_color="#6c757d", width=1.0, ax=ax)

    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size,
                           linewidths=0.8, edgecolors="#2f3e46", ax=ax)

    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size, ax=ax)

    # leaf names to the right (stripped + abbreviated)
    if draw_leaf_names_right:
        for n, (x, y) in pos.items():
            if not G.nodes[n]["is_leaf"]:
                continue
            ax.text(
                x + 0.03, y, _abbr(G.nodes[n]["stripped"], leaf_max_chars),
                ha="left", va="center", fontsize=font_size,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
                clip_on=False
            )

    # side annotations (stripped internal names)
    if annotate_internal_side:
        xs = [x for x, _ in pos.values()]
        x_min, x_max = (min(xs), max(xs)) if xs else (-1.0, 1.0)
        x_pad = (x_max - x_min) if xs else 2.0
        x_side = x_min - side_pad * (x_pad if x_pad > 0 else 2.0)

        by_level: Dict[int, List[int]] = {}
        for n in G.nodes:
            if not G.nodes[n]["is_leaf"]:
                by_level.setdefault(ti.depth[n], []).append(n)

        for d, nodes_at_level in by_level.items():
            if not nodes_at_level:
                continue
            ys = sorted({pos[n][1] for n in nodes_at_level})
            ax.text(x_side, ys[0] + 0.2, f"Level {d} groups:", ha="left", va="bottom",
                    fontsize=font_size, fontweight="bold", color="#2f3e46",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f9fa", edgecolor="#e0e0e0", alpha=0.9))
            for i, n in enumerate(sorted(nodes_at_level, key=lambda k: G.nodes[k]["stripped"])):
                nm_wrapped = _wrap(G.nodes[n]["stripped"], wrap_side_labels)
                yy = ys[0] - 0.3 - 0.22 * i
                ax.text(x_side, yy, f"• {nm_wrapped}", ha="left", va="top",
                        fontsize=font_size-1, color="#334", linespacing=1.1,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_title(f"TreeBH (Simes) — q={q}", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def simes_pvalue(pvals: Iterable[float]) -> float:
    """
    Simes (1986) p-value combiner on a 1D iterable of p-values.
    Ignores NaNs. Returns min_i m * p_(i) / i (truncated at 1), where m is the
    number of finite p-values and p_(i) are their order statistics.
    """
    p = np.asarray(list(pvals), dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    p_sorted = np.sort(p)
    m = p_sorted.size
    simes_vals = p_sorted * m / np.arange(1, m + 1, dtype=float)
    return float(np.minimum(1.0, np.nanmin(simes_vals)))


def fisher_pvalue(pvals: Iterable[float]) -> float:
    """
    Fisher (1925) method combining independent p-values.
    Statistic: T = -2 * sum log(p_i) ~ chi2_{2k} under independence; return sf.
    """
    p = np.asarray(list(pvals), dtype=float)
    p = p[np.isfinite(p)]
    k = p.size
    if k == 0:
        return np.nan
    if np.any(p <= 0):
        p = np.clip(p, 1e-300, 1.0)
    stat = -2.0 * float(np.sum(np.log(p)))
    try:
        from scipy.stats import chi2  # type: ignore
        return float(chi2.sf(stat, df=2 * k))
    except Exception as e:
        raise RuntimeError(
            "Fisher combination requires SciPy (scipy.stats.chi2). Install scipy or use combine='simes'."
        ) from e


def combine_pvalues(pvals: Iterable[float], method: str) -> float:
    """
    Dispatch to requested p-value combiner ('simes' or 'fisher').
    """
    method = method.lower()
    if method == "simes":
        return simes_pvalue(pvals)
    if method == "fisher":
        return fisher_pvalue(pvals)
    raise ValueError(f"Unknown combination method: {method!r}")


def bh_step_up(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg within-family

    pvals : May include NaN; they are treated as not-tested (reject=False, qval=NaN).
    alpha : Target FDR level for this family.

    Returns:
    - rejects : bool ndarray, True where the hypothesis is rejected.
    - qvals : float ndarray, BH adjusted p-values (q-values); NaN where input p was NaN.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]

    finite_mask = np.isfinite(p_sorted)
    p_finite = p_sorted[finite_mask]
    m = p_finite.size

    qvals_sorted = np.full(n, np.nan, dtype=float)
    rejects_sorted = np.zeros(n, dtype=bool)

    if m == 0 or alpha <= 0:
        inv = np.empty_like(order)
        inv[order] = np.arange(n)
        return rejects_sorted[inv], qvals_sorted[inv]

    # BH critical values i/m * alpha
    crit = alpha * np.arange(1, m + 1, dtype=float) / m
    below = p_finite <= crit
    k = (np.where(below)[0].max() + 1) if below.any() else 0
    if k > 0:
        rejects_sorted[:k] = True

    # Adjusted p-values (monotone nonincreasing when sorted)
    # Here we compute m * p(i) / i and apply a reverse cumulative minimum
    adj = np.minimum.accumulate((p_finite[::-1] * m / np.arange(m, 0, -1, dtype=float)))[::-1]
    adj = np.minimum(adj, 1.0)
    qvals_sorted[finite_mask] = adj

    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    return rejects_sorted[inv], qvals_sorted[inv]


@dataclass(frozen=True)
class Node:
    node_id: int
    name: str
    parent_id: Optional[int]  # None for roots
    pvalue: Optional[float]  # Only leaves (tests) have p-values


@dataclass
class TreeData:
    """
    Container tying together leaves and their group ids across levels.

    - leaves_df: rows are leaves with their p-values
    - groups: N x L int matrix mapping each leaf to its group id at each level
    (col 0 = top/root, col L-1 = leaf itself)
    - levels: list of unique group ids per level (useful for iteration)
    - level_names: mapping gid -> name per level (if available in CSV)
    - leaf_order: the leaf node ids in the same order as rows in leaves_df
    """
    leaves_df: pd.DataFrame
    groups: np.ndarray
    levels: List[List[int]]
    level_names: List[Dict[int, str]]
    leaf_order: List[int]


def build_groups_from_edge_csv(csv_path: str, id_col: str = "node_id", name_col: str = "analysis_name",
                               parent_col: str = "parent_id", p_col: str = "p-value") -> TreeData:
    """
    Build `TreeData` from a single CSV listing nodes and their parent links.

    - Roots: parent_id NA and p-value NA = highest groups
    - Internal groups: parent_id set, p-value NA
    - Leaves (tests): parent_id set, p-value set

    Returns an N x L `groups` matrix whose i-th row contains the group id for
    the i-th leaf at every level (0..L-1), where column L-1 equals the leaf id.

    We construct a `Node` dictionary for quick parent navigation, compute each node's depth iteratively,
    then for each leaf, walk its path to the root once to fill its row in `groups`
    """
    df = pd.read_csv(csv_path)

    # basic coercions
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    if parent_col in df.columns:
        df[parent_col] = pd.to_numeric(df[parent_col], errors="coerce").astype("Int64")
    else:
        df[parent_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")

    # node map
    nodes: Dict[int, Node] = {}
    for _, row in df.iterrows():
        nid = int(row[id_col]) if pd.notna(row[id_col]) else None
        if nid is None:
            continue
        pid = int(row[parent_col]) if (parent_col in df.columns and pd.notna(row[parent_col])) else None
        pval = float(row[p_col]) if pd.notna(row[p_col]) else None
        name = str(row[name_col]) if name_col in df.columns else str(nid)
        nodes[nid] = Node(node_id=nid, name=name, parent_id=pid, pvalue=pval)

    # identify leaves (tests)
    leaves = [n for n in nodes.values() if n.pvalue is not None]
    if not leaves:
        raise ValueError("No leaves with p-values found in CSV.")

    # compute depth for all nodes (root = 0)
    parent_of = {nid: n.parent_id for nid, n in nodes.items()}
    depth: Dict[int, int] = {nid: -1 for nid in nodes}
    roots = [nid for nid, pid in parent_of.items() if pid is None]
    for r in roots:
        depth[r] = 0

    # build children index for BFS
    children_of: Dict[int, List[int]] = {nid: [] for nid in nodes}
    for nid, pid in parent_of.items():
        if pid is not None and pid in children_of:
            children_of[pid].append(nid)
    queue = roots[:]
    while queue:
        u = queue.pop(0)
        for v in children_of.get(u, []):
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)

    # L = deepest leaf depth + 1 (levels are 0..L-1)
    max_depth = max(depth[leaf.node_id] for leaf in leaves)
    L = max_depth + 1

    # Helper: walk root->leaf path once and cache
    path_cache: Dict[int, List[int]] = {}

    def root_to_leaf_path(node_id: int) -> List[int]:
        # if cached, return immediately
        if node_id in path_cache:
            return path_cache[node_id]
        # otherwise climb to root then reverse
        path: List[int] = []
        curr = nodes[node_id]
        while curr is not None:
            path.append(curr.node_id)
            curr = nodes[curr.parent_id] if curr.parent_id is not None else None
        path = path[::-1]
        path_cache[node_id] = path
        return path

    # build N x L group matrix
    leaves_sorted = sorted(leaves, key=lambda x: x.node_id)
    leaf_ids = [leaf.node_id for leaf in leaves_sorted]
    N = len(leaf_ids)
    groups = np.zeros((N, L), dtype=int)

    for i, leaf_id in enumerate(leaf_ids):
        path = root_to_leaf_path(leaf_id)
        # if the leaf resides at depth < max_depth, pad by its leaf id so that every row has length L (matches the original behavior)
        if len(path) < L:
            path = path + [path[-1]] * (L - len(path))
        groups[i, :] = np.asarray(path[:L], dtype=int)

    # per-level metadata
    levels: List[List[int]] = []
    level_names: List[Dict[int, str]] = []
    for d in range(L):
        gids = list(dict.fromkeys(groups[:, d].tolist()))
        levels.append(gids)
        level_names.append({gid: nodes[gid].name for gid in gids})

    leaves_df = pd.DataFrame({
        "leaf_node_id": leaf_ids,
        "leaf_name": [nodes[i].name for i in leaf_ids],
        "pval": [nodes[i].pvalue for i in leaf_ids],
    })

    return TreeData(
        leaves_df=leaves_df,
        groups=groups,
        levels=levels,
        level_names=level_names,
        leaf_order=leaf_ids,
    )


def _compute_level_pvalues(leaves_df: pd.DataFrame, groups: np.ndarray, tests: Union[str, List[str]]) -> List[
    pd.DataFrame]:
    """
    Compute per-level *group* p-values from leaves upward.

    leaves_df : DataFrame with a 'pval' column of size N (one per leaf)
    groups    : (N x L) integer matrix; column d holds the group id at level d
    tests     : str or list[str]; for each upward step choose 'simes' or 'fisher'

    Returns: A list of length L. Entry d is a df with columns ['group_id', 'p_combined', 'n_children'] for level d.

    - Level L-1 (leaves): group ids are the leaves, p_combined equals leaf p-values
    - For d < L-1: combine the child groups' p-values into each parent's p-value
    """
    N, L = groups.shape

    # normalize test methods per upward step
    if isinstance(tests, str):
        tests_vec = [tests] * (L - 1)
    else:
        tests_vec = list(tests)
        if len(tests_vec) != (L - 1):
            raise ValueError(f"'tests' must have length {L - 1} (one per upward combination step).")

    # leaf level table (d = L-1)
    leaf_ids = groups[:, L - 1]
    leaf_p = pd.DataFrame({
        "group_id": leaf_ids,
        "p_combined": leaves_df["pval"].to_numpy(),
        "n_children": np.ones(N, dtype=int),
    }).drop_duplicates(subset="group_id", keep="first")

    level_tables: List[pd.DataFrame] = [None] * L  # type: ignore
    level_tables[L - 1] = leaf_p

    # precompute child group ids per parent per level to avoid recomputing
    children_gid: List[Dict[int, np.ndarray]] = []
    for d in range(L - 1):
        mapping: Dict[int, np.ndarray] = {}
        for gid in np.unique(groups[:, d]):
            where_parent = np.where(groups[:, d] == gid)[0]
            mapping[int(gid)] = np.unique(groups[where_parent, d + 1])
        children_gid.append(mapping)

    # from child level to parent level
    for d in range(L - 2, -1, -1):
        parents = np.unique(groups[:, d]).astype(int)
        rows: List[Dict[str, float]] = []

        child_table = level_tables[d + 1].set_index("group_id")["p_combined"].to_dict()
        method = tests_vec[d]
        for gid in parents:
            child_gids = children_gid[d][int(gid)]
            child_p = [child_table.get(int(cg), np.nan) for cg in child_gids]
            pval = combine_pvalues(child_p, method)
            rows.append({"group_id": int(gid), "p_combined": float(pval), "n_children": int(len(child_gids))})
        level_tables[d] = pd.DataFrame(rows, columns=["group_id", "p_combined", "n_children"])  # type: ignore

    return level_tables


@dataclass
class TreeBHResult:
    """
    Result of TreeBH.
    - selections_matrix: N x L array aligned to `groups` (marking first row per selected group)
    - per_level_tables: list of DataFrames for families actually *tested* at each level
    - group_pvalues:    per-level group p-values (from leaves upward)
    """
    selections_matrix: np.ndarray
    per_level_tables: List[pd.DataFrame]
    group_pvalues: List[pd.DataFrame]


def get_treebh_selections(leaves_df: pd.DataFrame, groups: np.ndarray, q: List[float],
                          tests: Union[str, List[str]] = "simes", ) -> TreeBHResult:
    """
    Run TreeBH top-down and collect selections & metadata.
    We compute upward p-values ONCE (`_compute_level_pvalues`) then sweep top-down.
    Within each selected parent family we run BH with the *working target*
    q_work(d) = q[d] * Π_ancestors (|S_r|/|F_r|). This is the working BH target used when you run BH inside a selected
    parent family at level ℓ.

    - leaves_df : DataFrame with columns ['leaf_node_id','leaf_name','pval']
    - groups    : (N x L) matrix; group id for each leaf at each level
    - q         : list of length L with target error bounds per level
    - tests     : combiner(s) for upward p-values ('simes' or 'fisher')

    Returns:
    TreeBHResult
        - selections_matrix marks selections (first row carrying each selected group)
        - per_level_tables include p-values, BH q-values, q_work, and flags per tested family
        - group_pvalues provide p_combined per group at each level
    """
    N, L = groups.shape
    if len(q) != L:
        raise ValueError(f"'q' must have length {L} (one bound per level).")

    level_pvals = _compute_level_pvalues(leaves_df, groups, tests)

    selections = np.zeros((N, L), dtype=int)
    per_level_tables: List[pd.DataFrame] = []

    # Map (level d, gid) → row indices of leaves under that group at level d
    group_to_leaf_idxs_per_level: List[Dict[int, np.ndarray]] = []
    for d in range(L):
        mapping: Dict[int, np.ndarray] = {}
        gids = level_pvals[d]["group_id"].tolist()
        for gid in gids:
            mapping[int(gid)] = np.where(groups[:, d] == gid)[0]
        group_to_leaf_idxs_per_level.append(mapping)

    # S/F counts
    S_counts: List[Dict[int, int]] = [dict() for _ in range(L)]
    F_counts: List[Dict[int, int]] = [dict() for _ in range(L)]

    # Level 0 (top)
    gdf0 = level_pvals[0].copy().reset_index(drop=True)
    p0 = gdf0["p_combined"].to_numpy()
    rejects0, qvals0 = bh_step_up(p0, q[0])
    gdf0["rejected"] = rejects0
    gdf0["bh_qvalue"] = qvals0
    gdf0["q_work"] = q[0]  # explicit for completeness at the top level
    per_level_tables.append(gdf0.copy())

    # mark selections (R-style: mark only the first leaf-row that carries the group)
    level0_groups = groups[:, 0]
    for gid, rej in zip(gdf0["group_id"].tolist(), rejects0.tolist()):
        idxs = np.where(level0_groups == gid)[0]
        if idxs.size > 0:
            selections[idxs[0], 0] = 1 if rej else 0

    S0 = int(rejects0.sum())
    F0 = int(len(rejects0))
    S_counts[0] = {"__GLOBAL__": S0}
    F_counts[0] = {"__GLOBAL__": F0}

    # helper: product Π_{r=0..parent_level} (|S_r|/|F_r|) along ancestry chain
    # we cache q_work factors within a level for speed and clarity
    qprod_cache: Dict[Tuple[int, int], float] = {}

    def ancestor_sf_product(parent_level: int, parent_gid: int) -> float:
        # check the cache first
        key = (parent_level, int(parent_gid))
        if key in qprod_cache:
            return qprod_cache[key]
        if parent_level < 0:
            qprod_cache[key] = 1.0
            return 1.0
        # start with level-0 global ratio
        prod = (S0 / F0) if F0 > 0 else 0.0
        # Identify a representative leaf under the parent group to trace ancestors
        leaf_idx = int(group_to_leaf_idxs_per_level[parent_level][int(parent_gid)][0])
        for r in range(1, parent_level + 1):
            anc_gid = int(groups[leaf_idx, r - 1])
            S_r = S_counts[r].get(anc_gid, 0)
            F_r = F_counts[r].get(anc_gid, 0)
            if F_r == 0:
                prod = 0.0
                break
            prod *= (S_r / F_r)
        qprod_cache[key] = float(prod)
        return float(prod)

    # Levels 1..L-1
    for d in range(1, L):
        parents = np.unique(groups[:, d - 1]).astype(int)
        child_table = level_pvals[d].set_index("group_id")  # has p_combined, n_children
        tested_rows: List[pd.DataFrame] = []
        S_counts[d] = {}
        F_counts[d] = {}

        for pg in parents:
            leaf_idxs = group_to_leaf_idxs_per_level[d - 1][int(pg)]
            # only test within selected parents
            if selections[leaf_idxs[0], d - 1] != 1:
                continue

            child_gids = np.unique(groups[leaf_idxs, d]).astype(int)
            child_df = child_table.loc[list(map(int, child_gids))].copy()
            child_df["parent_gid"] = int(pg)
            child_df["level"] = d

            q_work = float(q[d]) * ancestor_sf_product(parent_level=d - 1, parent_gid=int(pg))

            pvals_children = child_df["p_combined"].to_numpy()
            rejects, qvals = bh_step_up(pvals_children, q_work)
            child_df["rejected"] = rejects
            child_df["bh_qvalue"] = qvals
            child_df["q_work"] = q_work

            S_counts[d][int(pg)] = int(rejects.sum())
            F_counts[d][int(pg)] = int(len(rejects))

            # mark selections (first representative row carries the 1)
            for gid, rej in zip(child_df.index.tolist(), rejects.tolist()):
                leaf_idxs_child = group_to_leaf_idxs_per_level[d][int(gid)]
                if leaf_idxs_child.size > 0:
                    selections[leaf_idxs_child[0], d] = 1 if rej else 0

            tested_rows.append(child_df.reset_index(names="group_id"))

        if tested_rows:
            per_level_tables.append(pd.concat(tested_rows, axis=0, ignore_index=True))
        else:
            per_level_tables.append(
                pd.DataFrame(columns=[
                    "group_id", "p_combined", "n_children", "parent_gid", "level", "rejected", "bh_qvalue", "q_work"
                ])
            )

    return TreeBHResult(
        selections_matrix=selections,
        per_level_tables=per_level_tables,
        group_pvalues=level_pvals,
    )


def manage_treebh(all_save_path: str, csv_path: str, q: float = 0.05, tag: Optional[str] = None,
                  write_csvs: bool = True, plot: bool = False, plot_filename: Optional[str] = None) -> Dict[str, str]:
    """
    Manage a full TreeBH run (Simes-only) and write outputs
    """

    os.makedirs(all_save_path, exist_ok=True)

    # Build tree and run TreeBH
    td = build_groups_from_edge_csv(csv_path)
    L = td.groups.shape[1]
    res = get_treebh_selections(td.leaves_df, td.groups, q=[q] * L)  # default 'simes'

    base = f"treebh{('_' + tag) if tag else ''}"
    p_groups = os.path.join(all_save_path, f"{base}_groups.csv")
    p_group_p = os.path.join(all_save_path, f"{base}_group_pvalues.csv")
    p_select = os.path.join(all_save_path, f"{base}_selections.csv")
    p_disc = os.path.join(all_save_path, f"{base}_discoveries.csv")
    p_flip = os.path.join(all_save_path, f"{base}_flip.csv")
    # removed report
    p_augmented_input = os.path.join(all_save_path, f"{base}_input_with_treebh.csv")

    # Collate outputs from result (unchanged)
    groups_df = pd.DataFrame(td.groups, columns=[f"level_{i}_group_id" for i in range(L)])
    groups_df.insert(0, "leaf_node_id", td.leaf_order)
    groups_df.insert(1, "leaf_name", td.leaves_df["leaf_name"].tolist())
    groups_df.insert(2, "leaf_pval", td.leaves_df["pval"].tolist())

    gpv_all = []
    for lvl, gdf in enumerate(res.group_pvalues):
        tmp = gdf.copy()
        tmp["level"] = lvl
        gpv_all.append(tmp)
    gpv_all = pd.concat(gpv_all, axis=0, ignore_index=True)

    # selections.csv (per_level_tables stacked)
    sel_all_list = []
    for lvl, tbl in enumerate(res.per_level_tables):
        tmp = tbl.copy()
        tmp["level"] = lvl
        sel_all_list.append(tmp)
    sel_all = pd.concat(sel_all_list, axis=0, ignore_index=True) if sel_all_list else pd.DataFrame()

    # read input CSV once to attach names and raw p-values
    raw = pd.read_csv(csv_path)

    # standardize types/columns if present
    if "node_id" in raw.columns:
        raw["node_id"] = pd.to_numeric(raw["node_id"], errors="coerce").astype("Int64")
    # allow either 'p-value' or 'p_value'
    pcol = "p-value" if "p-value" in raw.columns else ("p_value" if "p_value" in raw.columns else None)
    if pcol is not None:
        raw[pcol] = pd.to_numeric(raw[pcol], errors="coerce")

    # attach names (if available) to selections
    if {"node_id", "analysis_name"}.issubset(raw.columns) and "group_id" in sel_all.columns:
        name_map = {int(r["node_id"]): str(r["analysis_name"])
                    for _, r in raw.iterrows() if pd.notna(r["node_id"])}
        sel_all["group_name"] = sel_all["group_id"].map(name_map)
        if "parent_gid" in sel_all.columns:
            sel_all["parent_name"] = sel_all["parent_gid"].map(name_map)

    # standardize BH column name once (selections)
    if "bh_qvalue" in sel_all.columns and "corrected_p_value" not in sel_all.columns:
        sel_all = sel_all.rename(columns={"bh_qvalue": "corrected_p_value"})

    # attach p_original to selections (if you keep discoveries/flip below)
    if "group_id" in sel_all.columns and pcol is not None and "node_id" in raw.columns:
        orig_p_map = {int(r["node_id"]): (float(r[pcol]) if pd.notna(r[pcol]) else np.nan)
                      for _, r in raw.iterrows() if pd.notna(r["node_id"])}
        sel_all["p_original"] = sel_all["group_id"].map(orig_p_map)

    # reorder to place p_original before p_combined (if both exist)
    cols = list(sel_all.columns)
    if "p_combined" in cols and "p_original" in cols:
        cols.remove("p_original")
        insert_at = cols.index("p_combined")
        cols = cols[:insert_at] + ["p_original"] + cols[insert_at:]
        sel_all = sel_all[cols]

    # === NEW: Build an augmented copy of the input with corrected_p_value and rejected per node_id ===
    # Start by default: NaN q-values and False rejected
    augmented = raw.copy()
    if "corrected_p_value" not in augmented.columns:
        augmented["corrected_p_value"] = np.nan
    if "rejected" not in augmented.columns:
        augmented["rejected"] = False

    # Create a lookup from all tested nodes across levels:
    # sel_all contains rows for any node that was actually tested at some level,
    # with columns 'group_id', 'corrected_p_value', and 'rejected'.
    tested_map_q = {}
    tested_map_rej = {}

    if not sel_all.empty and "group_id" in sel_all.columns:
        # Ensure column exists regardless of naming in earlier steps
        corr_col = "corrected_p_value" if "corrected_p_value" in sel_all.columns else (
            "bh_qvalue" if "bh_qvalue" in sel_all.columns else None
        )

        if corr_col is not None:
            for r in sel_all.itertuples(index=False):
                gid = getattr(r, "group_id", None)
                if gid is None:
                    continue
                qv = getattr(r, corr_col, np.nan)
                rej = getattr(r, "rejected", False)
                # last write wins if duplicates; that's fine because it's the same node at a fixed level
                tested_map_q[int(gid)] = (float(qv) if pd.notna(qv) else np.nan)
                tested_map_rej[int(gid)] = bool(rej)

    # Now map these onto the input 'node_id's
    if "node_id" in augmented.columns:
        # Fill corrected_p_value where available
        augmented["corrected_p_value"] = augmented["node_id"].apply(
            lambda nid: tested_map_q.get(int(nid), np.nan) if pd.notna(nid) else np.nan
        )
        # Fill rejected where available; keep False for untested
        augmented["rejected"] = augmented["node_id"].apply(
            lambda nid: tested_map_rej.get(int(nid), False) if pd.notna(nid) else False
        )

    # Write files
    if write_csvs:
        groups_df.to_csv(p_groups, index=False)
        gpv_all.to_csv(p_group_p, index=False)
        sel_all.to_csv(p_select, index=False)

        # discoveries
        keep_cols = [
            "level", "group_id", "group_name", "p_original", "p_combined",
            "corrected_p_value", "parent_gid", "parent_name", "rejected",
            "n_children", "q_work"
        ]
        disc = sel_all[[c for c in keep_cols if c in sel_all.columns]].copy()
        disc.to_csv(p_disc, index=False)

        # flip/diff file: raw significant vs q_work but BH not rejected (leaf rows only)
        if {"p_original", "q_work", "rejected"}.issubset(sel_all.columns):
            mask_flip = (
                    sel_all["p_original"].notna()
                    & sel_all["q_work"].notna()
                    & (sel_all["p_original"] <= sel_all["q_work"])
                    & (~sel_all["rejected"].fillna(False))
            )
            flip = sel_all.loc[mask_flip, [c for c in keep_cols if c in sel_all.columns]].copy()
        else:
            flip = pd.DataFrame(columns=[c for c in keep_cols if c in sel_all.columns])
        flip.to_csv(p_flip, index=False)

        # write the augmented input-with-results CSV
        augmented.to_csv(p_augmented_input, index=False)

    # Optional plot
    plot_path = None
    if plot:
        plot_path = os.path.join(all_save_path, plot_filename) if plot_filename else os.path.join(
            all_save_path, f"{base}_plot.svg")
        plot_treebh(csv_path=csv_path, q=q, save_path=plot_path, show=False)

    # Return paths (no report; includes augmented input)
    out = {
        "groups_csv": p_groups,
        "group_pvalues_csv": p_group_p,
        "selections_csv": p_select,
        "discoveries_csv": p_disc,
        "flip_csv": p_flip,
        "input_with_treebh_csv": p_augmented_input,
    }
    if plot:
        out["plot_svg"] = plot_path
    return out


def save_report_ready_table(res: "TreeBHResult", csv_path: str) -> None:
    """
    Prepare a concise, publication-ready summary table of TreeBH results.
    Columns (if available):
    - level: level index (0=root)
    - group_id, group_name
    - p_original (leaf raw p-value, if applicable)
    - p_combined (Simes/Fisher at this level)
    - corrected_p_value (BH q-value within-family)
    - q_work (working FDR bound used for this family)
    - rejected (bool flag)
    - parent_gid, parent_name (context)
    Saves to `csv_path`.
    """
    sel_all = []
    for lvl, tbl in enumerate(res.per_level_tables):
        tmp = tbl.copy()
        tmp["level"] = lvl
        sel_all.append(tmp)
    if not sel_all:
        pd.DataFrame().to_csv(csv_path, index=False)
        return
    sel_all = pd.concat(sel_all, axis=0, ignore_index=True)

    # Ensure standard column names
    if "bh_qvalue" in sel_all.columns:
        sel_all = sel_all.rename(columns={"bh_qvalue": "corrected_p_value"})

    keep = [
        "level", "group_id", "group_name", "p_original", "p_combined",
        "corrected_p_value", "q_work", "parent_gid", "parent_name", "rejected", "n_children"
    ]
    out = sel_all[[c for c in keep if c in sel_all.columns]]
    out.to_csv(csv_path, index=False)


def ascii_tree_from_csv(
    csv_path: str,
    out_path: Optional[str] = None,
    sort_by: str = "node_id",          # still supported; we already sorted by name+id in load_tree_index
    strip_parent_prefix: bool = True,
    min_ancestor_len_for_strip: int = 5,
) -> str:
    ti = load_tree_index(csv_path)

    lines: List[str] = []

    def render(nid: int, prefix: str = "", is_last: bool = True):
        connector = "└── " if is_last else "├── "
        raw_label = ti.name_map[nid]
        anc_names = ancestor_name_chain(nid, ti.parent_map, ti.name_map)
        label = strip_ancestors_prefix(raw_label, anc_names, min_ancestor_len_for_strip=min_ancestor_len_for_strip) \
                if strip_parent_prefix else raw_label

        # leaf?
        row = ti.nodes_df.loc[ti.nodes_df["node_id"] == nid].iloc[0]
        is_leaf = pd.notna(row["p_value"])
        if is_leaf:
            label += " (leaf)"

        lines.append(label if prefix == "" else prefix + connector + label)

        kids = ti.children.get(nid, [])
        if kids:
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, k in enumerate(kids):
                render(k, new_prefix, i == len(kids) - 1)

    for i, r in enumerate(ti.roots):
        render(r, "", i == len(ti.roots) - 1)

    sketch = "\n".join(lines)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(sketch + "\n")
    return sketch


if __name__ == "__main__":
    file_path = r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory"
    tree_file_path = os.path.join(file_path, "exploratory_p_values.csv")
    ascii_tree_from_csv(tree_file_path, out_path=os.path.join(file_path, "tree_sketch.txt"))

    paths = manage_treebh(
        all_save_path=r"C:\Users\Rony\Documents\projects\ethics\survey_analysis\data\analysis_data\all\exploratory",
        csv_path=tree_file_path,
        q=0.05,
        tag="",
        write_csvs=True,
        plot=True
    )
