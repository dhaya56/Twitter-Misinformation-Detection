"""
parse_raw.py
------------
Pure parsing of the rumdetect2017 raw format.
No feature computation. No normalization. No PyG objects.

Three public functions:
    parse_label_file(path)       -> dict {claim_id: label_str}
    parse_source_tweets(path)    -> dict {claim_id: tweet_text}
    parse_tree_file(path)        -> dict with keys:
                                      'claim_id'  : str
                                      'nodes'     : list of node dicts
                                      'edges'     : list of edge dicts

Node dict keys:
    node_idx   : int   (0-based, assigned in order of first appearance)
    user_id    : str   (original user_id string, 'ROOT' for root node)
    timestamp  : float (raw value from file, in minutes)
    is_root    : bool

Edge dict keys:
    src_idx    : int   (parent node_idx)
    dst_idx    : int   (child node_idx)
    src_time   : float (parent timestamp, after clamping)
    dst_time   : float (child timestamp, after clamping)

Design decisions (confirmed):
    - Index-based node identification: every edge occurrence creates a
      new node entry. Duplicate user_ids become separate nodes.
    - Monotonicity clamping: if child_time < parent_time,
      child_time is set to parent_time.
    - tweet_id field is ignored entirely (always equals root tweet id).
"""

import os
import re

# ── compiled regex for tree line parsing ─────────────────────────────────────
# Matches: ['uid', 'tid', 'time']->['uid', 'tid', 'time']
_EDGE_PATTERN = re.compile(
    r"\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]"
    r"\s*->\s*"
    r"\['([^']+)',\s*'([^']+)',\s*'([^']+)'\]"
)


# ── label parsing ─────────────────────────────────────────────────────────────

def parse_label_file(path):
    """
    Parse label.txt into a dict.

    File format (one entry per line):
        label:claim_id

    Returns:
        dict  {claim_id (str) -> label (str)}
        Labels are one of: 'non-rumor', 'true', 'false', 'unverified'

    Raises:
        FileNotFoundError if path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    labels = {}
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                skipped += 1
                continue
            # split on first colon only (label string never contains colon)
            colon_pos = line.index(":")
            label    = line[:colon_pos].strip()
            claim_id = line[colon_pos + 1:].strip()
            if not claim_id:
                skipped += 1
                continue
            labels[claim_id] = label

    if skipped:
        print(f"[parse_label_file] Skipped {skipped} malformed lines in {path}")

    return labels


# ── source tweet parsing ──────────────────────────────────────────────────────

def parse_source_tweets(path):
    """
    Parse source_tweets.txt into a dict.

    File format (one entry per line):
        claim_id<whitespace>tweet text

    Returns:
        dict  {claim_id (str) -> tweet_text (str)}
        tweet_text may be empty string if no text follows the claim_id.

    Raises:
        FileNotFoundError if path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source tweets file not found: {path}")

    sources = {}
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)   # split on first whitespace token
            claim_id = parts[0]
            text     = parts[1].strip() if len(parts) > 1 else ""
            sources[claim_id] = text

    return sources


# ── tree file parsing ─────────────────────────────────────────────────────────

def parse_tree_file(path):
    """
    Parse a single tree file into nodes and edges.

    File format (one edge per line):
        ['parent_uid', 'parent_tid', 'parent_time']->['child_uid', 'child_tid', 'child_time']

    Node identification strategy: index-based.
        - Each unique (user_id, occurrence_order) gets a new node index.
        - The ROOT node is always index 0.
        - When a user_id appears as a PARENT, we look up its existing node index
          (it must have appeared as a child in an earlier line, or be ROOT).
        - When a user_id appears as a CHILD, we always create a new node index.
        - This means duplicate child user_ids become separate nodes.

    Monotonicity clamping:
        - If child_time < parent_time, child_time is set to parent_time.

    Returns:
        dict with keys:
            'claim_id'         : str   (filename stem)
            'nodes'            : list of node dicts (sorted by node_idx)
            'edges'            : list of edge dicts
            'num_violations'   : int   (monotonicity violations found and clamped)
            'num_duplicates'   : int   (duplicate child user_ids encountered)

    Node dict:
        {
            'node_idx'  : int,
            'user_id'   : str,
            'timestamp' : float,   (after clamping)
            'is_root'   : bool,
        }

    Edge dict:
        {
            'src_idx'  : int,
            'dst_idx'  : int,
            'src_time' : float,
            'dst_time' : float,
        }

    Raises:
        FileNotFoundError if path does not exist.
        ValueError if file contains no parseable edges.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tree file not found: {path}")

    claim_id = os.path.splitext(os.path.basename(path))[0]

    # ── state ──────────────────────────────────────────────────────────────
    nodes        = []         # list of node dicts, ordered by node_idx
    edges        = []         # list of edge dicts

    # maps user_id -> node_idx for PARENT lookup only
    # ROOT and nodes that have appeared as children are registered here
    # when a user appears as parent, we find their idx here
    parent_index = {}         # {user_id: node_idx}  (last assigned idx for that uid)

    next_idx          = 0
    num_violations    = 0
    num_duplicates    = 0
    unparsed_lines    = 0

    def _register_root(uid, time):
        """Register the ROOT node at index 0."""
        nonlocal next_idx
        node = {
            "node_idx"  : 0,
            "user_id"   : uid,
            "timestamp" : time,
            "is_root"   : True,
        }
        nodes.append(node)
        parent_index[uid] = 0
        next_idx = 1

    def _register_child(uid, time):
        """
        Register a new child node with the next available index.
        Always creates a new node (index-based strategy).
        Returns the new node_idx.
        """
        nonlocal next_idx, num_duplicates
        if uid in parent_index:
            num_duplicates += 1
        idx = next_idx
        node = {
            "node_idx"  : idx,
            "user_id"   : uid,
            "timestamp" : time,
            "is_root"   : False,
        }
        nodes.append(node)
        # overwrite parent_index so this node's latest idx is used if
        # it later appears as a parent
        parent_index[uid] = idx
        next_idx += 1
        return idx

    # ── parse lines ────────────────────────────────────────────────────────
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = _EDGE_PATTERN.match(line)
            if not m:
                unparsed_lines += 1
                continue

            p_uid, _p_tid, p_time_str, c_uid, _c_tid, c_time_str = m.groups()

            try:
                p_time = float(p_time_str)
                c_time = float(c_time_str)
            except ValueError:
                unparsed_lines += 1
                continue

            # ── register parent if first line (ROOT) ──
            if p_uid == "ROOT" and "ROOT" not in parent_index:
                _register_root(p_uid, p_time)

            # ── get parent node index ──
            if p_uid not in parent_index:
                # parent appears before being registered as a child
                # this is a topological order violation — register now
                if p_uid == "ROOT":
                    _register_root(p_uid, p_time)
                else:
                    # register as a node without prior child record
                    _register_child(p_uid, p_time)

            src_idx  = parent_index[p_uid]
            src_time = nodes[src_idx]["timestamp"]

            # ── monotonicity clamping ──
            if c_time < src_time:
                c_time = src_time
                num_violations += 1

            # ── register child node (always new index) ──
            dst_idx = _register_child(c_uid, c_time)

            edges.append({
                "src_idx"  : src_idx,
                "dst_idx"  : dst_idx,
                "src_time" : src_time,
                "dst_time" : c_time,
            })

    if unparsed_lines > 0:
        print(f"[parse_tree_file] {claim_id}: {unparsed_lines} unparseable lines skipped")

    if len(edges) == 0:
        raise ValueError(f"No edges parsed from tree file: {path}")

    return {
        "claim_id"       : claim_id,
        "nodes"          : nodes,
        "edges"          : edges,
        "num_violations" : num_violations,
        "num_duplicates" : num_duplicates,
    }


# ── binary label mapping ──────────────────────────────────────────────────────

def map_label_to_binary(label_str):
    """
    Maps raw label string to binary integer.

    credible      (0): non-rumor, true
    misinformation(1): false, unverified

    Returns:
        int  0 or 1

    Raises:
        ValueError for unknown label strings.
    """
    credible = {"non-rumor", "true"}
    misinfo  = {"false", "unverified"}

    if label_str in credible:
        return 0
    if label_str in misinfo:
        return 1
    raise ValueError(f"Unknown label string: '{label_str}'")