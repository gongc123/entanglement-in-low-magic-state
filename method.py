
from __future__ import annotations
from dataclasses import dataclass
import random, cirq, math, itertools
import numpy as np
from typing import List, Tuple, Optional, Sequence, Dict, Iterable
from collections import defaultdict
from numpy.linalg import norm


def entropy(rho):
    ens = np.linalg.eigvals(rho)
    ens = [ele for ele in ens if ele> 0.000000000001]
    return round(float(sum([-ele*np.log2(ele) for ele in ens]).real),3)

def random_1q_clifford(rng: random.Random) -> cirq.Gate:
    # A simple generating set for single-qubit Cliffords.
    # You can expand this set if you want “more random”.
    return rng.choice([cirq.I, cirq.H, cirq.S, cirq.S**-1])


def random_local_2q_clifford(rng: random.Random, q1: cirq.Qid, q2: cirq.Qid) -> List[cirq.Operation]:
    """
    Return a small random nearest-neighbor 2q Clifford "entangler block".
    Using CZ/CNOT (and occasionally SWAP) keeps it Clifford and local.
    """
    choice = rng.randrange(4)
    if choice == 0:
        return []
    if choice == 1:
        return [cirq.CZ(q1, q2)]
    if choice == 2:
        # random CNOT direction
        return [cirq.CNOT(q1, q2)] if rng.random() < 0.5 else [cirq.CNOT(q2, q1)]
    # SWAP is also Clifford; optional
    if choice == 3:
        return [cirq.SWAP(q1, q2)]


def local_clifford_backbone_1d(n: int, d: int, seed: Optional[int] = None) -> Tuple[cirq.Circuit, List[cirq.LineQubit]]:
    """
    Build a local (nearest-neighbor) Clifford circuit of depth d (moments)
    on a 1D chain of n qubits.

    Pattern per moment:
      - Random 1q Cliffords on all qubits
      - Nearest-neighbor entanglers on either even bonds or odd bonds (alternating)
    """
    rng = random.Random(seed)
    qubits = list(cirq.LineQubit.range(n))

    moments: List[cirq.Moment] = []
    for t in range(d):
        ops: List[cirq.Operation] = []

        # 1q random Cliffords
        for q in qubits:
            g = random_1q_clifford(rng)
            if g != cirq.I:
                ops.append(g(q))

        moments.append(cirq.Moment(ops))

        # local entanglers on alternating bonds
        ops: List[cirq.Operation] = []
        start = 0 if (t % 2 == 0) else 1
        for i in range(start, n - 1, 2):
            ops.extend(random_local_2q_clifford(rng, qubits[i], qubits[i + 1]))

        moments.append(cirq.Moment(ops))

    return cirq.Circuit(moments), qubits

def random_1q_gate(rng: np.random.Generator | None = None) -> cirq.Gate:
    rng = rng or np.random.default_rng()
    U = cirq.testing.random_unitary(2, random_state=rng)  # Haar-random 2x2 unitary
    return cirq.MatrixGate(U)


def random_z_rotation_gate(seed) -> cirq.Gate:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2*np.pi)          # radians
    return cirq.rz(theta)                      # gate


def insert_exactly_k_T_gates(
    circuit: cirq.Circuit,
    qubits: List[cirq.Qid],
    k: int,
    other_angle = False,
    seed: Optional[int] = None,
    allow_multiple_T_same_spot: bool = False,
    allow_multiple_T_same_qubit_same_moment: bool = False,
) -> cirq.Circuit:
    """
    Insert exactly k single-qubit T gates into existing moments.

    By default:
      - no two T's share the same (moment, qubit) "spot"
      - and we also avoid adding a T if there's already any op on that qubit in that moment
        (to keep the circuit nicely layer-like).
    You can relax these with flags.
    """
    rng = random.Random(seed)
    depth = len(circuit)
    n = len(qubits)
    if depth == 0:
        raise ValueError("Circuit has zero moments.")

    # Precompute which (moment, qubit) are "free" if we want at most one op per qubit per moment
    existing = [[False] * n for _ in range(depth)]
    for mi, moment in enumerate(circuit):
        for op in moment.operations:
            for q in op.qubits:
                qi = qubits.index(q)
                existing[mi][qi] = True

    def spot_ok(mi: int, qi: int, chosen: set[Tuple[int, int]]) -> bool:
        if not allow_multiple_T_same_spot and (mi, qi) in chosen:
            return False
        if not allow_multiple_T_same_qubit_same_moment:
            # don't place T on a qubit that already has an op in that moment
            if existing[mi][qi]:
                return False
        return True

    # If we disallow collisions and also disallow placing on occupied qubits, count capacity
    if not allow_multiple_T_same_qubit_same_moment:
        capacity = sum(1 for mi in range(depth) for qi in range(n) if not existing[mi][qi])
        if k > capacity:
            raise ValueError(
                f"k={k} is too large. Only {capacity} (moment,qubit) slots are free "
                "under your placement constraints."
            )
    elif not allow_multiple_T_same_spot:
        capacity = depth * n
        if k > capacity:
            raise ValueError(f"k={k} is too large for unique (moment,qubit) placement: capacity={capacity}.")

    chosen: set[Tuple[int, int]] = set()
    while len(chosen) < k:
        mi = rng.randrange(depth)
        qi = rng.randrange(n)
        if spot_ok(mi, qi, chosen):
            chosen.add((mi, qi))

    new_moments: List[cirq.Moment] = []
    for mi, moment in enumerate(circuit):
        ops = list(moment.operations)
        for (m, qi) in chosen:
            if m == mi:
                if other_angle == 1:
                    cirq_U = random_z_rotation_gate(seed+mi+qi)
                else:
                    cirq_U = cirq.T
                ops.append(cirq_U(qubits[qi]))
        new_moments.append(cirq.Moment(ops))

    return cirq.Circuit(new_moments)


def random_local_1d_clifford_plus_kT(n: int, d: int, k: int, seed: Optional[int] = None) -> cirq.Circuit:
    backbone, qubits = local_clifford_backbone_1d(n, d, seed=seed)
    # separate RNG stream for T insertion
    return insert_exactly_k_T_gates(
        backbone,
        qubits,
        k,
        seed=None if seed is None else seed + 99991,
        allow_multiple_T_same_spot=False,
        allow_multiple_T_same_qubit_same_moment=False,
    )




@dataclass
class Segment:
    ops: List[cirq.Operation]
    boundary_nonclifford: Optional[cirq.Operation] = None


def split_into_clifford_segments(circuit: cirq.Circuit) -> List[Segment]:
    """
    Split into maximal stabilizer-effect (Clifford) segments separated by non-stabilizer ops.

    We reject measurements here because "push Pauli through circuit" is no longer a simple
    unitary conjugation in the presence of measurements / resets / classical control.
    """
    if cirq.is_measurement(circuit):
        raise ValueError("Circuit contains measurement(s). This script assumes a unitary circuit.")

    segments: List[Segment] = []
    cur_ops: List[cirq.Operation] = []

    for op in circuit.all_operations():
        if cirq.is_measurement(op):
            raise ValueError("Found a measurement op. This script assumes a unitary circuit.")

        if cirq.has_stabilizer_effect(op):
            cur_ops.append(op)
        else:
            # close current Clifford segment, record boundary non-Clifford
            segments.append(Segment(ops=cur_ops, boundary_nonclifford=op))
            cur_ops = []

    # final segment
    segments.append(Segment(ops=cur_ops, boundary_nonclifford=None))
    return segments


# ----------------------------
# 2) Stabilizers for |0>^n
# ----------------------------

def initial_zero_state_stabilizers(qubits: Sequence[cirq.Qid]) -> List[cirq.PauliString]:
    # Generators are Z(q) for each qubit q
    return [cirq.PauliString({q: cirq.Z}) for q in qubits]


# ----------------------------
# 3) Conjugate Paulis through a Clifford segment
# ----------------------------

def make_clifford_op_from_ops(ops: List[cirq.Operation], qubits: Sequence[cirq.Qid]) -> Optional[cirq.Operation]:
    """
    Build a single n-qubit Clifford operation equivalent to applying `ops` in order.
    Returns None if ops is empty.
    """
    if not ops:
        return None

    # cirq.CliffordGate.from_op_list expects ops in the order applied to the state
    gate = cirq.CliffordGate.from_op_list(list(ops), qubit_order=list(qubits))
    return gate.on(*qubits)


def propagate_stabilizers_through_clifford(
    stabs: List[cirq.PauliString],
    clifford_op: Optional[cirq.Operation],
) -> List[cirq.PauliString]:
    """
    If the state is updated by U, stabilizers update as:  S -> U S U^dagger.
    Cirq's PauliString.conjugated_by(C) returns C^dagger P C, so we pass C = U^dagger,
    i.e. conjugate_by(inverse(U)).

    Reference: PauliString.conjugated_by definition. :contentReference[oaicite:4]{index=4}
    """
    if clifford_op is None:
        return stabs

    u_dag = cirq.inverse(clifford_op)   # this is U^\dagger
    return [p.conjugated_by(u_dag) for p in stabs]


# ----------------------------
# 4) GF(2) elimination on (x|z) with ±1 signs
# ----------------------------

def paulistring_to_xz_sign(ps: cirq.PauliString, qubits: Sequence[cirq.Qid]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Convert a PauliString into binary (x,z) vectors and a sign bit for coefficient ±1.

    x_q = 1 if Pauli on q has X component (X or Y)
    z_q = 1 if Pauli on q has Z component (Z or Y)

    Coefficient must be ±1; if you see ±i, it’s not a valid stabilizer generator.
    """
    coef = complex(ps.coefficient)
    if not (np.isclose(coef, 1.0) or np.isclose(coef, -1.0)):
        raise ValueError(f"PauliString coefficient {ps.coefficient} is not ±1; not a stabilizer generator.")

    sign = 0 if np.isclose(coef, 1.0) else 1

    x = np.zeros(len(qubits), dtype=np.uint8)
    z = np.zeros(len(qubits), dtype=np.uint8)

    for i, q in enumerate(qubits):
        p = ps.get(q, None)
        if p is None:
            continue
        if p == cirq.X:
            x[i] = 1
        elif p == cirq.Z:
            z[i] = 1
        elif p == cirq.Y:
            x[i] = 1
            z[i] = 1
        else:
            raise ValueError(f"Unexpected single-qubit Pauli {p} on qubit {q}.")

    return x, z, sign


def xz_sign_to_paulistring(
    xz: np.ndarray,
    sign: int,
    qubits: Sequence[cirq.Qid],
) -> cirq.PauliString:
    d: Dict[cirq.Qid, cirq.Pauli] = {}
    n = len(qubits)
    x = xz[:n] & 1
    z = xz[n:] & 1
    for i, q in enumerate(qubits):
        xi = int(x[i])
        zi = int(z[i])
        if xi == 0 and zi == 0:
            continue
        if xi == 1 and zi == 0:
            d[q] = cirq.X
        elif xi == 0 and zi == 1:
            d[q] = cirq.Z
        else:
            d[q] = cirq.Y

    coef = -1 if (sign % 2 == 1) else 1
    return cirq.PauliString(d, coefficient=coef)


# @dataclass
# class ReducedStabilizers:
#     generators: List[cirq.PauliString]
#     rank: int
#     pivots: List[int]



def gf2_row_reduce_stabilizers(stabs: List[cirq.PauliString], qubits: Sequence[cirq.Qid]):
    # Minimal version: keep an independent set using elimination on (x|z);
    # row ops done by Pauli multiplication to keep sign correct.
    if not stabs:
        return stabs

    n = len(qubits)
    rows = list(stabs)
    m = len(rows)
    A = np.zeros((m, 2 * n), dtype=np.uint8)
    for i, ps in enumerate(rows):
        x, z, _ = paulistring_to_xz_sign(ps, qubits)
        A[i, :n] = x
        A[i, n:] = z

    r = 0
    for c in range(2 * n):
        piv = None
        for i in range(r, m):
            if A[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv], :] = A[[piv, r], :]
            rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] ^= A[r, :]
                rows[i] = rows[i] * rows[r]
                _ = paulistring_to_xz_sign(rows[i], qubits)  # sanity (±1)
        r += 1
        if r == m:
            break
    return rows[:r]

def gf2_nullspace_basis(M: np.ndarray) -> List[np.ndarray]:
    """
    Return a basis for the nullspace of M over GF(2).
    M has shape (p, r); we solve M v = 0 for v in GF(2)^r.
    Output: list of basis vectors v (each shape (r,), dtype uint8).
    """
    M = (M.copy() % 2).astype(np.uint8)
    p, r = M.shape
    A = M

    pivot_cols = []
    row = 0

    # Row-reduce A to RREF-ish form
    for col in range(r):
        # find pivot row
        piv = None
        for i in range(row, p):
            if A[i, col] == 1:
                piv = i
                break
        if piv is None:
            continue

        # swap
        if piv != row:
            A[[row, piv], :] = A[[piv, row], :]

        # eliminate all other 1s in this column
        for i in range(p):
            if i != row and A[i, col] == 1:
                A[i, :] ^= A[row, :]

        pivot_cols.append(col)
        row += 1
        if row == p:
            break

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(r) if c not in pivot_set]

    basis = []
    # For each free variable, build a nullspace vector
    for f in free_cols:
        v = np.zeros(r, dtype=np.uint8)
        v[f] = 1

        # Solve pivot variables from rows: pivot = sum(free * coeffs)
        # In our eliminated matrix, each pivot row corresponds to one pivot col.
        for i, pc in enumerate(pivot_cols):
            # Row i: x_pc + sum_{j>pc} A[i,j] x_j = 0  => x_pc = sum A[i,j] x_j
            # We can just compute dot(row, v) excluding pivot itself since v[pc]=0 initially.
            s = 0
            # XOR over columns where A[i, j]=1 and v[j]=1
            # (including free cols; pivot cols other than pc should be 0 already in this construction)
            for j in range(r):
                if j != pc and A[i, j] == 1 and v[j] == 1:
                    s ^= 1
            v[pc] = s

        basis.append(v)

    # If there are no free vars, nullspace is {0}
    return basis


def stabilizers_trivial_on_qubits(
    stabs: List[cirq.PauliString],
    qubits: Sequence[cirq.Qid],
    blocked: Sequence[cirq.Qid],
) -> List[cirq.PauliString]:
    """
    Return a generating set for the subgroup of <stabs> that acts trivially on `blocked` qubits.
    This includes products of generators, not just those individually disjoint from blocked.
    """
    stabs = gf2_row_reduce_stabilizers(stabs, qubits)
    if not stabs:
        return []

    n = len(qubits)
    r = len(stabs)
    blocked_set = set(blocked)
    blocked_indices = [i for i, q in enumerate(qubits) if q in blocked_set]
    if not blocked_indices:
        return stabs  # nothing to block

    # Build G_Q: r x (2*|blocked|)
    GQ = np.zeros((r, 2 * len(blocked_indices)), dtype=np.uint8)
    for i, ps in enumerate(stabs):
        x, z, _ = paulistring_to_xz_sign(ps, qubits)
        # pack x and z on blocked qubits
        xb = np.array([x[j] for j in blocked_indices], dtype=np.uint8)
        zb = np.array([z[j] for j in blocked_indices], dtype=np.uint8)
        GQ[i, :len(blocked_indices)] = xb
        GQ[i, len(blocked_indices):] = zb

    # Solve v^T GQ = 0  <=> (GQ^T) v = 0
    M = GQ.T  # shape (2|blocked|, r)
    basis_vs = gf2_nullspace_basis(M)

    survivors = []
    for v in basis_vs:
        # build product of generators with v_i=1
        acc = cirq.PauliString(coefficient=1)
        for i in range(r):
            if v[i] == 1:
                acc = acc * stabs[i]
        # drop identity
        if len(acc) == 0:  # no qubits in dict => identity (up to coefficient)
            continue
        # sanity check: trivial on blocked
        for qb in blocked:
            if acc.get(qb, None) is not None:
                raise RuntimeError("Bug: constructed survivor still acts on blocked qubit.")
        survivors.append(acc)

    # Reduce again for independence
    survivors = gf2_row_reduce_stabilizers(survivors, qubits)
    return survivors

# ----------------------------
# 5) Driver
# ----------------------------

def stabilizers_through_clifford_plus_T(circuit: cirq.Circuit, qubits: Optional[Sequence[cirq.Qid]] = None, print_stabs = 0) -> None:
    if qubits is None:
        qubits = sorted(circuit.all_qubits())

    segments = split_into_clifford_segments(circuit)
    stabs = initial_zero_state_stabilizers(qubits)
    stabs = gf2_row_reduce_stabilizers(stabs, qubits)
    if print_stabs==1:
        print(f"Input stabilizers (|0>^{{{len(qubits)}}}):")
        for g in stabs:
            print("  ", g)
        print()

    for k, seg in enumerate(segments):
        cliff_op = make_clifford_op_from_ops(seg.ops, qubits)
        stabs = propagate_stabilizers_through_clifford(stabs, cliff_op)
        stabs = gf2_row_reduce_stabilizers(stabs, qubits)
        if print_stabs==1:
            print(f"After Clifford segment {k} (ops={len(seg.ops)}): rank={len(stabs)}")
            for g in stabs:
                print("  ", g)
            print()
       
        if seg.boundary_nonclifford is not None:
            # print("Encountered non-Clifford op:", seg.boundary_nonclifford)
            blocked_qu = seg.boundary_nonclifford.qubits
            stabs = stabilizers_trivial_on_qubits(stabs, qubits, blocked_qu)
            stabs = gf2_row_reduce_stabilizers(stabs, qubits)
            if print_stabs==1:
                print(f"After non-Clifford {k} at {blocked_qu}: rank={len(stabs)}")
                for g in stabs:
                    print("  ", g)
                print()

    return stabs


# -----------------------------
# GF(2) linear algebra helpers
# -----------------------------

def gf2_rref(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Reduced row echelon form over GF(2).
    Returns (RREF matrix copy, pivot_col_indices).
    """
    A = (M.copy() & 1).astype(np.uint8)
    r, c = 0, 0
    pivots: List[int] = []
    nrows, ncols = A.shape

    while r < nrows and c < ncols:
        # find pivot
        pivot = None
        for rr in range(r, nrows):
            if A[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            c += 1
            continue

        # swap into place
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]

        # eliminate other rows
        for rr in range(nrows):
            if rr != r and A[rr, c] == 1:
                A[rr] ^= A[r]

        pivots.append(c)
        r += 1
        c += 1

    return A, pivots


def gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """
    Returns a basis for the nullspace of M over GF(2) as a matrix whose rows are basis vectors.
    Solves M x = 0.
    """
    M = (M & 1).astype(np.uint8)
    nrows, ncols = M.shape
    R, pivots = gf2_rref(M)

    pivot_set = set(pivots)
    free_cols = [j for j in range(ncols) if j not in pivot_set]
    if not free_cols:
        return np.zeros((0, ncols), dtype=np.uint8)

    # Build one basis vector per free variable
    basis = []
    # Map pivot col -> pivot row index in RREF (since we eliminated all other rows)
    # In our gf2_rref, pivot in row i at col pivots[i] (in increasing order).
    for free in free_cols:
        v = np.zeros(ncols, dtype=np.uint8)
        v[free] = 1
        # back-substitute: for each pivot row i, x[pivot_col] = sum(R[i, free_cols]*x[free_cols])
        for i, pcol in enumerate(pivots):
            # In RREF, row i has 1 at pcol, and other pivot cols are 0.
            # Constraint row: x[pcol] + sum_{j free} R[i,j] x[j] = 0
            # So x[pcol] = sum_{j free} R[i,j] x[j]
            if R[i, free] == 1:
                v[pcol] ^= 1
        basis.append(v)

    return np.stack(basis, axis=0).astype(np.uint8)


def gf2_rowspace_basis(rows: np.ndarray) -> np.ndarray:
    """
    Returns an independent basis (rows) spanning the same rowspace over GF(2).
    """
    if rows.size == 0:
        return rows.astype(np.uint8)
    R, pivots = gf2_rref(rows)
    # Nonzero rows in R are a basis
    nonzero = [i for i in range(R.shape[0]) if R[i].any()]
    return R[nonzero].astype(np.uint8)


def gf2_extend_basis(base: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """
    Starting from a row-basis 'base', add as many vectors from 'vecs' as possible that increase rank.
    Returns the extended independent basis as rows.
    """
    base = gf2_rowspace_basis(base)
    for v in vecs:
        trial = np.vstack([base, v[None, :]]) if base.size else v[None, :]
        new_base = gf2_rowspace_basis(trial)
        if new_base.shape[0] > (base.shape[0] if base.size else 0):
            base = new_base
    return base


def gf2_quotient_basis(V_basis: np.ndarray, W_basis: np.ndarray) -> np.ndarray:
    """
    Given row-bases for subspaces W ⊆ V (both in GF(2)^m),
    return a set of representatives forming a basis for the quotient V / W.
    """
    Vb = gf2_rowspace_basis(V_basis)
    Wb = gf2_rowspace_basis(W_basis)

    reps = []
    span = Wb
    for v in Vb:
        before_rank = gf2_rowspace_basis(span).shape[0] if span.size else 0
        trial = np.vstack([span, v[None, :]]) if span.size else v[None, :]
        after = gf2_rowspace_basis(trial)
        after_rank = after.shape[0]
        if after_rank > before_rank:
            reps.append(v.copy())
            span = after  # grow the span including this representative

    if not reps:
        return np.zeros((0, Vb.shape[1]), dtype=np.uint8)
    return np.stack(reps, axis=0).astype(np.uint8)

# -----------------------------
# Pauli <-> symplectic vectors
# -----------------------------

def paulistring_to_xz(
    ps: cirq.PauliString,
    qubits: Sequence[cirq.Qid],
    qindex: Dict[cirq.Qid, int],
) -> np.ndarray:
    """
    Convert a PauliString to a length-2n GF(2) vector [x|z].
    Ignore global phase.
    """
    n = len(qubits)
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)

    for q, op in ps.items():
        i = qindex[q]
        if op == cirq.X:
            x[i] ^= 1
        elif op == cirq.Z:
            z[i] ^= 1
        elif op == cirq.Y:
            x[i] ^= 1
            z[i] ^= 1
        else:
            raise ValueError(f"Unsupported op {op} on qubit {q}")
    return np.concatenate([x, z]).astype(np.uint8)


def xz_to_paulistring(
    xz: np.ndarray,
    qubits: Sequence[cirq.Qid],
) -> cirq.PauliString:
    """
    Convert a length-2n vector [x|z] to a Cirq PauliString (phase-free).
    """
    n = len(qubits)
    x = xz[:n] & 1
    z = xz[n:] & 1
    ops = {}
    for i, q in enumerate(qubits):
        xi, zi = int(x[i]), int(z[i])
        if xi == 0 and zi == 0:
            continue
        if xi == 1 and zi == 0:
            ops[q] = cirq.X
        elif xi == 0 and zi == 1:
            ops[q] = cirq.Z
        else:
            ops[q] = cirq.Y
    return cirq.PauliString(ops)


def symplectic_commutes(xz1: np.ndarray, xz2: np.ndarray) -> int:
    """
    Returns 1 if anticommute, 0 if commute, using symplectic inner product.
    For [x|z], [x'|z']:  <v,w> = x·z' + z·x' mod 2.
    """
    n2 = xz1.size
    n = n2 // 2
    x1, z1 = xz1[:n], xz1[n:]
    x2, z2 = xz2[:n], xz2[n:]
    return int((np.dot(x1, z2) + np.dot(z1, x2)) & 1)

# -----------------------------
# Main routine
# -----------------------------

def logical_operators_supported_on_A(
    stabilizer_gens: Sequence[cirq.PauliString],
    qubits: Sequence[cirq.Qid],
    A: Iterable[cirq.Qid],
) -> List[cirq.PauliString]:
    """
    Find a basis of Pauli operators supported entirely on A that commute with the stabilizer group,
    modulo the stabilizer group.

    Inputs:
      - stabilizer_gens: list of independent stabilizer generators as cirq.PauliString
      - qubits: ordered list of all n qubits (defines vectorization order)
      - A: iterable of qubits defining subregion A

    Output:
      - list of cirq.PauliString basis elements (phase-free), all supported within A.
    """
    n = len(qubits)
    qindex = {q: i for i, q in enumerate(qubits)}
    Aset: Set[cirq.Qid] = set(A)
    A_idx = [qindex[q] for q in qubits if q in Aset]
    notA_idx = [i for i in range(n) if i not in set(A_idx)]

    # Stabilizer generator matrix S (k x 2n)
    S = np.stack([paulistring_to_xz(g, qubits, qindex) for g in stabilizer_gens], axis=0).astype(np.uint8)
    k = S.shape[0]

    # Build commutation constraints M x = 0 for x in GF(2)^(2n):
    # For each stabilizer s=[xs|zs], require x·zs + z·xs = 0.
    # This is linear: [zs | xs] · [x|z] = 0 mod 2.
    M = np.zeros((k, 2 * n), dtype=np.uint8)
    for i in range(k):
        xs = S[i, :n]
        zs = S[i, n:]
        M[i, :n] = zs
        M[i, n:] = xs

    # Add support constraints: outside A, x_i = 0 and z_i = 0
    # Each is one linear equation selecting that coordinate.
    support_rows = []
    for i in notA_idx:
        row_x = np.zeros(2 * n, dtype=np.uint8)
        row_x[i] = 1
        support_rows.append(row_x)
        row_z = np.zeros(2 * n, dtype=np.uint8)
        row_z[n + i] = 1
        support_rows.append(row_z)

    if support_rows:
        C = np.stack(support_rows, axis=0).astype(np.uint8)
        constraints = np.vstack([M, C]).astype(np.uint8)
    else:
        constraints = M

    # V = { v : constraints v = 0 } = centralizer elements supported on A
    V_basis = gf2_nullspace(constraints)  # rows are basis vectors in GF(2)^(2n)

    # W = stabilizer span (rowspace of S)
    W_basis = gf2_rowspace_basis(S)

    # Quotient: V / (V ∩ W) is implemented by picking reps of V modulo W
    reps = gf2_quotient_basis(V_basis, W_basis)

    # Convert reps to PauliStrings; they are guaranteed to be supported on A by construction
    return [xz_to_paulistring(v, qubits) for v in reps]



def symp_ip(v: np.ndarray, w: np.ndarray) -> int:
    """Symplectic inner product over GF(2): 1 => anticommute, 0 => commute."""
    v = v & 1
    w = w & 1
    n = v.size // 2
    x, z = v[:n], v[n:]
    xp, zp = w[:n], w[n:]
    return int((np.dot(x, zp) + np.dot(z, xp)) & 1)

def gf2_add(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Addition in GF(2)^m = XOR."""
    return (v ^ w).astype(np.uint8)

def symplectic_canonicalize(
    gens: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Input:
      gens: (m, 2n) binary matrix; each row is a Pauli vector [x|z] (phase ignored).

    Output:
      pairs_P, pairs_Q, singlets
      - pairs_P[i] anticommutes with pairs_Q[i]
      - everything else commutes across different pairs and with singlets
      - each output element lies in the span of the original gens
      - span is preserved (same generated subgroup, ignoring phases)
    """
    G = (gens.copy() & 1).astype(np.uint8)
    m, dim = G.shape

    used = np.zeros(m, dtype=bool)

    P: List[np.ndarray] = []
    Q: List[np.ndarray] = []
    S: List[np.ndarray] = []

    # We will build a new generating set in-place by taking linear combinations of original rows.
    # Strategy:
    # 1) pick a vector a
    # 2) find b with <a,b>=1 -> make a pair (a,b)
    # 3) for every remaining vector v, "clean" it so it commutes with a and b:
    #       if <v,b>=1: v <- v + a
    #       if <v,a>=1: v <- v + b
    #    (this preserves span)
    #
    # Repeat; if no partner exists for a, it becomes a singlet after also cleaning it
    # with respect to existing pairs (done implicitly by the same cleaning loop).

    # Remaining list of vectors (we will keep them as a list for easier pop/modify)
    remaining = [G[i].copy() for i in range(m) if G[i].any()]

    def clean_against_pair(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Make v commute with a and b
        if symp_ip(v, b) == 1:
            v = gf2_add(v, a)
        if symp_ip(v, a) == 1:
            v = gf2_add(v, b)
        return v

    # First, ensure "remaining" vectors are linearly independent-ish is not required,
    # but removing exact zeros as we go helps.
    def strip_zeros(vecs: List[np.ndarray]) -> List[np.ndarray]:
        return [v for v in vecs if v.any()]

    while remaining:
        a = remaining.pop(0)

        # Find partner b with <a,b>=1
        partner_idx = None
        for j, cand in enumerate(remaining):
            if symp_ip(a, cand) == 1:
                partner_idx = j
                break

        if partner_idx is None:
            # a commutes with everything in remaining span -> singlet candidate.
            # Still, to guarantee it commutes with existing pairs, it should have been cleaned already
            # (it is, because we always clean the pool after forming each pair).
            S.append(a)
            continue

        b = remaining.pop(partner_idx)

        # We now have a pair (a,b). Clean all other vectors so they commute with both.
        new_remaining = []
        for v in remaining:
            v2 = clean_against_pair(v, a, b)
            new_remaining.append(v2)
        remaining = strip_zeros(new_remaining)

        P.append(a)
        Q.append(b)

    # At this point, due to the cleaning steps, all different pairs commute with each other,
    # and all singlets commute with all pairs and other singlets.
    return P, Q, S


def canonicalize_cirq_generators(
    gens: Sequence[cirq.PauliString],
    qubits: Sequence[cirq.Qid],
) -> Tuple[List[Tuple[cirq.PauliString, cirq.PauliString]], List[cirq.PauliString]]:
    qindex = {q: i for i, q in enumerate(qubits)}
    mat = np.stack([paulistring_to_xz(g, qubits, qindex) for g in gens], axis=0).astype(np.uint8)
    P, Q, S = symplectic_canonicalize(mat)
    pairs = [(xz_to_paulistring(p, qubits), xz_to_paulistring(q, qubits)) for p, q in zip(P, Q)]
    singlets = [xz_to_paulistring(s, qubits) for s in S]
    terms = pairs.copy()
    terms.extend([tuple([ele]) for ele in singlets])
    return pairs, singlets


def _is_zpow_magic(op: cirq.Operation) -> Optional[float]:
    """
    If op is a single-qubit ZPowGate, return its rotation angle phi in Rz(phi),
    where U = exp(-i phi Z/2) up to global phase.
    For cirq.ZPowGate(exponent=t), phi = pi * t.
    """
    g = op.gate
    if isinstance(g, cirq.ZPowGate) and len(op.qubits) == 1:
        return math.pi * float(g.exponent)
    return None


def _conjugate_paulistring_by_rz_dag(
    ps: cirq.PauliString, q: cirq.Qid, phi: float
) -> Dict[cirq.PauliString, complex]:
    """
    Compute Rz(phi)† * ps * Rz(phi) but only touching qubit q.
    This is the form needed for op† P op with op=Rz(phi).

    Action on Paulis at q:
      I -> I
      Z -> Z
      X -> cos(phi) X - sin(phi) Y
      Y -> sin(phi) X + cos(phi) Y

    Returns a dict {PauliString: coefficient_multiplier} for the branching result.
    Coefficients multiply the *whole* PauliString (including its current coefficient).
    """
    p = ps.get(q, None)
    if p is None or p == cirq.Z:
        return {ps: 1.0}

    # remove q from the base, we'll reinsert X/Y at q
    base_map = dict(ps)
    base_map.pop(q, None)
    base = cirq.PauliString(base_map, coefficient=ps.coefficient)

    c = math.cos(phi)
    s = math.sin(phi)

    out: Dict[cirq.PauliString, complex] = {}

    if p == cirq.X:
        # X -> c X - s Y
        if abs(c) > 0:
            out[base * cirq.PauliString({q: cirq.X})] = c
        if abs(s) > 0:
            out[base * cirq.PauliString({q: cirq.Y})] = -s
        return out

    if p == cirq.Y:
        # Y -> s X + c Y
        if abs(s) > 0:
            out[base * cirq.PauliString({q: cirq.X})] = s
        if abs(c) > 0:
            out[base * cirq.PauliString({q: cirq.Y})] = c
        return out

    raise ValueError(f"Unexpected Pauli {p} on qubit {q}")



def push_paulistring_clifford_plus_few_magic(
    circuit: cirq.Circuit,
    pauli: cirq.PauliString,
    *,
    magic_tolerance: float = 1e-12,
) -> complex:
    """
    Evaluate <0| U† pauli U |0> for a circuit that is Clifford plus a few
    injected single-qubit Z-rotations (e.g. T, T†, other ZPow).

    Works by pushing the Pauli operator through the circuit (Heisenberg picture),
    branching only at magic Z rotations when the operator has X/Y on that qubit.
    """
    # Operator sum represented as {PauliString: coefficient}
    # We keep PauliString objects with coefficient=1, store coefficient separately
    terms: Dict[cirq.PauliString, complex] = {cirq.PauliString(dict(pauli), coefficient=1): complex(pauli.coefficient)}

    for op in reversed(list(circuit.all_operations())):
        phi = _is_zpow_magic(op)

        if phi is None or cirq.has_stabilizer_effect(op):
            # Treat as Clifford (or stabilizer-effect) op: conjugation maps PauliString -> PauliString
            new_terms: Dict[cirq.PauliString, complex] = defaultdict(complex)
            for ps, coeff in terms.items():
                # want op† ps op, and PauliString.conjugated_by(op) gives op† ps op
                ps2 = ps.conjugated_by(op)
                # ps2 may carry a coefficient (phase) in its .coefficient; fold it into coeff
                coeff2 = coeff * complex(ps2.coefficient)
                ps2_nophase = cirq.PauliString(dict(ps2), coefficient=1)
                new_terms[ps2_nophase] += coeff2
            terms = new_terms
            continue

        # Non-Clifford ZPow on one qubit: branch only if X/Y present
        (q,) = op.qubits
        new_terms = defaultdict(complex)
        for ps, coeff in terms.items():
            branches = _conjugate_paulistring_by_rz_dag(ps, q, phi)
            for ps_b, mult in branches.items():
                # ps_b may have coefficient from base (we always keep coefficient=1 outside)
                coeff_b = coeff * complex(ps_b.coefficient) * complex(mult)
                ps_b_nophase = cirq.PauliString(dict(ps_b), coefficient=1)
                new_terms[ps_b_nophase] += coeff_b

        # optional: prune tiny coefficients
        terms = {ps: c for ps, c in new_terms.items() if abs(c) > magic_tolerance}

    return terms
    

def multiply_pauli_op_dicts(A, B):
    """
    Multiply two operators A and B represented as dicts {cirq.PauliString: coeff}.
    Returns dict in the same form, with canonicalized PauliString keys (coefficient=1).
    """
    out = {}

    for ps_a, a in A.items():
        if a == 0:
            continue
        for ps_b, b in B.items():
            if b == 0:
                continue

            prod = ps_a * ps_b  # Cirq handles Pauli multiplication + phase
            ps = cirq.ops.pauli_string.PauliString(dict(prod.items()))
            fac = a*b*prod.coefficient
            if ps in out:
                out[ps] += fac
            else:
                out[ps] = fac

    return out
    
def _expectation_on_zero(ps: cirq.PauliString) -> complex:
    """
    <0^n| ps |0^n>.
    Nonzero iff ps contains only I/Z on every qubit.
    In that case it's just ps.coefficient (since Z|0>=|0>).
    """
    for _, p in dict(ps).items():
        if p == cirq.X or p == cirq.Y:
            return 0.0
    return complex(ps.coefficient)


def expectation_from_terms(terms):
    # Final expectation on |0^n>
    expval = 0.0 + 0.0j
    for ps, coeff in terms.items():
        expval += coeff * _expectation_on_zero(ps)
    return expval


def all_subsets(input_list):
    """Generates all possible subsets of a given list."""
    s = list(input_list)
    return [
        list(subset)
        for r in range(len(s) + 1)
        for subset in itertools.combinations(s, r)
    ]


def logical_tomography(circuit, pairs, singlets):
    """
    Return the recovered logical state by logical operator tomography
    """
    l1 = len(pairs)
    l2 = len(singlets)
    N = 2*l1+l2
    bulkq = cirq.LineQubit.range(l1+l2)
    P_list = [cirq.PauliString({bulkq[k]:cirq.X}) for k in range(l1)]
    P_list.extend([cirq.PauliString({bulkq[k]:cirq.Z}) for k in range(l1+l2)]) 
    P_rep_list = [pairs[k][0] for k in range(l1)]
    P_rep_list.extend([pairs[k][1] for k in range(l1)])
    P_rep_list.extend([singlets[k] for k in range(l2)])
    pushed_terms_list = [push_paulistring_clifford_plus_few_magic(circuit, P_rep) for P_rep in P_rep_list]
    rho = cirq.PauliString({bulkq[0]:cirq.I})
    for r in range(1,N+1):
        for sub in itertools.combinations(range(N), r):
            P = cirq.PauliString({bulkq[0]:cirq.I})
            # P_rep=cirq.PauliString({bulkq[0]:cirq.I})
            term = {1:1}      # initialized to identity operator
            for ind in sub:
                P = P*P_list[ind]
                term = multiply_pauli_op_dicts(term, pushed_terms_list[ind])
                # P_rep = P_rep*P_rep_list[ind]
            rho += expectation_from_terms(term)*P
            # exp=expectation_from_terms(push_paulistring_clifford_plus_few_magic(circuit, P_rep))
            # rho += exp*P
            # print(expectation_from_terms(term),exp)
    rho = rho/2**(l1+l2)
    return rho.matrix(bulkq)




def stabilizers_supported_on_A(
    stabilizer_gens: Sequence[cirq.PauliString],
    qubits: Sequence[cirq.Qid],
    A: Iterable[cirq.Qid],
) -> List[cirq.PauliString]:
    """
    Given stabilizer generators, return generators of the subgroup consisting of
    stabilizer elements supported only on subregion A (i.e., identity on complement).

    Output is an independent generating set (phase-free representatives).
    """
    n = len(qubits)
    qindex = {q: i for i, q in enumerate(qubits)}
    Aset = set(A)

    # Build stabilizer generator matrix S: (k x 2n)
    S = np.stack([paulistring_to_xz(g, qubits, qindex) for g in stabilizer_gens], axis=0).astype(np.uint8)
    k = S.shape[0]

    # Indices of complement qubits (outside A)
    notA = [i for i, q in enumerate(qubits) if q not in Aset]

    # If A is the whole system, subgroup is the whole stabilizer group
    if len(notA) == 0:
        basis = gf2_rowspace_basis(S)
        return [xz_to_paulistring(v, qubits) for v in basis]

    # Extract columns corresponding to complement support (both X and Z bits)
    cols = notA + [n + i for i in notA]  # x bits outside A, then z bits outside A
    S_out = S[:, cols]  # shape (k, 2|notA|)

    # We want y^T S_out = 0  <=>  S_out^T y = 0
    # Solve for y in GF(2)^k
    Y_basis = gf2_nullspace(S_out.T)  # rows are y vectors (length k)

    if Y_basis.shape[0] == 0:
        return []  # only identity has support in A

    # Map each y to an operator: s = y^T S (mod 2)
    # Y_basis is (r x k); compute (r x 2n)
    SA_candidates = (Y_basis @ S) & 1

    # Reduce to an independent generating set
    SA_basis = gf2_rowspace_basis(SA_candidates)

    # Convert to Cirq PauliStrings
    return [xz_to_paulistring(v, qubits) for v in SA_basis]


def _coef_to_signbit(coef: complex) -> int:
    """
    Accept only overall ±1 coefficients.
    Returns 0 for +1, 1 for -1.
    """
    # Cirq stores coefficient as complex; allow tiny numerical noise.
    if abs(coef - 1) < 1e-9:
        return 0
    if abs(coef + 1) < 1e-9:
        return 1
    raise ValueError(f"Only ±1 coefficients supported, got {coef}.")


def stabilizer_intersect_phaseblind_group(
    stabilizer_gens: Sequence[cirq.PauliString],
    G_gens: Sequence[cirq.PauliString],
    qubits: Sequence[cirq.Qid],
) -> List[cirq.PauliString]:
    """
    Compute generators for intersection:
        Stabilizer group (with stabilizer-consistent signs)
    intersect
        group generated by G, but allowing a ± sign flip (phase-blind on G side).

    More precisely: return generators of the subgroup of stabilizers whose
    phase-free Pauli [x|z] lies in span([x|z] of G_gens).

    Notes:
      - Requires stabilizer_gens and G_gens have coefficients ±1 only.
      - Phases on G side are ignored in membership (± allowed), but we still
        only accept ±1 input for simplicity.
      - Output generators are returned with the stabilizer-consistent sign.
    """
    n = len(qubits)
    qindex = {q: i for i, q in enumerate(qubits)}

    # Build S_xz (k x 2n) and stabilizer sign bits s (k,)
    S_xz_list = []
    S_sbits = []
    for s in stabilizer_gens:
        x,z, sb = paulistring_to_xz_sign(s, qubits)
        xz = np.concatenate([x,z])
        S_xz_list.append(xz)
        S_sbits.append(sb)
    S_xz = np.stack(S_xz_list, axis=0).astype(np.uint8)
    S_sbits = np.array(S_sbits, dtype=np.uint8)  # sign bits of generators
    k = S_xz.shape[0]

    # Build G_xz (m x 2n); we IGNORE signs for membership, but still parse ±1 safely
    G_xz_list = []
    for g in G_gens:
        x,z, _ = paulistring_to_xz_sign(g, qubits)
        xz = np.concatenate([x,z])
        G_xz_list.append(xz)
    G_xz = np.stack(G_xz_list, axis=0).astype(np.uint8)
    m = G_xz.shape[0]

    # Solve for (u, v) such that u^T S_xz = v^T G_xz  (phase-free equality)
    # <=> u^T S_xz + v^T G_xz = 0
    # Let M = [S_xz; G_xz] of shape (k+m, 2n). We need w=[u,v] such that w^T M = 0.
    # That is: M^T w = 0.
    M = np.vstack([S_xz, G_xz]).astype(np.uint8)  # (k+m) x (2n)
    W_basis = gf2_nullspace(M.T)  # rows are w vectors of length (k+m)

    if W_basis.shape[0] == 0:
        return []  # only identity in the intersection

    # Map each solution w -> intersection element a = u^T S_xz, with stabilizer signbit = u·S_sbits
    candidates_xz = []
    candidates_sbit = []

    for w in W_basis:
        u = w[:k] & 1
        # v = w[k:]  (not needed)
        a_xz = (u @ S_xz) & 1
        # skip identity (xz all zeros); we do NOT want to return I as a generator
        if not a_xz.any():
            continue
        a_sbit = int((np.dot(u, S_sbits) & 1))  # sign from stabilizer generators
        candidates_xz.append(a_xz.astype(np.uint8))
        candidates_sbit.append(a_sbit)

    if not candidates_xz:
        return []

    C = np.stack(candidates_xz, axis=0).astype(np.uint8)

    # Reduce to an independent generating set for the intersection subgroup (phase-free)
    # We'll do rowspace basis on C; then pick corresponding signs by expressing each basis row
    # as a combination of the candidate rows (simple approach below).
    #
    # To keep this simple+reliable without bookkeeping, we instead:
    #   - take the rowspace basis as R (nonzero rows in RREF),
    #   - for each row r in R, solve C^T alpha = r^T (over GF(2)) to get some combination alpha,
    #     then compute its signbit as sum alpha_i * candidates_sbit[i].
    #
    # This keeps signs consistent with stabilizer group.

    R, _ = gf2_rref(C)
    basis_rows = [R[i] for i in range(R.shape[0]) if R[i].any()]
    basis = []
    # Precompute for solving C^T alpha = r^T:
    # We solve (C^T) alpha = r (vector length 2n), alpha length = num_candidates.
    CT = C.T.astype(np.uint8)

    for r in basis_rows:
        # Find one alpha such that CT @ alpha = r (GF(2))
        # Convert to solving [CT | r] by RREF.
        A = np.hstack([CT, r[:, None].astype(np.uint8)])  # (2n) x (num_cand + 1)
        Ar, piv = gf2_rref(A)
        nvar = CT.shape[1]

        # Check consistency: no row with all zeros in variables but RHS=1
        inconsistent = False
        for i in range(Ar.shape[0]):
            if not Ar[i, :nvar].any() and Ar[i, nvar] == 1:
                inconsistent = True
                break
        if inconsistent:
            # Shouldn't happen because r is in rowspace(C).
            continue

        # Choose a particular solution by setting free vars=0, pivot vars from RHS.
        alpha = np.zeros(nvar, dtype=np.uint8)
        pivot_cols = [p for p in piv if p < nvar]
        # In our rref, each pivot row corresponds to one pivot col; read RHS.
        prow = 0
        for pcol in pivot_cols:
            # Find the row that has pivot at pcol
            # Since it's RREF, it's the row where Ar[row, pcol]=1 and it's pivot.
            row = None
            for rr in range(Ar.shape[0]):
                if Ar[rr, pcol] == 1:
                    # ensure it's a pivot: no other 1s in that col
                    row = rr
                    break
            if row is None:
                continue
            alpha[pcol] = Ar[row, nvar]  # because free vars=0, pivot var = RHS
            prow += 1

        # Compute stabilizer-consistent signbit for this basis element
        sb = 0
        for i, abit in enumerate(alpha):
            if abit:
                sb ^= (candidates_sbit[i] & 1)

        basis.append(xz_sign_to_paulistring(r.astype(np.uint8), sb, qubits))

    return basis



def reduced_density_matrix_from_circuit(
    circuit: cirq.Circuit,
    keep_qubits: list[cirq.Qid],
    qubit_order: list[cirq.Qid] | None = None,
) -> np.ndarray:
    if qubit_order is None:
        qubit_order = sorted(circuit.all_qubits())
    sim = cirq.Simulator()
    result = sim.simulate(circuit, qubit_order=qubit_order)
    psi = np.array(result.final_state_vector)
    psi = psi/norm(psi)
    keep_indices=[qubit_order.index(q) for q in keep_qubits]
    dim = 2**(len(keep_indices))
    discard = [i for i in range(len(qubit_order)) if i not in keep_indices]
    psi = psi.reshape([2 for i in range(len(qubit_order))])
    rho = np.tensordot(psi, psi.conj(),[discard,discard]).reshape(dim,dim)
    return rho