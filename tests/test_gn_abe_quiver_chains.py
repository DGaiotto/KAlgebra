"""Linear U(N_1)x...xU(N_n) quiver chains on the GENERAL (G, N) tier —
`GNAbeKAlgebra(root_datum.product_datum([u_n(N_1), ...]), bifundamentals)`.

This is the certification that lets the type-A `UNQuiverKAlgebra` retire.  Its adversarial review (same day)
found that the general tier had NO tracked test on a product datum, so the
gate for the move was: port the chain certification here and ship it first.

What is pinned, per chain (U(1)x U(1), U(2)x U(1), U(2)x U(2), U(1)^3 with two
bifundamentals):

* the contract axioms on a label basket — rho round-trips, rho fixes the
  identity, rho is an automorphism on a pair, bar involution, orthonormality
  at K=3, and the W1/W2 acceptance `certify_canonical` on every label;
* the products, the identity's trace and diagonal inner products that
  the suite in the source repository pinned on the old class (the U(2)x U(2)
  meson seam `E.F = L_{(1,-1)} + q L_{(det_1, det_2^{-1})}` among them), plus
  the root-bubbling labels its review added (m=(1,-1,0), (0,-2,0) at
  U(2)xU(1); (-2,0), (-1,1) at U(1)xU(1)).  The expected values were generated
  on 2026-09-19 with the old class as the positive control (equal on every
  pin: 19 products, 4 traces, 16 inner products), through the per-node label
  dictionary (a flat (m, e) split node by node) and, for trace / inner
  product, through `forget()` — the D8b specialization of the link U(1)
  flavour rings that the old class carried as its coefficient ring.

Run:  `python3 run_tests.py`
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from root_datum import u_n, product_datum
from laurent_poly import LaurentPoly
from gn_abe_kalgebra import GNAbeKAlgebra

K = 4

PINS = {
 "u1u1": {
  "ranks": [
   1,
   1
  ],
  "matter": [
   [
    1,
    -1
   ]
  ],
  "labels": {
   "id": [
    [
     0,
     0
    ],
    [
     0,
     0
    ]
   ],
   "W1": [
    [
     0,
     0
    ],
    [
     1,
     0
    ]
   ],
   "W2": [
    [
     0,
     0
    ],
    [
     0,
     1
    ]
   ],
   "W12": [
    [
     0,
     0
    ],
    [
     1,
     -1
    ]
   ],
   "M1": [
    [
     -1,
     0
    ],
    [
     0,
     0
    ]
   ],
   "M1p": [
    [
     1,
     0
    ],
    [
     0,
     0
    ]
   ],
   "B1": [
    [
     -2,
     0
    ],
    [
     0,
     0
    ]
   ],
   "B2": [
    [
     -1,
     1
    ],
    [
     0,
     0
    ]
   ]
  },
  "multiply": {
   "M1*M1p": {
    "(((0,), (0,)), ((0,), (0,)))": {
     "0": 1
    },
    "(((0,), (1,)), ((0,), (-1,)))": {
     "-1": 1
    }
   },
   "M1*W1": {
    "(((-1,), (1,)), ((0,), (0,)))": {
     "-1": 1
    }
   },
   "M1*M1": {
    "(((-2,), (0,)), ((0,), (0,)))": {
     "0": 1
    }
   },
   "W1*W2": {
    "(((0,), (1,)), ((0,), (1,)))": {
     "0": 1
    }
   },
   "B1*M1p": {
    "(((-1,), (0,)), ((0,), (0,)))": {
     "0": 1
    },
    "(((-1,), (1,)), ((0,), (-1,)))": {
     "-2": 1
    }
   },
   "B2*W12": {
    "(((-1,), (1,)), ((1,), (-1,)))": {
     "-2": 1
    }
   }
  },
  "trace_id": {
   "0": 1,
   "2": -3,
   "4": 1
  },
  "inner": {
   "id": {
    "0": 1,
    "2": -3,
    "4": 1
   },
   "W1": {
    "0": 1,
    "2": -3,
    "4": 1
   },
   "W2": {
    "0": 1,
    "2": -3,
    "4": 1
   },
   "W12": {
    "0": 1,
    "2": -3,
    "4": 1
   }
  }
 },
 "u2u1": {
  "ranks": [
   2,
   1
  ],
  "matter": [
   [
    1,
    0,
    -1
   ]
  ],
  "labels": {
   "id": [
    [
     0,
     0,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "E": [
    [
     1,
     0,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "F": [
    [
     0,
     -1,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "F2": [
    [
     0,
     0,
     -1
    ],
    [
     0,
     0,
     0
    ]
   ],
   "CHI": [
    [
     0,
     0,
     0
    ],
    [
     1,
     0,
     0
    ]
   ],
   "W2": [
    [
     0,
     0,
     0
    ],
    [
     0,
     0,
     1
    ]
   ],
   "ADJ": [
    [
     1,
     -1,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "FF": [
    [
     0,
     -2,
     0
    ],
    [
     0,
     0,
     0
    ]
   ]
  },
  "multiply": {
   "E*F": {
    "(((1, -1), (0, 0)), ((0,), (0,)))": {
     "0": 1
    }
   },
   "F*F2": {
    "(((0, -1), (0, 0)), ((-1,), (0,)))": {
     "0": 1
    },
    "(((0, -1), (0, 1)), ((-1,), (-1,)))": {
     "-1": 1
    }
   },
   "E*F2": {
    "(((1, 0), (0, 0)), ((-1,), (0,)))": {
     "0": 1
    }
   },
   "CHI*F": {
    "(((0, -1), (1, 0)), ((0,), (0,)))": {
     "0": 1
    },
    "(((0, -1), (0, 1)), ((0,), (0,)))": {
     "1": 1
    }
   },
   "CHI*W2": {
    "(((0, 0), (1, 0)), ((0,), (1,)))": {
     "0": 1
    }
   },
   "ADJ*E": {
    "(((2, -1), (0, 0)), ((0,), (0,)))": {
     "0": 1
    }
   },
   "FF*E": {
    "(((1, -2), (0, 0)), ((0,), (0,)))": {
     "0": 1
    }
   }
  },
  "trace_id": {
   "0": 1,
   "2": -3
  },
  "inner": {
   "id": {
    "0": 1,
    "2": -3
   },
   "E": {
    "4": 4,
    "0": 1,
    "2": -4
   },
   "F": {
    "0": 1,
    "2": -4,
    "4": 4
   },
   "F2": {
    "2": -4,
    "4": 4,
    "0": 1
   }
  }
 },
 "u2u2": {
  "ranks": [
   2,
   2
  ],
  "matter": [
   [
    1,
    0,
    0,
    -1
   ]
  ],
  "labels": {
   "id": [
    [
     0,
     0,
     0,
     0
    ],
    [
     0,
     0,
     0,
     0
    ]
   ],
   "E": [
    [
     1,
     0,
     0,
     0
    ],
    [
     0,
     0,
     0,
     0
    ]
   ],
   "F": [
    [
     0,
     -1,
     0,
     0
    ],
    [
     0,
     0,
     0,
     0
    ]
   ],
   "F2": [
    [
     0,
     0,
     0,
     -1
    ],
    [
     0,
     0,
     0,
     0
    ]
   ],
   "E2": [
    [
     0,
     0,
     1,
     0
    ],
    [
     0,
     0,
     0,
     0
    ]
   ]
  },
  "multiply": {
   "E*F": {
    "(((1, -1), (0, 0)), ((0, 0), (0, 0)))": {
     "0": 1
    },
    "(((0, 0), (1, 1)), ((0, 0), (-1, -1)))": {
     "1": 1
    }
   },
   "E2*F2": {
    "(((0, 0), (0, 0)), ((1, -1), (0, 0)))": {
     "0": 1
    },
    "(((0, 0), (1, 1)), ((0, 0), (-1, -1)))": {
     "-1": 1
    }
   },
   "E*E2": {
    "(((1, 0), (0, 0)), ((1, 0), (0, 0)))": {
     "0": 1
    },
    "(((1, 0), (1, 0)), ((1, 0), (-1, 0)))": {
     "1": 1
    }
   }
  },
  "trace_id": {
   "0": 1,
   "2": -3
  },
  "inner": {
   "id": {
    "0": 1,
    "2": -3
   },
   "E": {
    "0": 1,
    "2": -4,
    "4": 3
   },
   "F": {
    "0": 1,
    "2": -4,
    "4": 3
   },
   "F2": {
    "0": 1,
    "2": -4,
    "4": 3
   }
  }
 },
 "u1u1u1": {
  "ranks": [
   1,
   1,
   1
  ],
  "matter": [
   [
    1,
    -1,
    0
   ],
   [
    0,
    1,
    -1
   ]
  ],
  "labels": {
   "id": [
    [
     0,
     0,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "m": [
    [
     -1,
     0,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "p": [
    [
     1,
     0,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "mid": [
    [
     0,
     -1,
     0
    ],
    [
     0,
     0,
     0
    ]
   ],
   "W2": [
    [
     0,
     0,
     0
    ],
    [
     0,
     1,
     0
    ]
   ]
  },
  "multiply": {
   "m*p": {
    "(((0,), (0,)), ((0,), (0,)), ((0,), (0,)))": {
     "0": 1
    },
    "(((0,), (1,)), ((0,), (-1,)), ((0,), (0,)))": {
     "-1": 1
    }
   },
   "m*mid": {
    "(((-1,), (0,)), ((-1,), (0,)), ((0,), (0,)))": {
     "0": 1
    },
    "(((-1,), (1,)), ((-1,), (-1,)), ((0,), (0,)))": {
     "-1": 1
    }
   },
   "W2*mid": {
    "(((0,), (0,)), ((-1,), (1,)), ((0,), (0,)))": {
     "1": 1
    }
   }
  },
  "trace_id": {
   "0": 1,
   "2": -4,
   "4": 4
  },
  "inner": {
   "id": {
    "0": 1,
    "2": -4,
    "4": 4
   },
   "m": {
    "0": 1,
    "2": -5,
    "4": 7
   },
   "p": {
    "2": -5,
    "4": 7,
    "0": 1
   },
   "mid": {
    "2": -6,
    "4": 11,
    "0": 1
   }
  }
 }
}


def _split(m, e, ranks):
    off = [0]
    for r in ranks:
        off.append(off[-1] + r)
    return tuple((tuple(m[off[a]:off[a + 1]]), tuple(e[off[a]:off[a + 1]]))
                 for a in range(len(ranks)))


def _push(x, ranks):
    """A GN Element to the per-node-keyed dict (flavour slot collapsed)."""
    out = {}
    for ((m, e), _w), c in x.terms.items():
        if hasattr(c, "is_zero") and c.is_zero():
            continue
        k = repr(_split(m, e, ranks))
        out[k] = (out[k] + c) if k in out else c
    return {k: {int(e): int(v) for e, v in c._coeffs.items() if v}
            for k, c in out.items() if not c.is_zero()}


def _flat(rps):
    out = {}
    for e, c in rps.coeffs.items():
        z = sum(c.terms.values()) if hasattr(c, "terms") else c
        if z:
            out[int(e)] = int(z)
    return out


def _build(rec):
    ranks = tuple(rec["ranks"])
    matter = tuple(tuple(w) for w in rec["matter"])
    D = product_datum([u_n(r) for r in ranks])
    GN = GNAbeKAlgebra(D, matter[0] if len(matter) == 1 else matter)
    w0 = GN.identity()[1]
    labels = {n: ((tuple(g[0]), tuple(g[1])), w0) for n, g in rec["labels"].items()}
    return ranks, GN, labels


def _run_chain(name, rec):
    ranks, GN, gl = _build(rec)
    n_ok = 0
    # ---- axioms on the basket -------------------------------------------
    idn = GN.identity()
    assert GN.verify_rho_fixes_identity(), name
    for lab in gl.values():
        assert GN.rho_inverse(GN.rho(lab)) == lab, (name, lab)
        assert GN.certify_canonical(lab), (name, lab)     # W1 + W2 acceptance
        n_ok += 2
    names = list(gl)
    a, b = gl[names[1]], gl[names[2]]
    assert GN.verify_rho_is_automorphism(a, b), (name, a, b)
    assert GN.verify_bar_involution(a, b), (name, a, b)
    n_ok += 2
    for x in names:
        for y in names:
            assert GN.verify_orthonormality(gl[x], gl[y], K=3), (name, x, y)
            n_ok += 1
    # ---- pinned products ------------------------------------------------
    for key, want in rec["multiply"].items():
        x, y = key.split("*")
        got = _push(GN.multiply(gl[x], gl[y]), ranks)
        assert got == {k: {int(e): v for e, v in d.items()} for k, d in want.items()}, (name, key, got, want)
        n_ok += 1
    # ---- trace / inner product through forget() --------------------------
    Fg = GN.forget()
    sec = (lambda l: Fg._section_of(l)) if hasattr(Fg, "_section_of") else (lambda l: l)
    got = _flat(Fg.trace(sec(gl["id"]), K=K))
    assert got == {int(e): v for e, v in rec["trace_id"].items()}, (name, got, rec["trace_id"])
    n_ok += 1
    for lab, want in rec["inner"].items():
        got = _flat(Fg.inner_product(sec(gl[lab]), sec(gl[lab]), K=K))
        assert got == {int(e): v for e, v in want.items()}, (name, lab, got, want)
        n_ok += 1
    print(f"  PASS {name}: {n_ok} checks")
    return n_ok


def test_u1_u1():
    _run_chain("u1u1", PINS["u1u1"])


def test_u2_u1():
    _run_chain("u2u1", PINS["u2u1"])


def test_u2_u2():
    _run_chain("u2u2", PINS["u2u2"])


def test_u1_cubed_two_bifundamentals():
    _run_chain("u1u1u1", PINS["u1u1u1"])


def test_product_datum_rejects_fundamental_misfire():
    """`faster_equivalent()` must not hand a product datum to a single-node
    class (the 2026-09-19 review found it did)."""
    GN = GNAbeKAlgebra(product_datum([u_n(1), u_n(1)]), (1, 0))
    assert GN.faster_equivalent() is None


if __name__ == "__main__":
    import time
    t = time.time()
    for fn in (test_u1_u1, test_u2_u1, test_u1_cubed_two_bifundamentals, test_u2_u2,
               test_product_datum_rejects_fundamental_misfire):
        fn()
    print(f"All GNAbeKAlgebra quiver-chain tests passed.   [{time.time() - t:.1f}s]")

