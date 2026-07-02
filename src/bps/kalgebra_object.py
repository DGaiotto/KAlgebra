"""`KAlgebraObject` — one abstract K_𝖖-algebra, held as its
realizations plus the `KAlgebraIso` witnesses identifying them.

The repo's core epistemology (the "Core principle") is that
one abstract K_𝖖-algebra admits several concrete *implementations*
(presentations), and a `KAlgebraIso` *certifies* that two of them
present the same object.  `KAlgebraObject` is that epistemology made
executable: a **connected groupoid** whose objects are realizations
(concrete `KAlgebra` instances under string keys) and whose morphisms
are verified `KAlgebraIso` witnesses, with composed transport between
any two realizations and a path-independence (coherence) certificate.

Deliberate non-features:

* **`KAlgebraObject` is NOT itself a `KAlgebra`.**  The seven contract
  primitives are label-typed; a delegating object would need one
  authoritative label space, and callers would inevitably feed labels
  of the *wrong* realization — the "charges vs labels" bug class,
  multiplied across presentations.  The object polices the
  presentation/object distinction; it must not blur it.  (If a concrete
  consumer ever needs to pass one where a `KAlgebra` goes, an explicit
  `as_kalgebra(authority=key)` adapter view can be added — additively.)
* **Witnesses are held, not searched.**  Discovery stays in the tools
  that own it (`bpskalgebra_iso`, per-theory constructors, and external
  isomorphism-search tooling); the object stores
  verified `KAlgebraIso`s and composes them on demand.
* **Capability tags are declared, not introspected** — short strings
  like ``'multiply-fast'``, ``'trace-exact'``, ``'chart'`` supplied at
  `add_realization` time; `preferred(tag)` returns the first
  realization carrying the tag (insertion order = preference).

A natural population is the finite-type zoo of `src/cone/`, whose
algebras present one abstract object per Dynkin type.
"""
from __future__ import annotations

from typing import Iterable

from kalgebra import Element, KAlgebra
from kalgebra_iso import KAlgebraIso
from laurent_poly import LaurentPoly


__all__ = ["KAlgebraObject"]


class KAlgebraObject:
    """An abstract K_𝖖-algebra: realizations + iso witnesses.

    See the module docstring.  Realization keys are short strings
    (``'cone-frozen'``, ``'bps'``, ``'closed-form'``, …).
    """

    def __init__(self, name: str):
        self.name = name
        self._realizations: dict[str, KAlgebra] = {}
        self._capabilities: dict[str, frozenset[str]] = {}
        self._witnesses: dict[tuple[str, str], KAlgebraIso] = {}
        self._iso_cache: dict[tuple[str, str], KAlgebraIso] = {}

    # ---- population ----------------------------------------------------

    def add_realization(
        self, key: str, algebra: KAlgebra,
        capabilities: Iterable[str] = (),
    ) -> None:
        if key in self._realizations:
            raise ValueError(f"{self.name}: realization {key!r} already "
                             f"present")
        self._realizations[key] = algebra
        self._capabilities[key] = frozenset(capabilities)

    def add_iso(self, src: str, dst: str, iso: KAlgebraIso) -> None:
        """Record a verified witness `src → dst`.  The iso's endpoint
        instances must be the registered realizations."""
        for key in (src, dst):
            if key not in self._realizations:
                raise KeyError(f"{self.name}: unknown realization {key!r}")
        if iso.source is not self._realizations[src]:
            raise ValueError(
                f"{self.name}: iso.source is not the {src!r} realization "
                f"instance")
        if iso.target is not self._realizations[dst]:
            raise ValueError(
                f"{self.name}: iso.target is not the {dst!r} realization "
                f"instance")
        self._witnesses[(src, dst)] = iso
        self._iso_cache.clear()

    # ---- access / routing -----------------------------------------------

    def keys(self) -> list[str]:
        return list(self._realizations)

    def realization(self, key: str) -> KAlgebra:
        return self._realizations[key]

    def capabilities(self, key: str) -> frozenset[str]:
        return self._capabilities[key]

    def preferred(self, capability: str) -> KAlgebra:
        """First realization (insertion order) carrying the tag."""
        for key, caps in self._capabilities.items():
            if capability in caps:
                return self._realizations[key]
        raise KeyError(
            f"{self.name}: no realization declares capability "
            f"{capability!r} (have "
            f"{sorted(set().union(*self._capabilities.values()) if self._capabilities else set())})")

    # ---- groupoid: composed isos + transport ------------------------------

    def _edges(self) -> dict[str, list[tuple[str, KAlgebraIso]]]:
        """Adjacency of the witness graph: stored witnesses plus their
        inverses (a witness certifies both directions)."""
        adj: dict[str, list[tuple[str, KAlgebraIso]]] = {
            k: [] for k in self._realizations}
        for (s, d), iso in self._witnesses.items():
            adj[s].append((d, iso))
            adj[d].append((s, iso.invert()))
        return adj

    def iso(self, src: str, dst: str) -> KAlgebraIso:
        """The composed witness `src → dst` along a shortest witness
        path (BFS).  Cached.  Raises if the groupoid is disconnected
        between the two keys."""
        if (src, dst) in self._iso_cache:
            return self._iso_cache[(src, dst)]
        if src == dst:
            out = KAlgebraIso.identity(self._realizations[src],
                                       name=f"id[{src}]")
            self._iso_cache[(src, dst)] = out
            return out
        adj = self._edges()
        prev: dict[str, tuple[str, KAlgebraIso]] = {}
        frontier = [src]
        seen = {src}
        while frontier and dst not in seen:
            nxt = []
            for u in frontier:
                for v, e in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        prev[v] = (u, e)
                        nxt.append(v)
            frontier = nxt
        if dst not in prev:
            raise KeyError(
                f"{self.name}: no witness path {src!r} → {dst!r}")
        # reconstruct path and compose
        chain: list[KAlgebraIso] = []
        node = dst
        while node != src:
            u, e = prev[node]
            chain.append(e)
            node = u
        out = chain[-1]
        for e in reversed(chain[:-1]):
            out = out.compose(e)
        out.name = out.name or f"{self.name}[{src}→{dst}]"
        self._iso_cache[(src, dst)] = out
        return out

    def transport(self, label, src: str, dst: str) -> Element:
        """The image of the `src` canonical basis label `L_label` in the
        `dst` realization, along the composed witness."""
        one = LaurentPoly.one()
        return self.iso(src, dst).map(Element({label: one}))

    def discover(self, src: str, dst: str, searcher, **kw) -> KAlgebraIso:
        """Find-and-store a witness `src → dst` using an external
        `searcher(src_alg, dst_alg, **kw) -> KAlgebraIso` (discovery
        stays in the tools that own it — e.g. a generator-matching
        search for cone↔cone dictionaries).  The searcher is responsible
        for certification; whatever it returns is stored as a witness
        and returned."""
        iso = searcher(self._realizations[src], self._realizations[dst],
                       **kw)
        self.add_iso(src, dst, iso)
        return iso

    # ---- certification ----------------------------------------------------

    def verify_pairwise(
        self,
        samples_by_key: dict[str, list],
        *,
        pairs: bool = True,
        trace_K: int = 8,
        edge_samples: "dict[tuple[str, str], tuple[list, list]] | None" = None,
    ) -> dict[tuple[str, str], dict]:
        """Run each stored witness's `verify_all` battery on the supplied
        per-realization label samples.  Returns
        `{(src, dst): {check: bool}}`; all values should be all-True.

        `edge_samples` optionally overrides the `(src_labels,
        dst_labels)` used for a specific stored edge.  This exists for
        edges whose realizations are exact but *expensive* (e.g. a
        derived RG-flow trace at several minutes per call): every
        battery check pulls the **destination** list back through the
        source algebra too, so trimming only the cheap side's per-key
        list cannot bound the cost — the heavy edge needs small lists
        on BOTH sides, while the cheap edges keep the full ones."""
        one = LaurentPoly.one()
        out: dict[tuple[str, str], dict] = {}
        for (s, d), iso in self._witnesses.items():
            if edge_samples and (s, d) in edge_samples:
                src_l, dst_l = edge_samples[(s, d)]
            else:
                src_l, dst_l = samples_by_key[s], samples_by_key[d]
            src_e = [Element({l: one}) for l in src_l]
            dst_e = [Element({l: one}) for l in dst_l]
            src_pairs = ([(a, b) for a in src_e for b in src_e]
                         if pairs else [])
            dst_pairs = ([(a, b) for a in dst_e for b in dst_e]
                         if pairs else [])
            out[(s, d)] = iso.verify_all(
                src_e, dst_e, src_pairs, dst_pairs, trace_K=trace_K)
        return out

    def verify_coherence(
        self, samples_by_key: dict[str, list], *, max_path_len: int = 4,
    ) -> bool:
        """Path-independence of transport: for every ordered pair of
        realizations and every pair of distinct simple witness paths
        between them (up to `max_path_len` edges), the transported
        samples agree.  This is the well-definedness certificate for
        the abstract object (the groupoid is coherent on the samples)."""
        one = LaurentPoly.one()
        adj = self._edges()

        def paths(u, dst, used, acc):
            if len(acc) > max_path_len:
                return
            if u == dst and acc:
                yield list(acc)
                return
            for v, e in adj[u]:
                if v in used:
                    continue
                used.add(v)
                acc.append(e)
                yield from paths(v, dst, used, acc)
                acc.pop()
                used.discard(v)

        keys = list(self._realizations)
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                found = list(paths(src, dst, {src}, []))
                if len(found) < 2:
                    continue
                images = []
                for chain in found:
                    iso = chain[0]
                    for e in chain[1:]:
                        iso = iso.compose(e)
                    images.append([
                        iso.map(Element({l: one}))
                        for l in samples_by_key[src]
                    ])
                ref = images[0]
                for other in images[1:]:
                    if other != ref:
                        return False
        return True

    def __repr__(self) -> str:
        return (f"KAlgebraObject[{self.name}: "
                f"{', '.join(self._realizations)}; "
                f"{len(self._witnesses)} witnesses]")
