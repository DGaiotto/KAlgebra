"""render_battery.py — the claim registry rendered two ways.

Reads `claims.json` (one entry per claim or conjecture of the author's
`K_q-algebras` draft, keyed on the draft's LaTeX labels), asserts that EVERY
label of the draft is either attached to a claim or listed under `uncovered`
with a reason, and writes

  * `paper_companion_battery.tex` at the repo root — the appendix the companion
    `\input`s (one longtable per section of the paper, plus a census);
  * `battery.md` — the Markdown twin
    for sessions (id, labels, kind, standing, cost tier, environment, what
    implements it).

Every claim carries two execution axes: the COST tier `sweep` (fast /
extensive / none) and the ENVIRONMENT list (`web` = the tracked repository is
the whole input, so the web session and CI can run it; `local` = the run needs
data only the local machine holds), with `requires` naming the inputs per
environment.  The renderer validates both and prints them.

Pure Python, no dependencies.  Run from the repo root:

    PYTHONPATH=. python3 battery/render_battery.py

Exit status 1 with the list of uncovered labels if the coverage assertion
fails — that failure is the point of the check (a new label in the draft must
be classified before the companion can be regenerated).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))     # the release root: battery/ sits at its top
REG = os.path.join(HERE, "claims.json")
TEX_OUT = os.path.join(ROOT, "paper_companion_battery.tex")
MD_OUT = os.path.join(HERE, "battery.md")
RESULTS = os.path.join(HERE, "results")


def latest_records() -> dict:
    """{claim id: {environment[ (extensive)]: record}} — the latest result record per (claim, environment, depth);
    an extensive record is printed beside the fast one."""
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "*", "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        key = rec["environment"] + (" (extensive)" if rec.get("depth") == "extensive" else "")
        out.setdefault(rec["id"], {})[key] = rec
    return out

STANDING_MACRO = {
    "derived": r"\DERIVED", "certified": r"\CERTIFIED", "measured": r"\MEASURED",
    "open": r"\OPEN", "planned": r"\PLANNED", "n.a.": r"---",
}
SWEEPS = ("fast", "extensive", "none")
ENVIRONMENTS = ("web", "local")
SECTION_LABELS = [
    ("sec:kq", r"Section~\pref{sec:kq}: $K_\fq$-algebras"),
    ("sec:coulomb", r"Section~\pref{sec:coulomb}: Coulomb branch algebras"),
    ("sec:rg", r"Section~\pref{sec:rg}: RG flows"),
    ("sec:skein", r"Section~\pref{sec:skein}: skein algebras"),
    ("app:physics", r"Appendix~\pref{app:physics}: physical motivations"),
    ("app:finite", r"Appendix~\pref{app:finite}: finite type"),
]


def tex_text(s: str) -> str:
    """Plain prose (notes, populations) -> LaTeX-safe; code is NOT passed here."""
    rep = [("\\", "\\textbackslash{}"), ("_", r"\_"), ("^", r"\^{}"), ("#", r"\#"),
           ("&", r"\&"), ("%", r"\%"), ("§", "sec."), ("±", r"$\pm$"), ("≤", r"$\le$"),
           ("≥", r"$\ge$"), ("→", r"$\to$"), ("𝖖", r"$\fq$"), ("ρ", r"$\rho$"), ("Ω", r"$\Omega$"), ("Π", r"$\Pi$"), ("★", r"$\star$"), ("·", r"$\cdot$"), ("Σ", r"$\Sigma$"), ("Ξ", r"$\Xi$"), ("Δ", r"$\Delta$"), ("Λ", r"$\Lambda$"), ("Φ", r"$\Phi$"), ("Ψ", r"$\Psi$"), ("Θ", r"$\Theta$"), ("κ", r"$\kappa$"), ("ω", r"$\omega$"), ("ν", r"$\nu$"), ("θ", r"$\theta$"), ("ε", r"$\epsilon$"), ("η", r"$\eta$"), ("ζ", r"$\zeta$"), ("ξ", r"$\xi$"), ("φ", r"$\phi$"), ("ψ", r"$\psi$"), ("β", r"$\beta$"), ("ħ", r"$\hbar$"),
           ("~", r"$\sim$"), (">=", r"$\ge$"), ("<=", r"$\le$"),
           ("γ", r"$\gamma$"), ("μ", r"$\mu$"), ("δ", r"$\delta$"), ("Γ", r"$\Gamma$"), ("α", r"$\alpha$"),
           ("λ", r"$\lambda$"), ("χ", r"$\chi$"), ("τ", r"$\tau$"), ("σ", r"$\sigma$"), ("π", r"$\pi$"), ("∞", r"$\infty$"),
           ("⁻¹", r"$^{-1}$"), ("⁻", r"$^{-}$"), ("∘", r"$\circ$"), ("ℓ", r"$\ell$"), ("−", r"$-$")]
    for a, b in rep:
        s = s.replace(a, b)
    return s


def tex_code(s: str) -> str:
    """A url-style \\code argument must be ASCII: map the few symbols that occur."""
    for a, b in [("§", "sec. "), ("±", "+/-"), ("—", "-"), ("–", "-"), ("𝖖", "q"), ("ρ", "rho")]:
        s = s.replace(a, b)
    return r"\code{" + s + "}"


PATH_SUFFIXES = (".json", ".py", ".md", ".tex", ".gz", ".log")


def tex_prose_paths(s: str) -> str:
    """Prose that mentions repository paths: each path-like token is set in the
    url-style \\code (so it can break at / . _ -), the rest goes through tex_text."""
    out = []
    for tok in s.split(" "):
        lead = ""
        while tok and tok[0] in "(":
            lead += tok[0]
            tok = tok[1:]
        trail = ""
        while tok and tok[-1] in ",;:)":
            trail = tok[-1] + trail
            tok = tok[:-1]
        is_path = tok and ("/" in tok or tok.endswith(PATH_SUFFIXES) or "_*" in tok) and " " not in tok and "=" not in tok
        core = tex_code(tok) if is_path else tex_text(tok)
        out.append(tex_text(lead) + core + tex_text(trail))
    return " ".join(out)


def group_of(claim, order):
    if all(l.startswith("comp:") for l in claim["labels"]):
        return "comp"
    idx = min(order.index(l) for l in claim["labels"] if l in order) if any(l in order for l in claim["labels"]) else -1
    grp = "intro"
    for lab, _title in SECTION_LABELS:
        if lab in order and order.index(lab) <= idx:
            grp = lab
    return grp


def main() -> int:
    reg = json.load(open(REG, encoding="utf-8"))
    labels = reg["draft"]["labels"]
    claims = reg["claims"]
    covered = {l for c in claims for l in c["labels"]}
    missing = [l for l in labels if l not in covered and l not in reg["uncovered"]]
    # labels beginning with "comp:" are the COMPANION's own (its extra examples, Section 2 of the companion): they name
    # companion sections, not draft labels, and are exempt from the draft-membership check; a claim mixes the two spaces never
    unknown = sorted({l for l in covered if l not in labels and not l.startswith("comp:")})
    for c in claims:
        kinds_of_labels = {l.startswith("comp:") for l in c["labels"]}
        assert len(kinds_of_labels) == 1, "a claim's labels are all draft labels or all companion labels: " + c["id"]
    if missing or unknown:
        print("COVERAGE FAILURE: uncovered draft labels:", missing, "; claim labels absent from the draft:", unknown)
        return 1
    ids = [c["id"] for c in claims]
    assert len(ids) == len(set(ids)), "duplicate claim ids"
    for c in claims:
        assert c["sweep"] in SWEEPS, (c["id"], c["sweep"])
        env = c["environment"]
        assert isinstance(env, list) and all(e in ENVIRONMENTS for e in env) and len(set(env)) == len(env), (c["id"], env)
        assert (c["sweep"] == "none") == (env == []), "a claim has a cost tier iff it names an environment: " + c["id"]
        assert (c["kind"] == "definition") == (c["standing"] == "n.a."), "a definition row has no standing, and only a definition row lacks one: " + c["id"]
        assert c["kind"] != "definition" or c["sweep"] == "none", "a definition row is not run: " + c["id"]
        assert isinstance(c["requires"], dict) and set(c["requires"]) <= set(env), (c["id"], c["requires"])
        if "local" in env:
            assert c["requires"].get("local"), "a local claim must name the local-only inputs: " + c["id"]

    groups = OrderedDict([("intro", r"Section~1: introduction")] + SECTION_LABELS + [("comp", r"The companion's own examples (\S\ref{sec:extra-examples} of this companion, not in the paper)")])
    by_group = {g: [] for g in groups}
    for c in claims:
        by_group[group_of(c, labels)].append(c)

    records = latest_records()
    for cid in records:
        assert cid in {c["id"] for c in claims}, "result record for an id absent from the registry: " + cid
    standing = Counter(c["standing"] for c in claims)
    kinds = Counter(c["kind"] for c in claims)
    sweeps = Counter(c["sweep"] for c in claims)
    envs = Counter(("+".join(c["environment"]) or "none") for c in claims)

    out = []
    out.append("%% GENERATED by battery/render_battery.py")
    out.append("%% from claims.json — do not edit by hand; edit the registry and re-render.")
    out.append(r"\subsection*{Census}")
    out.append(r"The registry holds %d claims covering all %d labels of the %s version of the draft (%d labels are section heads or bare definitions, listed as such in the registry)."
               % (len(claims), len(labels), reg["draft"]["version"], len(reg["uncovered"])))
    out.append(r"\begin{center}\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{11.5cm}@{}}\toprule")
    out.append(r"standing & " + ", ".join(f"{STANDING_MACRO[k]}{{}} {v}" for k, v in sorted(standing.items()) if k != "n.a.") + r"; definition rows (no standing) %d \\" % standing.get("n.a.", 0))
    out.append(r"kind & " + ", ".join(f"{tex_text(k)} {v}" for k, v in sorted(kinds.items())) + r" \\")
    out.append(r"cost tier & " + ", ".join(f"{tex_text(k)} {v}" for k, v in sorted(sweeps.items())) + r" \\")
    out.append(r"environment & " + ", ".join(f"{tex_text(k)} {v}" for k, v in sorted(envs.items())) + r" \\")
    n_runs = sum(len(v) for v in records.values())
    n_checks = sum(r["counts"]["pass"] + r["counts"]["fail"] for v in records.values() for r in v.values())
    n_fail = sum(r["counts"]["fail"] for v in records.values() for r in v.values())
    out.append(r"result records & %d claims run, %d checks, %d failing (latest per environment and depth; %s) \\"
               % (len(records), n_checks, n_fail, ", ".join(sorted({r["date"][:10] for v in records.values() for r in v.values()})) or "none"))
    out.append(r"\bottomrule\end{tabular}\end{center}")
    out.append(r"\noindent\emph{Kinds}: a \emph{definition} row is not a claim --- a definition has no truth value, and checking an axiom on a realisation tests the realisation, not the definition (the author's ruling) --- so it carries no standing; it records per tier where the axiom is enforced by construction and where it is emergent, and names the rows that test it there, the verifiers being the instruments; a \emph{theorem} is proved in the paper or the repository and checked for consistency; an \emph{identity} is a closed form checked to a stated order; a \emph{conjecture} gathers evidence; \emph{coverage} is a classification table; \emph{not-testable} has no counterpart in code.  \emph{Cost tiers}: \emph{fast} runs in minutes with the suite; \emph{extensive} runs for hours with windows and depths that grow from run to run.  \emph{Environments}: \emph{web} means the tracked repository is the whole input, so the claim runs in the web session and in continuous integration; \emph{local} means the run needs data that only the local machine holds --- the gitignored dictionary builds (the enumerated build beyond the shipped weight, $E\ge13$, which stays on the local machine, the flavoured staging builds, the weak dictionary with its mutation edges) --- so it runs there and its result records are committed from there.  A claim listing both has a web-sized population (the shipped dictionary tiers, whose coverage is read from their manifests) and a local-sized one, each named under \emph{Inputs}.  \emph{Standing} \PLANNED{} marks a test designed here and not yet implemented.")
    for g, title in groups.items():
        cs = by_group[g]
        if not cs:
            continue
        out.append("")
        out.append(r"\subsection*{%s}" % title)
        out.append(r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{4.7cm}>{\raggedright\arraybackslash}p{4.95cm}>{\raggedright\arraybackslash}p{1.85cm}@{}}")
        out.append(r"\toprule claim & statement & test: procedure, population, controls & standing \\ \midrule \endfirsthead")
        out.append(r"\toprule claim & statement & test: procedure, population, controls & standing \\ \midrule \endhead")
        out.append(r"\bottomrule \endfoot")
        for c in cs:
            cell1 = r"\code{%s}\newline %s" % (c["id"], c["where"])
            cell3 = c["procedure"]
            if c["population"] and c["population"] != "n.a.":
                cell3 += r" \emph{Population:} " + c["population"].rstrip().rstrip(".") + "."
            if c.get("controls"):
                cell3 += r" \emph{Controls:} " + c["controls"].rstrip().rstrip(".") + "."
            if c["evidence"] and c["evidence"] != "n.a.":
                cell3 += r" \emph{Evidence:} " + tex_text(c["evidence"].rstrip().rstrip(".")) + "."
            if c["implemented_by"]:
                cell3 += r" \emph{Where:} " + ", ".join(tex_code(x) for x in c["implemented_by"]) + "."
            if c["requires"]:
                cell3 += r" \emph{Inputs:} " + "; ".join(
                    r"\emph{%s} --- %s" % (e, "; ".join(tex_prose_paths(x) for x in c["requires"][e]))
                    for e in c["environment"] if c["requires"].get(e)) + "."
            if c.get("notes"):
                cell3 += r" \emph{Note:} " + tex_text(c["notes"].rstrip().rstrip(".")) + "."
            env_txt = ", ".join(c["environment"]) if c["environment"] else "no test"
            if c["kind"] == "definition":
                cell4 = r"---\newline {\footnotesize definition, not run}"
            else:
                cell4 = STANDING_MACRO[c["standing"]] + r"\newline {\footnotesize " + tex_text(c["kind"]) + ", " + tex_text(c["sweep"]) + r"\newline " + env_txt + "}"
            if c.get("formalised_by"):
                cell4 += r"\newline {\footnotesize formalised: " + tex_code(c["formalised_by"]) + "}"
            for env_name, rec in sorted(records.get(c["id"], {}).items()):
                cell4 += r"\newline {\footnotesize run %s %s: %d/%d}" % (env_name, rec["date"][:10], rec["counts"]["pass"], rec["counts"]["pass"] + rec["counts"]["fail"])
                cell3 += r" \emph{Run (%s, %s):} %d checks, %d failing; %s." % (env_name, rec["date"][:10], rec["counts"]["pass"] + rec["counts"]["fail"], rec["counts"]["fail"],
                                                                              tex_text("; ".join(f"{k}: {v}" for k, v in rec["population"].items())))
            out.append(" & ".join([cell1, c["statement"], cell3, cell4]) + r" \\ \addlinespace")
        out.append(r"\end{longtable}")
    open(TEX_OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")

    md = ["# The battery of tests for the paper's claims — registry twin (GENERATED)", "",
          "Generated by `render_battery.py` from `claims.json`; do not edit by hand.  Draft version %s; %d claims over %d labels; standing: %s; environment: %s (`web` = the tracked repository suffices, so the web session and CI can run it; `local` = needs the local machine's gitignored dictionary builds; `requires` in the registry names the inputs per environment)."
          % (reg["draft"]["version"], len(claims), len(labels), ", ".join(f"{k} {v}" for k, v in sorted(standing.items())), ", ".join(f"{k} {v}" for k, v in sorted(envs.items()))), "",
          "| id | draft labels | kind | standing | cost tier | environment | latest run (env date pass/total) | implemented by |", "|---|---|---|---|---|---|---|---|"]
    for c in claims:
        runs = "; ".join(f"{e} {r['date'][:10]} {r['counts']['pass']}/{r['counts']['pass'] + r['counts']['fail']}" for e, r in sorted(records.get(c["id"], {}).items())) or "—"
        md.append("| `%s` | %s | %s | **%s** | %s | %s | %s | %s |" % (c["id"], ", ".join(f"`{l}`" for l in c["labels"]), c["kind"], c["standing"], c["sweep"], ", ".join(c["environment"]) or "—", runs, ", ".join(f"`{x}`" for x in c["implemented_by"]) or "—"))
    md.append("")
    md.append("Labels not attached to a claim, with the reason: " + "; ".join(f"`{k}` ({v})" for k, v in reg["uncovered"].items()) + ".")
    open(MD_OUT, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"rendered {len(claims)} claims -> {os.path.relpath(TEX_OUT, ROOT)}, {os.path.relpath(MD_OUT, ROOT)}; standing {dict(standing)}; environment {dict(envs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
