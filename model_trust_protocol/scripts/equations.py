# ruff: noqa: RUF001
"""The manuscript's displayed equations, defined once.

Both builders read from here, so the Word and LaTeX renderings cannot drift.
Each equation carries a LaTeX body and a Word form given as typed segments,
because Word cannot typeset an equation from a plain string and the house style
requires italic variables with real subscripts either way.

Segment kinds: ``r`` roman, ``i`` italic, ``s`` subscript roman,
``si`` subscript italic, ``u`` superscript roman.
"""

from __future__ import annotations

from dataclasses import dataclass

Segment = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Equation:
    number: int
    latex: str
    word: tuple[Segment, ...]

    @property
    def plain(self) -> str:
        return "".join(t for t, _ in self.word)


EQUATIONS: dict[int, Equation] = {
    1: Equation(
        1,
        r"z^{2}_{i,t+1} \;=\; \alpha_{i} + \delta_{t} + \beta\, s_{i,t} "
        r"+ \boldsymbol{\gamma}'\mathbf{x}_{i,t} + \varepsilon_{i,t}",
        (
            ("z", "i"), ("2", "u"), ("i,t+1", "si"), (" = ", "r"),
            ("α", "i"), ("i", "si"), (" + ", "r"),
            ("δ", "i"), ("t", "si"), (" + ", "r"),
            ("β", "i"), ("s", "i"), ("i,t", "si"), (" + ", "r"),
            ("γ", "i"), ("′", "r"), ("x", "i"), ("i,t", "si"), (" + ", "r"),
            ("ε", "i"), ("i,t", "si"),
        ),
    ),
    2: Equation(
        2,
        r"NF_{t} \;=\; \frac{\sum_{j} b_{j,t} - \sum_{j} s_{j,t}}{G_{t-1}}",
        (
            ("NF", "i"), ("t", "si"), (" = ( Σ", "r"),
            ("b", "i"), ("j,t", "si"), (" − Σ", "r"),
            ("s", "i"), ("j,t", "si"), (" ) / ", "r"),
            ("G", "i"), ("t−1", "si"),
        ),
    ),
    3: Equation(
        3,
        r"\left|r_{t+1}\right| \;=\; a + b\,NF_{t-2} + c\,NEG_{t-2} "
        r"+ \sum_{k=0}^{4}\varphi_{k}\left|r_{t-k}\right| + \psi\,V_{t} + u_{t}",
        (
            ("| ", "r"), ("r", "i"), ("t+1", "si"), (" | = ", "r"),
            ("a", "i"), (" + ", "r"),
            ("b", "i"), (" ", "r"), ("NF", "i"), ("t−2", "si"), (" + ", "r"),
            ("c", "i"), (" ", "r"), ("NEG", "i"), ("t−2", "si"), (" + Σ", "r"),
            ("φ", "i"), ("k", "si"), (" | ", "r"),
            ("r", "i"), ("t−k", "si"), (" | + ", "r"),
            ("ψ", "i"), ("V", "i"), ("t", "si"), (" + ", "r"),
            ("u", "i"), ("t", "si"),
        ),
    ),
    4: Equation(
        4,
        r"c^{\ast} \;=\; \frac{\mathbb{E}\left[\pi_{t}\right]}"
        r"{\mathbb{E}\left[TO_{t}\right]}",
        (
            ("c", "i"), ("*", "u"), (" = E[ ", "r"),
            ("π", "i"), ("t", "si"), (" ] / E[ ", "r"),
            ("TO", "i"), ("t", "si"), (" ]", "r"),
        ),
    ),
}
