"""The attrition ladder, assembled.

Seven applications of the record plus one external control, each placed at the
LOWEST level it does not survive. Where an application could plausibly sit at
more than one level it is placed at the earliest point it demonstrably fails -
a judgement made by the people who ran it, and stated as such.
"""
from __future__ import annotations

import json

import pandas as pd

from src.common.utilities import local

APPLICATIONS = [
    ("A1", "Institutional share -> next-day stock volatility", "L2", "availability",
     "t = -5.53 contemporaneous, -0.43 at the deployable lag"),
    ("A2", "Market density, flow tilt on a GJR-GARCH base", "L3", "competition",
     "DM +0.28; the better-specified base already holds the information"),
    ("A4", "Episode-end hazard model", "L3", "competition",
     "real skill (AUC 0.657 test) but loses 31 bp to a three-line rule"),
    ("A5", "Cross-sectional concentration book", "L5", "execution",
     "breakeven 7.41 bp against a cost stack whose STT alone is 10 bp/side"),
    ("A6", "Participant-composition feature block", "L4", "detectability",
     "74.8 bp in the extremes, incremental IC 0.0012 against a 0.005 bar"),
    ("A7", "Flow-regime conditioning in the stock-day density", "L4", "detectability",
     "indistinguishable from its own ablation where it should bind hardest"),
    ("A8", "Aggregate flow -> market-level density (EWMA base)", "survives", "-",
     "clears all six; reported under five stated boundary conditions"),
    ("Cext", "Foreign index return -> domestic weekly volatility (CONTROL)", "L3",
     "competition", "+1.58 against a -2.0 gate; a NON-FLOW predictor, same failure"),
]

LADDER = [
    ("L0", "existence", "is the effect present under conventional inference?", 0),
    ("L1", "persistence", "does it hold on an era the design never touched?", 0),
    ("L2", "availability", "is it knowable when the decision must be made?", 1),
    ("L3", "competition", "does it beat the baseline it faces in use?", 2),
    ("L4", "detectability", "can the consuming metric resolve it?", 2),
    ("L5", "execution", "does the edge exceed the cost of capturing it?", 1),
]


def applications_table() -> pd.DataFrame:
    return pd.DataFrame(APPLICATIONS,
                        columns=["id", "application", "binding_level",
                                 "constraint", "evidence"])


def ladder_table() -> pd.DataFrame:
    rows, standing = [], 7
    for level, name, question, lost in LADDER:
        rows.append({"level": level, "constraint": name, "question": question,
                     "applications_lost": lost, "still_standing": standing - lost})
        standing -= lost
    return pd.DataFrame(rows)


def write_attrition_summary() -> str:
    apps = applications_table()
    ladder = ladder_table()

    apps.to_csv(local("tables") / "applications_by_binding_constraint.csv", index=False)
    ladder.to_csv(local("tables") / "attrition_ladder.csv", index=False)
    (local("machine") / "attrition.json").write_text(json.dumps(
        {"ladder": ladder.to_dict("records"),
         "applications": apps.to_dict("records")}, indent=2))

    lines = [f"{'level':<6}{'constraint':<16}{'lost':>6}{'standing':>10}"]
    for _, r in ladder.iterrows():
        lines.append(f"{r['level']:<6}{r['constraint']:<16}"
                     f"{r['applications_lost']:>6}{r['still_standing']:>10}")
    lines.append("")
    lines.append("Nothing is lost at L0 or L1. All attrition occurs under the")
    lines.append("operational constraints, and it is spread across FOUR levels -")
    lines.append("no single friction explains the pattern.")
    lines.append("")
    for _, r in apps.iterrows():
        lines.append(f"{r['id']:<5} {r['binding_level']:<9} {r['application']}")
    return "\n".join(lines)
