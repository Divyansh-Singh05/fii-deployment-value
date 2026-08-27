"""Make page 1's media box match the rest of the document.

WHY THIS EXISTS. `IEEEtran.cls` sets the PDF media box only when `\\pdfoutput`
is defined - that is, under pdfLaTeX (lines 546-551). Under XeTeX, which is
what tectonic runs, it falls back to `\\AtBeginDvi{\\special{papersize=...}}`.
xdvipdfmx fixes page 1's media box before that special takes effect, so page 1
keeps the driver default (US Letter, 612x792 pt) while pages 2+ get the class
trim size (203.2 x 276.2 mm = 576 x 782.93 pt).

This is a property of the class and the engine, not of the manuscript. It is
present in every build of this template made with tectonic.

The proper fix is to compile with pdfLaTeX. Where that is unavailable, this
script repairs the output: it crops page 1 to the trim size and shifts its
content down by the height difference, because TeX anchors its reference point
1 inch from the top of the PHYSICAL page - so page 1's content sits 9.07 pt
higher than it would on a correctly sized page. Cropping alone would leave
page 1 with a visibly smaller top margin.

Usage:  python normalize_pagesize.py IEEE_Access_manuscript.pdf
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject


def main(path: str) -> int:
    src = Path(path)
    reader = PdfReader(src)

    sizes = [(round(float(p.mediabox.width), 2), round(float(p.mediabox.height), 2))
             for p in reader.pages]
    target, n = Counter(sizes).most_common(1)[0]
    odd = [i for i, s in enumerate(sizes) if s != target]
    if not odd:
        print(f"all {len(sizes)} pages already {target[0]} x {target[1]} pt")
        return 0

    print(f"target {target[0]} x {target[1]} pt ({n} pages); "
          f"repairing page(s) {[i + 1 for i in odd]}")

    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        if i in odd:
            dy = float(page.mediabox.height) - target[1]
            if dy:
                # TeX anchors 1in from the physical top edge, so a taller
                # canvas places content higher. Shift it back down.
                page.add_transformation((1, 0, 0, 1, 0, -dy))
            page.mediabox = RectangleObject((0, 0, target[0], target[1]))
            for box in ("/CropBox", "/TrimBox", "/BleedBox", "/ArtBox"):
                if box in page:
                    page[box] = RectangleObject((0, 0, target[0], target[1]))
        writer.add_page(page)

    out = src.with_name(src.stem + "_uniform.pdf")
    with out.open("wb") as fh:
        writer.write(fh)
    after = [(round(float(p.mediabox.width), 2), round(float(p.mediabox.height), 2))
             for p in PdfReader(out).pages]
    print(f"wrote {out.name}: {Counter(after).most_common()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "IEEE_Access_manuscript.pdf"))
