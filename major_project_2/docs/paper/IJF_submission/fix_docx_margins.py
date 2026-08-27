"""Re-inject 1-inch page margins into pandoc-generated .docx files.

Pandoc writes its own <w:sectPr> and ignores the page margins set in the
reference document, so the IJF 1-inch requirement has to be applied after
conversion. Idempotent; safe to re-run.
"""
import os
import re
import zipfile

PGMAR = ('<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
         'w:header="720" w:footer="720" w:gutter="0"/>')


def fix(path: str) -> str:
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        data = {n: zin.read(n) for n in names}

    doc = data["word/document.xml"].decode("utf-8")
    if "<w:pgMar" in doc:
        doc, how = re.sub(r"<w:pgMar[^/]*/>", PGMAR, doc), "replaced"
    elif "<w:sectPr" in doc:
        doc = re.sub(r"(<w:sectPr[^>]*>)", r"\1" + PGMAR, doc, count=1)
        how = "injected"
    else:
        doc = doc.replace("</w:body>",
                          "<w:sectPr>" + PGMAR + "</w:sectPr></w:body>", 1)
        how = "added"
    data["word/document.xml"] = doc.encode("utf-8")

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, data[n])
    os.replace(tmp, path)
    return how


if __name__ == "__main__":
    for f in ("IJF_manuscript.docx", "IJF_cover_page.docx"):
        if os.path.exists(f):
            print(f"{f}: margins {fix(f)}")
