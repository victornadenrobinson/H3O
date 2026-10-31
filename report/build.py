"""Build report/paper.pdf (reportlab) and report/paper.tex (revtex4-2, for Overleaf) from report/paper_text.py.

    python3 report/make_figures.py && python3 report/build.py

The PDF is a single-column rendering; LaTeX was not available in the build environment, so the .tex has not been compiled here.
"""
import io
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_text as T

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TTF = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

# ---------------------------------------------------------------- fonts
for name, f in [("Serif", "DejaVuSerif.ttf"), ("Serif-B", "DejaVuSerif-Bold.ttf"), ("Serif-I", "DejaVuSerif-Italic.ttf"),
                ("Serif-BI", "DejaVuSerif-BoldItalic.ttf"), ("Sans", "DejaVuSans.ttf"), ("Sans-B", "DejaVuSans-Bold.ttf"),
                ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(TTF, f)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans", boldItalic="Sans-B")

# ---------------------------------------------------------------- LaTeX-lite -> reportlab markup
SYM = {"approx": "≈", "le": "≤", "ge": "≥", "to": "→", "lambda": "λ", "Lambda": "Λ", "mu": "μ", "sim": "∼", "lesssim": '<font name="Sans">≲</font>',
       "pm": "±", "in": "∈", "propto": "∝", "Delta": "Δ", "cdot": "·", "times": "×", "infty": "∞", "gamma": "γ", "ln": "ln",
       "AA": "Å", "pi": "π"}
SPACE = {",": "\u2009", ";": "\u2005", " ": " ", "!": ""}
esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def group(s, i):
    """s[i] is '{': return (content, index after the matching '}')."""
    depth, j = 0, i
    while j < len(s):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces: " + s)


def math(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                cmd = m.group(1)
                i += len(m.group(0))
                if cmd in ("text", "mathrm") and i < len(s) and s[i] == "{":
                    g, i = group(s, i)
                    out.append(text(g))
                elif cmd in ("left", "right"):
                    pass
                elif cmd == "sqrt" and s[i] == "{":
                    g, i = group(s, i)
                    out.append("√(" + math(g) + ")")
                else:
                    out.append(SYM.get(cmd, cmd))
            else:
                out.append(SPACE.get(s[i + 1], s[i + 1]))
                i += 2
        elif c in "_^":
            i += 1
            if s[i] == "{":
                g, i = group(s, i)
            elif s[i] == "\\":
                m = re.match(r"\\[A-Za-z]+", s[i:])
                g, i = m.group(0), i + len(m.group(0))
            else:
                g, i = s[i], i + 1
            tag = "sub" if c == "_" else "super"
            out.append(f"<{tag}>{math(g)}</{tag}>")
        elif c.isalpha():
            out.append(f"<i>{c}</i>")
            i += 1
        elif c == "~":
            out.append("\u00a0")
            i += 1
        elif c in "{}":
            i += 1
        else:
            out.append(esc(c) if c != "-" else "−")
            i += 1
    return "".join(out)


def text(s):
    """Text-mode LaTeX-lite, with $...$ math segments (scanned together so commands may contain math)."""
    return _plain(s)


def _plain(s):
    out, i = [], 0
    while i < len(s):
        if s.startswith("---", i):
            out.append("—"); i += 3
        elif s.startswith("--", i):
            out.append("–"); i += 2
        elif s.startswith("``", i):
            out.append("“"); i += 2
        elif s.startswith("''", i):
            out.append("”"); i += 2
        elif s[i] == "$":
            j = s.index("$", i + 1)
            out.append(math(s[i + 1:j])); i = j + 1
        elif s[i] == "~":
            out.append("\u00a0"); i += 1
        elif s[i] == "\\":
            m = re.match(r"\\([A-Za-z]+)\s*", s[i:])
            if m and m.group(1) in ("textbf", "textit", "emph", "texttt", "cite", "ref", "note", "text"):
                cmd, i = m.group(1), i + len(m.group(0))
                g, i = group(s, i)
                if cmd == "textbf":
                    out.append(f"<b>{text(g)}</b>")
                elif cmd in ("textit", "emph"):
                    out.append(f"<i>{text(g)}</i>")
                elif cmd == "texttt":
                    out.append(f'<font name="Mono" size="-1">{esc(g.replace(chr(92) + "_", "_"))}</font>')
                elif cmd == "text":
                    out.append(text(g))
                elif cmd == "cite":
                    nums = [str(REFN.setdefault(k, len(REFN) + 1)) for k in g.split(",")]
                    out.append(f"<super>{','.join(nums)}</super>")
                elif cmd == "ref":
                    out.append(str(FIGN[g.split(":")[1]]))
                elif cmd == "note":
                    out.append(f'&nbsp;<font color="#c0392b" size="-1">[{text(g)}]</font> ')
            elif m and m.group(1) == "AA":
                out.append("Å"); i += len(m.group(0))
            elif s[i + 1] == "%":
                out.append("%"); i += 2
            elif s[i + 1] == "_":
                out.append("_"); i += 2
            elif s[i + 1] in ",;":
                out.append(SPACE[s[i + 1]]); i += 2
            elif s[i + 1] == " ":
                out.append(" "); i += 2
            else:
                out.append(esc(s[i])); i += 1
        else:
            out.append(esc(s[i])); i += 1
    return "".join(out)


REFN = {}
FIGN = {}
n = 0
for b in T.BLOCKS:
    if b[0] == "fig":
        n += 1
        FIGN[b[1]] = n

# ---------------------------------------------------------------- equation images (matplotlib mathtext)
def eq_png(line, path):
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${line}$", fontsize=11, usetex=False)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------- styles
INK = colors.HexColor("#0b0b0b")
S = {
    "title": ParagraphStyle("title", fontName="Serif-B", fontSize=17, leading=21, alignment=TA_LEFT, spaceAfter=6),
    "author": ParagraphStyle("author", fontName="Serif", fontSize=11, leading=14),
    "affil": ParagraphStyle("affil", fontName="Serif-I", fontSize=8.5, leading=11, textColor=colors.HexColor("#52514e")),
    "abs": ParagraphStyle("abs", fontName="Serif-B", fontSize=8.8, leading=11.8, alignment=TA_JUSTIFY),
    "h": ParagraphStyle("h", fontName="Sans-B", fontSize=10.5, leading=13, spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Serif", fontSize=9.3, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5),
    "li": ParagraphStyle("li", fontName="Serif", fontSize=9.3, leading=12.6, alignment=TA_JUSTIFY, leftIndent=16, firstLineIndent=-16, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="Serif", fontSize=7.8, leading=10.2, alignment=TA_JUSTIFY),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=7.8, leading=10, leftIndent=14, firstLineIndent=-14, spaceAfter=2),
    "nh": ParagraphStyle("nh", fontName="Sans-B", fontSize=8.6, leading=11, spaceBefore=6, spaceAfter=1, keepWithNext=1),
    "nb": ParagraphStyle("nb", fontName="Sans", fontSize=7.6, leading=10, alignment=TA_JUSTIFY),
}
W, H = A4
LM = RM = 20 * mm
CW = W - LM - RM


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Sans", 6.8)
    canv.setFillColor(colors.HexColor("#52514e"))
    canv.drawString(LM, 11 * mm, "Draft rendering of Argon's write-up with repository data; panels marked DATA NOT IN REPO are placeholders.")
    canv.drawRightString(W - RM, 11 * mm, f"{doc.page}")
    canv.restoreState()


def image(path, width):
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)


def build_pdf(path):
    doc = BaseDocTemplate(path, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=18 * mm, bottomMargin=18 * mm,
                          title=T.TITLE, author=T.AUTHOR)
    doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(LM, 18 * mm, CW, H - 36 * mm, id="f", leftPadding=0, rightPadding=0,
                                                         topPadding=0, bottomPadding=0)], onPage=footer)])
    fl = [Paragraph(esc(T.TITLE), S["title"]), Paragraph(T.AUTHOR, S["author"]), Paragraph(esc(T.AFFIL), S["affil"]), Spacer(1, 6)]
    box = Table([[Paragraph(text(T.ABSTRACT), S["abs"])]], colWidths=[CW])
    box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f3ef")), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    fl += [box, Spacer(1, 4)]
    figno = 0
    for b in T.BLOCKS:
        k = b[0]
        if k == "section":
            fl.append(Paragraph(esc(b[1]), S["h"]))
        elif k == "p":
            fl.append(Paragraph(text(b[1]), S["p"]))
        elif k == "enum":
            for i, it in enumerate(b[1], 1):
                fl.append(Paragraph(f"{i}.&nbsp;&nbsp;" + text(it), S["li"]))
        elif k == "eq":
            rows = []
            for j, line in enumerate(b[1]):
                pth = os.path.join(FIG, f"_eq_{b[2]}_{j}.png")
                eq_png(line, pth)
                im = image(pth, min(CW * 0.9, PILImage.open(pth).size[0] / 300 * 72))
                rows.append([im, Paragraph("(1)" if j == 0 else "", S["p"])])
            t = Table(rows, colWidths=[CW - 34, 34])
            t.setStyle(TableStyle([("ALIGN", (0, 0), (0, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
            fl += [Spacer(1, 2), t, Spacer(1, 3)]
        elif k == "fig":
            figno += 1
            cap = Paragraph(f"<b>Fig. {figno} |</b> " + text(b[3]), S["cap"])
            fl.append(KeepTogether([Spacer(1, 4), image(os.path.join(FIG, b[2] + ".png"), CW), Spacer(1, 3), cap, Spacer(1, 8)]))
    # extended data + references
    name, cap = T.EDFIG
    fl.append(KeepTogether([Spacer(1, 4), image(os.path.join(FIG, name + ".png"), CW), Spacer(1, 3), Paragraph(text(cap), S["cap"]), Spacer(1, 8)]))
    fl.append(Paragraph("References", S["h"]))
    for k, i in sorted(REFN.items(), key=lambda kv: kv[1]):
        fl.append(Paragraph(f"{i}.&nbsp;&nbsp;{esc(T.BIB[k])}", S["ref"]))
    fl.append(Paragraph("<i>Reference details were filled in from memory; verify before use.</i>", S["ref"]))
    # appendix
    fl += [PageBreak(), Paragraph("Appendix: data provenance and open issues (not part of the manuscript)", S["h"]),
           Paragraph(esc(T.NOTES_INTRO), S["nb"]), Spacer(1, 6), Paragraph("Figure data coverage", S["nh"])]
    rows = [[Paragraph(f"<b>{esc(a)}</b>", S["nb"]), Paragraph(esc(b), S["nb"])] for a, b in T.COVERAGE]
    t = Table(rows, colWidths=[24 * mm, CW - 24 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dcdbd6")),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    fl.append(t)
    fl.append(Paragraph("Where the draft and the repository notes differ", S["nh"]))
    for a, h, body in T.NOTES:
        fl.append(Paragraph(f"{a}. {esc(h)}", S["nh"]))
        fl.append(Paragraph(esc(body), S["nb"]))
    doc.build(fl)


# ---------------------------------------------------------------- LaTeX
PRE = r"""\documentclass[twocolumn,secnumarabic,amssymb,nobibnotes,aps,prl,superscriptaddress,10pt]{revtex4-2}

\usepackage{graphicx}
\usepackage{amsmath,amssymb,bm}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!70!black,citecolor=blue!70!black,urlcolor=blue!70!black]{hyperref}

% Nature-style figure labels
\renewcommand{\figurename}{\textbf{Fig.}}
\renewcommand{\thefigure}{\textbf{\arabic{figure}}}

% editorial check marks: where the draft differs from the repository notes (see report/README.md, Appendix)
\newcommand{\note}[1]{{\color{red}\small[#1]}}

\begin{document}
"""


def build_tex(path):
    o = [PRE, f"\\title{{{T.TITLE}}}\n\n\\author{{{T.AUTHOR}}}\n\\affiliation{{{T.AFFIL}}}\n\n\\date{{\\today}}\n",
         "\\begin{abstract}\n\\noindent\\textbf{" + T.ABSTRACT + "}\n\\end{abstract}\n\n\\maketitle\n"]
    for b in T.BLOCKS:
        k = b[0]
        if k == "section":
            o.append(f"\\section*{{{b[1]}}}\n")
        elif k == "p":
            o.append(b[1] + "\n")
        elif k == "enum":
            o.append("\\begin{enumerate}\n" + "".join(f"    \\item {it}\n" for it in b[1]) + "\\end{enumerate}\n")
        elif k == "eq":
            l1, l2 = b[1]
            l1 = l1.replace(" = ", " &= ", 1)
            o.append(f"\\begin{{align}}\n{l1}\\nonumber\\\\\n&{l2}\n\\label{{eq:{b[2]}}}\n\\end{{align}}\n")
        elif k == "fig":
            o.append(f"\\begin{{figure*}}[t!]\n    \\centering\n    \\includegraphics[width=0.95\\textwidth]{{figures/{b[2]}.pdf}}\n"
                     f"    \\caption{{{b[3]}}}\n    \\label{{fig:{b[1]}}}\n\\end{{figure*}}\n")
    name, cap = T.EDFIG
    o.append(f"\\begin{{figure*}}[t!]\n    \\centering\n    \\includegraphics[width=0.8\\textwidth]{{figures/{name}.pdf}}\n    \\caption{{{cap}}}\n\\end{{figure*}}\n")
    o.append("\\begin{thebibliography}{99}\n% details filled in from memory: verify\n" +
             "".join(f"\\bibitem{{{k}}} {v}\n" for k, v in sorted(REFN.items(), key=lambda kv: kv[1])) + "\\end{thebibliography}\n\n\\end{document}\n")
    open(path, "w").write("\n".join(o))


if __name__ == "__main__":
    build_pdf(os.path.join(HERE, "paper.pdf"))   # fills REFN as a side effect
    build_tex(os.path.join(HERE, "paper.tex"))
    for f in os.listdir(FIG):
        if f.startswith("_eq_"):
            os.remove(os.path.join(FIG, f))
    print("built paper.pdf, paper.tex")
