"""Small dependency-free terminal table; ANSI only when explicitly enabled."""
def bar(fraction, plain=False):
    if fraction is None:
        return "--"
    fraction = min(1., max(0., fraction))
    done = int(round(8*fraction))
    fill, empty = ("#", "-") if plain else ("█", "░")
    return fill*done+empty*(8-done)+f" {100*fraction:3.0f}%"


def paint(text, color, enabled):
    codes = dict(header="1;36", normal="36", done="32", failed="1;31", prior="33")
    return f"\033[{codes.get(color,'0')}m{text}\033[0m" if enabled else text


def clipped(text, width, plain=False):
    if len(text) <= width:
        return text.ljust(width)
    return text[:width-3]+"..." if plain else text[:width-1]+"…"


def render(title, overview, rows, width=140, color=False, plain=False):
    # Fold the thermodynamic column below the table on narrow terminals.
    wide = width >= 125
    widths = [9,18,13,20,25,30] if wide else [8,14,13,19,20]
    headers = ["Host", "Phase", "Progress", "Elapsed / left ETA", "GPU / VRAM"]
    if wide: headers += ["Transfer dg (meV/H)"]
    total = sum(widths)+2*(len(widths)-1)
    if width < 85:
        widths = [8,13,7,17,19]; total=sum(widths)+8
        headers[2] = "%"
    edge = "+" if plain else "╔"
    right = "+" if plain else "╗"
    bottom, bottomright = ("+", "+") if plain else ("╚", "╝")
    horizontal = "-" if plain else "═"
    vertical = "|" if plain else "║"
    lines = [paint(edge+horizontal*total+right,"header",color),
             paint(vertical+clipped(title,total,plain)+vertical,"header",color),
             paint(bottom+horizontal*total+bottomright,"header",color), overview, "",
             "  ".join(clipped(t,w,plain) for t,w in zip(headers,widths)),
             ("-" if plain else "─")*total]
    for row in rows:
        progress = bar(row.get("fraction"), plain)
        if width < 85:
            progress = "--" if row.get("fraction") is None else f"{100*row['fraction']:.0f}%"
        cells = [row["name"],row["phase"],progress,row["timing"],row["gpu"]]
        if wide: cells.append(row["free_energy"])
        text = "  ".join(clipped(t,w,plain) for t,w in zip(cells,widths))
        role = "failed" if "FAIL" in row["phase"] else "done" if row["phase"] == "DONE" else "normal"
        lines.append(paint(text,role,color))
    if not wide:
        lines += ["", "  |  ".join(f"{r['name']}: {r['free_energy']}" for r in rows if r["free_energy"] != "H reference")]
    lines += ["", "ETA is for the current stage; -- means estimating. Prior = previous run, not a live update."]
    lines += ["Free-energy errors are statistical only; completion does not establish convergence."]
    return "\n".join(lines)
