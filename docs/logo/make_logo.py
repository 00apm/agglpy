"""Generate the agglpy logo SVGs in this folder.

Usage: python make_logo.py path/to/IBMPlexSans-SemiBold.ttf

The font (SIL Open Font License) is not kept in the repository:
https://github.com/IBM/plex (packages/plex-sans/fonts/complete/ttf).
Needs fontTools (installed with matplotlib).

Every particle is an SVG <circle>; every neck is a separate <path> under
the circles: two arcs of a fillet circle tangent to both particles,
closed through the particle centres (hidden by the circles). The
wordmark is converted to outlines, so the files need no font.
"""

import math
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).parent

# name -> (cx, cy, r). Agglomerate A-E; F and G isolated.
PARTICLES = {
    "A": (393.0, 385.0, 72.0),
    "B": (516.0, 350.0, 32.0),
    "C": (540.0, 440.0, 65.0),
    "D": (410.0, 546.0, 80.0),
    "E": (275.0, 569.0, 47.0),
    "F": (282.0, 466.0, 28.0),
    "G": (556.0, 592.0, 46.0),
}
# (particle, particle, width at the waist). B and C overlap: no neck.
NECKS = [("A", "D", 40.0), ("A", "C", 24.0), ("D", "E", 28.0)]
ACCENT = {"G"}

# Icon for 16-32 px: fewer particles, wider necks.
ICON_PARTICLES = ["A", "C", "D", "G"]
ICON_NECKS = [("A", "D", 48.0), ("A", "C", 38.0)]

# theme -> (main colour, accent colour)
THEMES = {
    "light": ("#30343B", "#D98236"),
    "dark": ("#ECEAE6", "#EC9550"),
}
WORD = "agglpy"
ACCENT_FROM = 4  # "py" in the accent colour
FONT_SIZE = 228.0


def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fillet(c1, r1, c2, r2, rho):
    """Centres of the fillet circles on both sides and their offset."""
    d = math.dist(c1, c2)
    ux, uy = (c2[0] - c1[0]) / d, (c2[1] - c1[1]) / d
    a, b = r1 + rho, r2 + rho
    x = (d * d + a * a - b * b) / (2 * d)
    h = math.sqrt(max(a * a - x * x, 0.0))
    bx, by = c1[0] + ux * x, c1[1] + uy * x
    return (bx - uy * h, by + ux * h), (bx + uy * h, by - ux * h), h


def solve_rho(c1, r1, c2, r2, waist):
    """Fillet radius that gives the neck this width at its waist."""
    lo = max((math.dist(c1, c2) - r1 - r2) / 2, 0.0) + 1e-6
    hi = 10 * (r1 + r2)
    for _ in range(200):
        mid = (lo + hi) / 2
        if 2 * (fillet(c1, r1, c2, r2, mid)[2] - mid) < waist:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def touch(c, r, p, rho):
    """Point where the fillet circle at p touches circle (c, r)."""
    k = r / (r + rho)
    return (c[0] + (p[0] - c[0]) * k, c[1] + (p[1] - c[1]) * k)


def sweep(p, start, end):
    """SVG sweep flag of the short arc around p."""
    a1 = math.atan2(start[1] - p[1], start[0] - p[0])
    a2 = math.atan2(end[1] - p[1], end[0] - p[0])
    return 1 if (a2 - a1 + math.pi) % (2 * math.pi) - math.pi > 0 else 0


def neck_path(p, q, waist):
    x1, y1, r1 = PARTICLES[p]
    x2, y2, r2 = PARTICLES[q]
    c1, c2 = (x1, y1), (x2, y2)
    rho = solve_rho(c1, r1, c2, r2, waist)
    pa, pb, _ = fillet(c1, r1, c2, r2, rho)
    t1a, t2a = touch(c1, r1, pa, rho), touch(c2, r2, pa, rho)
    t1b, t2b = touch(c1, r1, pb, rho), touch(c2, r2, pb, rho)

    def pt(v):
        return f"{fmt(v[0])} {fmt(v[1])}"

    arc = f"A{fmt(rho)} {fmt(rho)} 0 0"
    return (
        f"M{pt(t1a)}{arc} {sweep(pa, t1a, t2a)} {pt(t2a)}"
        f"L{pt(c2)}L{pt(t2b)}{arc} {sweep(pb, t2b, t1b)} {pt(t1b)}"
        f"L{pt(c1)}Z"
    )


def symbol(names, necks, main, accent):
    lines = [f'    <g id="necks" fill="{main}">']
    for p, q, w in necks:
        d = neck_path(p, q, w)
        lines.append(f'      <path id="neck-{p}{q}" d="{d}"/>')
    lines += ["    </g>", f'    <g id="particles" fill="{main}">']
    for n in names:
        x, y, r = PARTICLES[n]
        fill = f' fill="{accent}"' if n in ACCENT else ""
        lines.append(
            f'      <circle id="particle-{n}" cx="{fmt(x)}" cy="{fmt(y)}"'
            f' r="{fmt(r)}"{fill}/>'
        )
    lines.append("    </g>")
    return ['  <g id="symbol">', *lines, "  </g>"]


def bbox(names):
    ps = [PARTICLES[n] for n in names]
    return (
        min(x - r for x, _, r in ps),
        min(y - r for _, y, r in ps),
        max(x + r for x, _, r in ps),
        max(y + r for _, y, r in ps),
    )


def glyphs(font, x, baseline):
    """Outlines of WORD as path data, and the x where the word ends."""
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = FONT_SIZE / font["head"].unitsPerEm
    out = []
    for ch in WORD:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyph_set)
        transform = (scale, 0, 0, -scale, x, baseline)
        glyph_set[name].draw(TransformPen(pen, transform))
        out.append((ch, pen.getCommands()))
        x += font["hmtx"][name][0] * scale
    return out, x


def svg(view_box, body):
    vb = " ".join(fmt(v) for v in view_box)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"'
            ' role="img" aria-labelledby="title">',
            '  <title id="title">agglpy</title>',
            *body,
            "</svg>",
            "",
        ]
    )


def logo(font, main, accent):
    x0, y0, x1, y1 = bbox(PARTICLES)
    # Centre the x-height band on the symbol: lowercase reads as the body.
    x_height = font["OS/2"].sxHeight * FONT_SIZE / font["head"].unitsPerEm
    baseline = (y0 + y1) / 2 + x_height / 2
    word, x_end = glyphs(font, x1 + 0.24 * (y1 - y0), baseline)
    body = symbol(PARTICLES, NECKS, main, accent)
    body += ['  <g id="wordmark">', f'    <g id="aggl" fill="{main}">']
    for i, (ch, d) in enumerate(word):
        if i == ACCENT_FROM:
            body += ["    </g>", f'    <g id="py" fill="{accent}">']
        body.append(f'      <path id="glyph-{i}-{ch}" d="{d}"/>')
    body += ["    </g>", "  </g>"]
    pad = 24.0
    view_box = (x0 - pad, y0 - pad, x_end - x0 + 2 * pad, y1 - y0 + 2 * pad)
    return svg(view_box, body)


def square(names, necks, main, accent, pad):
    x0, y0, x1, y1 = bbox(names)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    view_box = (cx - side / 2, cy - side / 2, side, side)
    return svg(view_box, symbol(names, necks, main, accent))


def main():
    font = TTFont(sys.argv[1])
    for theme, (m, a) in THEMES.items():
        files = {
            "logo": logo(font, m, a),
            "mark": square(PARTICLES, NECKS, m, a, 0.04),
            "icon": square(ICON_PARTICLES, ICON_NECKS, m, a, 0.02),
        }
        for kind, text in files.items():
            path = HERE / f"agglpy-{kind}-{theme}.svg"
            path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
