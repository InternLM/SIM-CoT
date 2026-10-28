#!/usr/bin/env python3
"""Render README artwork from verified paper results and the Figure 4 example.

Requires Pillow, NumPy and Matplotlib. Run from any directory. This animation
illustrates the published example; it is not a recorded model execution.
"""
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = json.loads((OUT / "source.json").read_text())
FONT = Path("/usr/share/fonts/truetype/lato")
MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
INK = "#f2f4ff"
MUTED = "#aebad3"
MINT = "#6ef0d2"
VIOLET = "#b7a5ff"
BLUE = "#8ab5ff"


def font(size, bold=False, mono=False):
    path = MONO if mono else FONT / ("Lato-Heavy.ttf" if bold else "Lato-Regular.ttf")
    return ImageFont.truetype(str(path), size)


def text(d, xy, value, size=30, fill=INK, bold=False, mono=False, anchor=None):
    d.text(xy, value, font=font(size, bold, mono), fill=fill, anchor=anchor)


def backdrop(w, h):
    yy, xx = np.mgrid[:h, :w]
    base = np.empty((h, w, 3), dtype=np.float64)
    base[:] = (10, 16, 33)
    for cx, cy, radius, color, strength in [
        (w * .04, h * .15, w * .56, (78, 48, 125), .7),
        (w * .92, h * .34, w * .38, (15, 85, 95), .58),
    ]:
        a = np.exp(-((xx-cx)**2+(yy-cy)**2)/(radius**2))[:, :, None] * strength
        base += a * np.array(color)
    im = Image.fromarray(np.clip(base, 0, 255).astype("uint8"), "RGB")
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(0, w, 80):
        d.line((x, 0, x, h), fill=(190, 210, 255, 10), width=1)
    for y in range(0, h, 80):
        d.line((0, y, w, y), fill=(190, 210, 255, 10), width=1)
    return im


def card(d, box, outline="#394562", fill="#152239", radius=22, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(d, start, end, color=MUTED, width=3):
    d.line((*start, *end), fill=color, width=width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    p1 = (end[0]-14*math.cos(a-.48), end[1]-14*math.sin(a-.48))
    p2 = (end[0]-14*math.cos(a+.48), end[1]-14*math.sin(a+.48))
    d.polygon([end, p1, p2], fill=color)


def dashed(d, start, end, color=MUTED, width=2, dash=10):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = math.hypot(dx, dy)
    for a in range(0, int(length), dash*2):
        b = min(a+dash, length)
        d.line((start[0]+dx*a/length, start[1]+dy*a/length,
                start[0]+dx*b/length, start[1]+dy*b/length), fill=color, width=width)


def latent(im, center, label, color, active=True, size=86):
    d = ImageDraw.Draw(im, "RGBA")
    x, y = center
    if active:
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        for r in range(size+26, size, -3):
            d.ellipse((x-r, y-r, x+r, y+r), fill=(*rgb, 3))
    r = size/2
    d.rounded_rectangle((x-r, y-r, x+r, y+r), radius=22,
                        fill="#202d49" if active else "#152037",
                        outline=color if active else "#44516b", width=3)
    text(d, (x, y), label, 35, color if active else "#66728b", True, anchor="mm")


def hero():
    im = backdrop(1600, 820)
    d = ImageDraw.Draw(im, "RGBA")
    card(d, (64, 54, 275, 104), VIOLET, "#30294d", 25)
    text(d, (170, 79), "ICLR 2026", 26, VIOLET, True, anchor="mm")
    text(d, (304, 80), "OFFICIAL IMPLEMENTATION", 22, MUTED, anchor="lm")
    text(d, (64, 135), "SIM", 110, MINT, True)
    text(d, (258, 135), "-CoT", 110, INK, True)
    text(d, (67, 270), "Supervised Implicit Chain-of-Thought", 31, MUTED)
    text(d, (64, 344), "Give every latent", 57, INK, True)
    text(d, (64, 410), "a reasoning step.", 57, INK, True)
    text(d, (67, 508), "Train with step supervision.", 32, MUTED)
    text(d, (67, 551), "Reason without the auxiliary decoder.", 32, MUTED)
    text(d, (1180, 168), "A STRUCTURED REASONING SPACE", 24, MUTED, anchor="mm")
    positions = [(908, 292), (1180, 292), (1452, 292)]
    for i, (x, y) in enumerate(positions):
        if i < 2:
            arrow(d, (x+53, y), (positions[i+1][0]-53, y), "#7887aa", 3)
    for i, (x, y) in enumerate(positions):
        c = [MINT, BLUE, VIOLET][i]
        latent(im, (x, y), f"z{i+1}", c, size=96)
        d = ImageDraw.Draw(im, "RGBA")
        dashed(d, (x, y+56), (x, 424), c)
        card(d, (x-104, 432, x+104, 510), c, "#1b2943", 16)
        text(d, (x, 471), f"Reasoning step {i+1}", 25, c, True, anchor="mm")
    text(d, (1180, 552), "Auxiliary decoder · during training", 29, MUTED, anchor="mm")
    d.line((65, 632, 1535, 632), fill="#4a526c", width=2)
    metrics = [
        (66, "+8.2 pp", "GPT-2 · Coconut", "GSM8k-Aug accuracy gain"),
        (590, "+3.0 pp", "LLaMA 3.1 8B · CODI", "GSM8k-Aug accuracy gain"),
        (1140, "0", "Auxiliary decoder at inference", "Same implicit inference path"),
    ]
    for x, value, label, sub in metrics:
        text(d, (x, 659), value, 62, MINT if x != 590 else VIOLET, True)
        text(d, (x, 735), label, 25, INK, True)
        text(d, (x, 772), sub, 23, MUTED)
    im.save(OUT / "hero.png", optimize=True)


def walkthrough_frame(phase):
    im = backdrop(1440, 760)
    d = ImageDraw.Draw(im, "RGBA")
    text(d, (48, 32), "THINK IN LATENTS. READ EVERY STEP.", 37, INK, True)
    text(d, (48, 89), "EXPLANATORY ANIMATION  /  FIGURE 4 + APPENDIX I.1", 22, MUTED)
    card(d, (45, 153, 435, 617), "#465574", "#19273e", 24)
    text(d, (75, 184), "THE QUESTION", 22, MINT, True)
    lines = ["120 watermelons.", "30% harvested first.", "Two weeks later,", "3/4 of the remaining", "melons are harvested.", "", "How many are left?"]
    for i, line in enumerate(lines):
        text(d, (75, 231+i*45), line, 28, INK if i in (0, 6) else MUTED, i in (0, 6))
    stage = "01 / ENCODE QUESTION" if phase == 0 else "02 / IMPLICIT INFERENCE" if phase <= 5 else "03 / OPTIONAL INTERPRETATION"
    text(d, (492, 155), stage, 25, MINT if phase <= 5 else VIOLET, True)
    text(d, (492, 198), "The trained LLM reasons through latent states.", 26, MUTED)
    arrow(d, (435, 327), (516, 327), BLUE)
    card(d, (523, 265, 679, 389), BLUE, "#1e2b47", 18)
    text(d, (601, 314), "LLM", 32, BLUE, True, anchor="mm")
    text(d, (601, 355), "prompt", 21, MUTED, anchor="mm")
    centers = [(766, 327), (906, 327), (1046, 327), (1186, 327)]
    colors = [MINT, BLUE, VIOLET, "#f4c68e"]
    arrow(d, (679, 327), (719, 327), MUTED)
    for i, p in enumerate(centers):
        if i < 3: arrow(d, (p[0]+47, 327), (centers[i+1][0]-47, 327), MUTED)
    arrow(d, (1230, 327), (1247, 327), MUTED)
    for i, p in enumerate(centers):
        latent(im, p, f"z{i+1}", colors[i], phase >= i+1, size=80)
    d = ImageDraw.Draw(im, "RGBA")
    card(d, (1250, 266, 1390, 389), MINT if phase >= 5 else "#44516b", "#1e2b47", 18)
    text(d, (1320, 292), "ANSWER", 19, MUTED, True, anchor="mm")
    text(d, (1320, 342), "21" if phase >= 5 else "…", 46, MINT if phase >= 5 else "#66728b", True, anchor="mm")
    if phase <= 5:
        card(d, (491, 456, 1390, 616), "#3c4a68", "#142138", 20)
        text(d, (522, 490), "INFERENCE", 22, MINT, True)
        text(d, (522, 534), "Question → latent reasoning → answer", 29, INK, True)
        text(d, (522, 579), "The auxiliary decoder is removed.", 26, MUTED)
    else:
        card(d, (491, 434, 1390, 619), VIOLET, "#25233f", 20)
        text(d, (520, 453), "OPTIONAL DECODER  /  PUBLISHED DECODED STEPS", 22, VIOLET, True)
        coords = [(520, 528, "0.3 × 120 = 36"), (960, 528, "120 − 36 = 84"),
                  (520, 575, "3/4 × 84 = 63"), (960, 575, "84 − 63 = 21")]
        for i, (x, y, value) in enumerate(coords):
            if phase >= 6+i:
                text(d, (x, y), f"z{i+1}  ", 23, colors[i], True, mono=True)
                text(d, (x+55, y-1), value, 24, INK, mono=True)
        for i, (x, _) in enumerate(centers):
            if phase >= 6+i:
                dashed(d, (x, 370), (x, 432), colors[i])
    steps = ["Encode", "Reason in latents", "Answer", "Decode on demand"]
    current = 0 if phase == 0 else 1 if phase <= 4 else 2 if phase == 5 else 3
    for i, name in enumerate(steps):
        x = 52+i*345
        d.ellipse((x, 664, x+26, 690), fill=MINT if i <= current else "#44516b")
        text(d, (x+41, 663), name, 25, INK if i <= current else MUTED, i == current)
    text(d, (48, 716), "Schematic of the paper example · Steps are decoded for inspection, not emitted during ordinary inference.", 21, MUTED)
    return im


def walkthrough():
    frames = [walkthrough_frame(i) for i in range(10)]
    palette = frames[-1].quantize(colors=128).getpalette()
    palette_im = Image.new("P", (1, 1)); palette_im.putpalette(palette)
    quantized = [im.quantize(palette=palette_im, dither=Image.Dither.NONE) for im in frames]
    quantized[0].save(OUT / "latent-walkthrough.gif", save_all=True,
                      append_images=quantized[1:], duration=[1600, 900, 900, 900, 900, 1800, 1100, 1100, 1100, 4000],
                      loop=0, optimize=False, disposal=2)
    frames[-1].save(OUT / "latent-walkthrough.png", optimize=True)


def results():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14})
    fig, ax = plt.subplots(figsize=(14, 8.6), dpi=140)
    fig.patch.set_facecolor("#111b30"); ax.set_facecolor("#111b30")
    fig.subplots_adjust(left=.31, right=.9, top=.78, bottom=.2)
    rows = SOURCE["results"]
    for i, row in enumerate(rows):
        y = 3-i
        ax.barh(y+.18, row["baseline_accuracy"], height=.28, color="#8695bb", zorder=3)
        ax.barh(y-.18, row["simcot_accuracy"], height=.28, color=MINT, zorder=3)
        for yy, key, c in [(y+.18,"baseline_accuracy","#d0d8ec"),(y-.18,"simcot_accuracy",MINT)]:
            ax.text(row[key]+1.2, yy, f'{row[key]:.1f}', color=c, va="center", fontsize=15, weight="bold")
        ax.text(108, y, f'+{row["simcot_accuracy"]-row["baseline_accuracy"]:.1f} pp',
                va="center", ha="right", fontsize=17, color=MINT, weight="bold")
    ax.set_yticks(range(4), [f'{r["backbone"]}\n{r["method"]}' for r in reversed(rows)], color=INK, fontsize=15)
    ax.set_xlim(0, 109); ax.set_ylim(-.65, 3.65)
    ax.set_xticks([0,20,40,60,80,100]); ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0, pad=18)
    ax.set_xlabel("GSM8k-Aug accuracy (%)", color=MUTED, labelpad=12, fontsize=14)
    ax.grid(axis="x", color="#33415a", linewidth=.8, alpha=.75, zorder=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.055,.93,"Better reasoning. Same inference budget.", fontsize=25, weight="bold", color=INK)
    fig.text(.055,.873,"GSM8k-Aug · Matched comparisons from paper Tables 1–3", fontsize=15, color=MUTED)
    fig.text(.55,.82,"■  Original baseline", color="#8695bb", fontsize=14)
    fig.text(.76,.82,"■  + SIM-CoT", color=MINT, fontsize=14)
    fig.text(.055,.082,"GPT-2 + Coconut: 44.8% accuracy with 12.2 average tokens,", color=INK, fontsize=15, weight="bold")
    fig.text(.055,.04,"vs. SFT-CoT: 42.7% with 27.6 tokens (2.3× token efficiency).", color=MUTED, fontsize=13)
    fig.savefig(OUT / "results-overview.png", facecolor=fig.get_facecolor())
    fig.savefig(OUT / "results-overview.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    hero()
    walkthrough()
    results()
    for name in ["hero.png", "latent-walkthrough.gif", "latent-walkthrough.png", "results-overview.png"]:
        print(name, (OUT / name).stat().st_size)
