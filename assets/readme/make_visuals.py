#!/usr/bin/env python3
"""Render paper-style README artwork from the official figures and results.

Requires Pillow and Matplotlib. The animation illustrates the published
Figure 4 / Appendix I.1 example; it is not a recorded model execution.
"""
from pathlib import Path
import json
import math
from PIL import Image, ImageDraw, ImageFont
from make_results import results

OUT = Path(__file__).resolve().parent
SOURCE = json.loads((OUT / 'source.json').read_text())
FONT = Path('/usr/share/fonts/truetype/lato')
TIMES = Path('/root/.local/share/fonts/msttcorefonts')
FALLBACK = Path('/usr/share/fonts/truetype/dejavu')
INK = '#262c30'
MUTED = '#687278'
TEAL = '#397f78'
BLUE = '#3e739a'
ORANGE = '#956b30'
PINK = '#965a81'


def font(size, bold=False, serif=True, italic=False):
    if serif:
        name = 'Timesbi.TTF' if bold and italic else 'Timesbd.TTF' if bold else 'Timesi.TTF' if italic else 'Times.TTF'
        path = TIMES / name
        if not path.is_file():
            path = FALLBACK / ('DejaVuSerif-Bold.ttf' if bold else 'DejaVuSerif.ttf')
    else:
        path = FONT / ('Lato-Heavy.ttf' if bold else 'Lato-Regular.ttf')
    return ImageFont.truetype(str(path), size)


def text(d, xy, value, size=30, fill=INK, bold=False, serif=True, italic=False, anchor=None):
    d.text(xy, value, font=font(size, bold, serif, italic), fill=fill, anchor=anchor)


def card(d, box, outline='#cbd4d5', fill='white', radius=12, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(d, start, end, color='#879ba7', width=3):
    d.line((*start, *end), fill=color, width=width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    p1 = (end[0]-12*math.cos(a-.48), end[1]-12*math.sin(a-.48))
    p2 = (end[0]-12*math.cos(a+.48), end[1]-12*math.sin(a+.48))
    d.polygon([end, p1, p2], fill=color)


def dashed(d, start, end, color='#b49a68', width=2, dash=9):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = math.hypot(dx, dy)
    for a in range(0, int(length), dash*2):
        b = min(a+dash, length)
        d.line((start[0]+dx*a/length, start[1]+dy*a/length,
                start[0]+dx*b/length, start[1]+dy*b/length), fill=color, width=width)


def latent(im, center, label, active=True, size=80):
    d = ImageDraw.Draw(im)
    x, y = center
    r = size/2
    card(d, (x-r, y-r, x+r, y+r), '#6ba9cb' if active else '#cbd4d5',
         '#a5d2ee' if active else '#fbfcfc', radius=9)
    text(d, (x, y), label, 36, '#274f69' if active else '#a2aaad', italic=True, anchor='mm')


def hero():
    im = Image.new('RGB', (1600, 650), 'white')
    d = ImageDraw.Draw(im)
    text(d, (800, 66), 'Supervise each latent reasoning step.', 53, bold=True, anchor='mm')
    text(d, (800, 128), 'Learn distinct steps during training. Keep inference compact.', 31,
         MUTED, serif=False, anchor='mm')
    d.line((64, 175, 1536, 175), fill='#d4e3e0', width=2)
    original = Image.open(OUT.parent / 'simcot_arch.png').convert('RGB')
    crop = original.crop((0, 1560, 6263, 3165))
    crop = crop.resize((1480, round(1480 * crop.height / crop.width)), Image.Resampling.LANCZOS)
    im.paste(crop, (60, 200))
    text(d, (800, 613), 'Auxiliary decoder during training · Removed for ordinary inference', 29,
         TEAL, serif=False, anchor='mm')
    im.save(OUT / 'hero.png', optimize=True)


def walkthrough_frame(phase):
    im = Image.new('RGB', (1440, 760), 'white')
    d = ImageDraw.Draw(im)
    text(d, (48, 37), 'From question to interpretable reasoning steps', 41, bold=True)
    text(d, (48, 91), 'Explanatory animation · Figure 4 and Appendix I.1', 24, MUTED, serif=False)
    card(d, (45, 153, 435, 617), '#a3ccc5', '#eaf6f3', 16)
    text(d, (75, 180), 'Question', 32, TEAL, True)
    lines = ['120 watermelons.', '30% harvested first.', 'Two weeks later,', '3/4 of the remaining', 'melons are harvested.', '', 'How many are left?']
    for i, line in enumerate(lines):
        text(d, (75, 231+i*45), line, 31, INK, i in (0, 6))
    stage = '1. Encode the question' if phase == 0 else '2. Reason in latent space' if phase <= 5 else '3. Inspect the latent steps'
    text(d, (492, 155), stage, 31, INK, True)
    text(d, (492, 202), 'The trained LLM reasons through latent states.', 27, MUTED)
    arrow(d, (435, 327), (516, 327))
    card(d, (523, 265, 679, 389), '#87b8b0', '#dcefeb', 12)
    text(d, (601, 314), 'LLM', 35, TEAL, True, anchor='mm')
    text(d, (601, 355), 'prompt', 24, MUTED, anchor='mm')
    centers = [(766, 327), (906, 327), (1046, 327), (1186, 327)]
    arrow(d, (679, 327), (719, 327))
    for i, p in enumerate(centers):
        if i < 3: arrow(d, (p[0]+47, 327), (centers[i+1][0]-47, 327))
    arrow(d, (1230, 327), (1247, 327))
    for i, p in enumerate(centers):
        latent(im, p, f'z{i+1}', phase >= i+1)
    d = ImageDraw.Draw(im)
    card(d, (1250, 266, 1390, 389), '#d49abb' if phase >= 5 else '#cbd4d5',
         '#f5dce9' if phase >= 5 else '#fbfcfc', 12)
    text(d, (1320, 292), 'Answer', 24, PINK if phase >= 5 else MUTED, anchor='mm')
    text(d, (1320, 343), '21' if phase >= 5 else '…', 49, INK if phase >= 5 else '#a2aaad', True, anchor='mm')
    if phase <= 5:
        card(d, (491, 456, 1390, 616), '#d6dddd', '#f7f9f8', 14)
        text(d, (522, 483), 'Ordinary inference', 29, TEAL, True)
        text(d, (522, 528), 'Question → latent reasoning → answer', 31, INK)
        text(d, (522, 574), 'The auxiliary decoder is removed.', 28, MUTED)
    else:
        card(d, (491, 434, 1390, 619), '#d9bd88', '#fff4d6', 14)
        text(d, (520, 452), 'Optional decoder: published reasoning steps', 30, ORANGE, True)
        coords = [(520, 528, '0.3 × 120 = 36'), (960, 528, '120 − 36 = 84'),
                  (520, 575, '3/4 × 84 = 63'), (960, 575, '84 − 63 = 21')]
        for i, (x, y, value) in enumerate(coords):
            if phase >= 6+i:
                text(d, (x, y), f'z{i+1}', 29, BLUE, italic=True)
                text(d, (x+57, y-1), value, 32, INK)
        for i, (x, _) in enumerate(centers):
            if phase >= 6+i:
                dashed(d, (x, 370), (x, 432))
    steps = ['Encode', 'Reason in latents', 'Answer', 'Decode on demand']
    current = 0 if phase == 0 else 1 if phase <= 4 else 2 if phase == 5 else 3
    d.line((48, 650, 1392, 650), fill='#e0e6e4', width=2)
    for i, name in enumerate(steps):
        x = 52+i*345
        d.ellipse((x, 676, x+17, 693), fill='#6c9d98' if i <= current else '#d7dedd')
        text(d, (x+32, 663), name, 28, INK if i <= current else MUTED, i == current)
    text(d, (48, 720), 'Steps are decoded for inspection; ordinary inference emits the answer after the latent phase.', 22,
         MUTED, serif=False)
    return im


def walkthrough():
    frames = [walkthrough_frame(i) for i in range(10)]
    palette = frames[-1].quantize(colors=128).getpalette()
    palette_im = Image.new('P', (1, 1)); palette_im.putpalette(palette)
    quantized = [im.quantize(palette=palette_im, dither=Image.Dither.NONE) for im in frames]
    quantized[0].save(OUT / 'latent-walkthrough.gif', save_all=True,
                      append_images=quantized[1:], duration=[1600, 900, 900, 900, 900, 1800, 1100, 1100, 1100, 4000],
                      loop=0, optimize=False, disposal=2)
    frames[-1].save(OUT / 'latent-walkthrough.png', optimize=True)


if __name__ == '__main__':
    hero()
    walkthrough()
    results()
    for name in ['hero.png', 'latent-walkthrough.gif', 'latent-walkthrough.png', 'results-overview.png']:
        print(name, (OUT / name).stat().st_size)
