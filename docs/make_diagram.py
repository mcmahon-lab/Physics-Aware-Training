import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

DIGI, DIGI_F = "#1f4e79", "#e9f1f8"          # runs on your computer
PHYS, PHYS_F = "#a63603", "#fdeee2"          # runs on the device
PARAM, PARAM_F = "#14622f", "#e7f3ea"        # the weights
INK, MUTE, GREY_F = "#1a1a1a", "#6b6b6b", "#f5f5f5"
BAD, WARN = "#8f1d1d", "#8a6d00"

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def canvas(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 135); ax.set_ylim(0, 95); ax.axis("off")
    return ax


def box(ax, x, y, w, h, ec, fc, r=1.2, lw=1.4, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                ec=ec, fc=fc, lw=lw, linestyle=ls, zorder=z))


def txt(ax, x, y, s, size=10, c=INK, ha="center", va="center", w="normal", st="normal", z=4):
    ax.text(x, y, s, size=size, color=c, ha=ha, va=va, weight=w, style=st, zorder=z, linespacing=1.5)


def mono(ax, x, y, s, size=9.5, c=INK, w="normal", z=4):
    ax.text(x, y, s, size=size, color=c, ha="left", va="center", family="DejaVu Sans Mono",
            weight=w, zorder=z)


def mono_parts(ax, x, y, parts, size=11):
    """Lay out monospace runs left to right so argument positions can be coloured."""
    cw = 0.6018 * size / 7.2          # char width in data units (13.5in page, 135 units)
    for text, c, w in parts:
        ax.text(x, y, text, size=size, color=c, ha="left", va="center",
                family="DejaVu Sans Mono", weight=w, zorder=4)
        x += cw * len(text)
    return x


def mono_parts(ax, x, y, parts, size=11):
    """Lay out monospace runs left to right so argument positions can be coloured."""
    cw = 0.6018 * size / 7.2          # char width in data units (13.5in page, 135 units)
    for text, c, w in parts:
        ax.text(x, y, text, size=size, color=c, ha="left", va="center",
                family="DejaVu Sans Mono", weight=w, zorder=4)
        x += cw * len(text)
    return x


def arrow(ax, x1, y1, x2, y2, c=INK, lw=1.7, ls="-", rad=0.0, z=3, ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def rule(ax, y, x1=6, x2=129, c="#d8d8d8"):
    ax.plot([x1, x2], [y, y], color=c, lw=1.0, zorder=0)


PART_ONE = ("PART ONE   ·   WHAT PAT IS", INK)
PART_TWO = ("PART TWO   ·   WHAT THIS REPO DOES", PHYS)


def head(ax, step, title, sub, part=PART_ONE):
    label, c = part
    txt(ax, 129, 91.6, label, 9.5, c, ha="right", w="bold")
    ax.plot([129 - 0.34 * len(label), 129], [89.6, 89.6], color=c, lw=1.4, zorder=2)
    if step:
        txt(ax, 6, 91, f"Step {step}", 13, MUTE, ha="left", w="bold")
    txt(ax, 19 if step else 6, 91, title, 19, INK, ha="left", w="bold")
    txt(ax, 6, 87.2, sub, 12, MUTE, ha="left", st="italic")
    rule(ax, 84)


def bullet(ax, x, y, n, c, r=2.0, size=11):
    ax.add_patch(plt.Circle((x, y), r, color=c, zorder=4))
    txt(ax, x, y, str(n), size, "#ffffff", w="bold", z=5)


def cross(ax, x, y, s=2.6, lw=3.4):
    ax.plot([x - s, x + s], [y + s, y - s], color=BAD, lw=lw, zorder=5)
    ax.plot([x + s, x - s], [y + s, y - s], color=BAD, lw=lw, zorder=5)


def chain(ax, y, h, layer2):
    lab, ec, fc, lw, ls = layer2
    cells = [(8, 14, "batch $x$", "#999999", GREY_F, 1.2, "-", 10),
             (26, 18, "Layer 1", DIGI, DIGI_F, 1.4, "-", 11),
             (48, 18, lab, ec, fc, lw, ls, 11),
             (70, 18, "Layer N", DIGI, DIGI_F, 1.4, "-", 11),
             (92, 14, "logits", "#999999", GREY_F, 1.2, "-", 10),
             (110, 16, "loss\n$cross\\_entropy$", DIGI, DIGI_F, 1.4, "-", 9.5)]
    for x, w, label, e, f, l, st, s in cells:
        box(ax, x, y, w, h, e, f, lw=l, ls=st)
        txt(ax, x + w / 2, y + h / 2, label, s, INK)
    for x1, x2 in ((22, 26), (44, 48), (66, 70), (88, 92), (106, 110)):
        arrow(ax, x1, y + h / 2, x2, y + h / 2, ms=12)
    txt(ax, 68, y + h + 1.8, "⋯", 13, MUTE)      # the stack continues


# ═══════════════════════ 1 · the anchor ═══════════════════════
def p_summary(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, None, "Physics-Aware Training — the whole thing on one page",
         "come back here whenever a detail page stops making sense")

    txt(ax, 6, 78.5, "THE IDEA", 11, MUTE, ha="left", w="bold")
    box(ax, 24, 68, 42, 12, "#999999", GREY_F, lw=1.4)
    txt(ax, 45, 76.4, "an ordinary layer", 11, MUTE, w="bold")
    txt(ax, 45, 72, "$h = ReLU(Wx + b)$\narithmetic on a chip", 10.5, INK)
    txt(ax, 71, 74, "→", 20, INK)
    box(ax, 77, 68, 48, 12, PHYS, PHYS_F, lw=2.2)
    txt(ax, 101, 76.4, "a physical layer", 11, PHYS, w="bold")
    txt(ax, 101, 72, "pull the pendulums back, let go,\nmeasure where they end up", 10.5, INK)
    txt(ax, 67.5, 65.4, "same job — mix the inputs, bend them, produce outputs.  The weights are the spring stiffnesses.",
        10.5, MUTE)

    rule(ax, 62.5)

    txt(ax, 6, 58.5, "THE PROBLEM", 11, MUTE, ha="left", w="bold")
    box(ax, 24, 50, 30, 8, PHYS, PHYS_F, lw=1.8)
    txt(ax, 39, 54, "the plate", 11.5, PHYS, w="bold")
    arrow(ax, 55, 56.4, 96, 56.4, lw=1.6)
    txt(ax, 75.5, 58.8, "forward — fine", 8.5, MUTE, st="italic")
    cross(ax, 62, 51.4)
    arrow(ax, 96, 51.4, 68, 51.4, c="#555555", lw=1.6, ls=(0, (4, 2)))
    txt(ax, 82, 49, "backward — blocked", 8.5, BAD, st="italic")
    box(ax, 97, 50, 28, 8, DIGI, DIGI_F, lw=1.5)
    txt(ax, 111, 54, "loss", 11.5, INK)
    txt(ax, 67.5, 46.4, "backprop needs the derivative of every layer. A lump of metal has no code, so it has no derivative.",
        10.5, INK)
    txt(ax, 67.5, 43.4, "and you cannot measure your way out: a measurement is a value, not a slope. one poke per knob means millions of runs.",
        10.5, MUTE)

    rule(ax, 40.5)

    txt(ax, 6, 36.5, "THE FIX", 11, MUTE, ha="left", w="bold")
    box(ax, 24, 27, 45, 8, PHYS, PHYS_F, lw=2.2)
    txt(ax, 46.5, 31, "FORWARD  →  the real plate", 12, PHYS, w="bold")
    box(ax, 76, 27, 49, 8, DIGI, DIGI_F, lw=2.2)
    txt(ax, 100.5, 31, "BACKWARD  →  a simulation", 12, DIGI, w="bold")
    txt(ax, 46.5, 24.2, "run it, measure it", 10, INK)
    txt(ax, 100.5, 24.2, "differentiated at the plate's measured values", 10, INK)
    txt(ax, 67.5, 19.4, "the simulation is wrong, so the gradient is approximate — but reality re-anchors the state at every layer, so the error never compounds.",
        10.5, INK)

    rule(ax, 16.5)

    txt(ax, 6, 13.6, "THE RESULT", 11, MUTE, ha="left", w="bold")
    txt(ax, 6, 11, "Fashion-MNIST", 9, MUTE, ha="left", st="italic")
    txt(ax, 76, 13.8, "tested on the DEVICE", 10, PHYS, w="bold")
    txt(ax, 76, 11.4, "what you actually get", 8.5, MUTE)
    txt(ax, 108, 13.8, "tested on the SIMULATOR", 10, DIGI, w="bold")
    txt(ax, 108, 11.4, "what you would report", 8.5, MUTE)
    for y, lab, lc, a, ac, b, bc in ((7.4, "PAT", PARAM, "87.80%", PARAM, "86.55%", INK),
                                     (3.6, "trained on the simulation only", BAD, "79.25%", BAD, "87.55%", WARN)):
        txt(ax, 62, y, lab, 11, lc, ha="right", w="bold")
        txt(ax, 76, y, a, 16, ac, w="bold")
        txt(ax, 108, y, b, 16, bc, w="bold")
    txt(ax, 67.5, 0.8, "the left column is reality. The right column is what your simulator would tell you — and it says the WRONG method won.",
        11, BAD, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 2 · the cast ═══════════════════════
def p_cast(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, None, "The cast",
         "every name used in this deck, defined once — and the one word that causes all the trouble.")

    txt(ax, 6, 80.5, "TWO MACHINES", 11, MUTE, ha="left", w="bold")
    machines = [
        (6, PHYS, PHYS_F, "THE DEVICE", "$f_{exp}$", "hardware — a plate, a crystal, a circuit",
         ["you can RUN it", "you cannot look inside it", "you cannot differentiate it"]),
        (70, DIGI, DIGI_F, "THE SIMULATOR", "$f_{model}$", "code, on your computer",
         ["you can run it AND differentiate it", "it is wrong, by a few percent",
          "its answers are thrown away — only its slopes are kept"]),
    ]
    for x, ec, fc, name, sym, what, bullets in machines:
        box(ax, x, 59, 59, 19, ec, fc, lw=2.0)
        txt(ax, x + 4, 74.6, name, 12.5, ec, ha="left", w="bold")
        txt(ax, x + 26, 74.6, sym, 13, ec, ha="left")
        txt(ax, x + 4, 71.2, what, 9.5, MUTE, ha="left", st="italic")
        for k, b in enumerate(bullets):
            txt(ax, x + 4, 67.4 - k * 3.3, "·   " + b, 10, INK, ha="left")

    rule(ax, 56)

    txt(ax, 6, 52.5, "FIVE KINDS OF NUMBER", 11, MUTE, ha="left", w="bold")
    for x, name in ((20, "what it is"), (78, "how many numbers"), (103, "where it lives")):
        txt(ax, x, 48.5, name, 9.5, MUTE, ha="left", w="bold")
    rows = [("$x$", "the data going INTO a layer", "a vector", "computer  →  device", PHYS),
            (r"$\theta$", "the knob settings — these are the weights", "millions", "the computer, always", PARAM),
            (r"$\hat{y}$", "what the device MEASURED coming out", "a vector", "device  →  computer", PHYS),
            ("$L$", "the loss", "ONE number", "the computer", DIGI),
            ("$g$", "a gradient — how the loss moves per knob", "one per knob", "the computer", DIGI)]
    for k, (sym, what, size, lives, c) in enumerate(rows):
        y = 44 - k * 4.6
        if k % 2 == 0:
            box(ax, 8, y - 2.2, 121, 4.4, "#f7f7f7", "#f7f7f7", lw=0.5, z=0)
        txt(ax, 13, y, sym, 14, c, w="bold")
        txt(ax, 20, y, what, 10.5, INK, ha="left")
        txt(ax, 78, y, size, 10.5, MUTE, ha="left")
        txt(ax, 103, y, lives, 10.5, c, ha="left")

    rule(ax, 19)

    txt(ax, 6, 15.6, "The word that causes all the trouble", 12.5, BAD, ha="left", w="bold")
    box(ax, 6, 1, 123, 12, BAD, "#fdecec", lw=1.6)
    txt(ax, 67.5, 10.4, "In machine learning, \"the model\" means the network you are training.  The PAT paper calls $f_{model}$ \"the differentiable digital model\".",
        11, INK)
    txt(ax, 67.5, 7.4, "Those are two completely different things. This deck never says \"the model\" on its own:", 11, INK)
    txt(ax, 24, 3.6, "the network", 11, INK, ha="right", w="bold")
    txt(ax, 26, 3.6, "= what you are training, and it has physical layers", 10.5, MUTE, ha="left")
    txt(ax, 80, 3.6, "the simulator", 11, DIGI, ha="right", w="bold")
    txt(ax, 82, 3.6, "= $f_{model}$", 10.5, MUTE, ha="left")
    txt(ax, 106, 3.6, "the device", 11, PHYS, ha="right", w="bold")
    txt(ax, 108, 3.6, "= $f_{exp}$", 10.5, MUTE, ha="left")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 3 · the loop, and the break ═══════════════════════
def p_loop(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 1, "The loop you know, and the one box that breaks it",
         "a transformer, or any network — the boxes stand for a whole stack. Only one of them changes.")

    txt(ax, 6, 79.5, "AS YOU KNOW IT", 11, MUTE, ha="left", w="bold")
    chain(ax, 68, 9.5, ("Layer 2", PHYS, DIGI_F, 2.0, (0, (4, 2))))
    arrow(ax, 122, 64.5, 10, 64.5, c="#555555", lw=1.6, ls=(0, (4, 2)))
    txt(ax, 66, 61.4, "backward — the gradient walks back through every layer, then $optimizer.step()$ updates $\\theta$",
        11, INK)
    txt(ax, 66, 58.4, "$\\theta$ just means \"all the trainable numbers\" — floats in your computer's memory. Nothing else ever changes them.",
        10, MUTE)

    rule(ax, 55)

    txt(ax, 6, 51, "WITH A PHYSICAL LAYER", 11, PHYS, ha="left", w="bold")
    chain(ax, 39, 10.5, ("a metal plate\nthe physics", PHYS, PHYS_F, 2.4, "-"))
    arrow(ax, 122, 34.5, 70, 34.5, c="#555555", lw=1.6, ls=(0, (4, 2)))
    cross(ax, 64, 34.5)
    txt(ax, 57, 34.5, "the backward pass stops here", 11, BAD, ha="right", w="bold")
    txt(ax, 67.5, 29.4, "one layer = let 0.5 seconds pass.  On the plate that IS half a second of physics; simulating it costs 20 matrix multiplies.",
        10.5, INK)

    rule(ax, 26)

    box(ax, 6, 5, 59, 19, DIGI, "#ffffff", lw=1.5)
    txt(ax, 35.5, 21, "UNCHANGED", 12.5, DIGI, w="bold")
    for i, t in enumerate(["the data loading, the other layers, the logits",
                           "the loss  —  still $F.cross\\_entropy(\\hat{y},\\ label)$",
                           "the optimizer  —  still Adam",
                           "$\\theta$  —  still ordinary floats on your computer"]):
        txt(ax, 10, 16.6 - i * 3.4, "✓   " + t, 10, INK, ha="left")

    box(ax, 70, 5, 59, 19, PHYS, "#fffaf6", lw=1.5)
    txt(ax, 99.5, 21, "BROKEN", 12.5, PHYS, w="bold")
    for i, t in enumerate(["the plate has no source code",
                           "autograd recorded nothing on the way through",
                           "there is no derivative to walk back",
                           "so this layer cannot be trained"]):
        txt(ax, 74, 16.6 - i * 3.4, "✗   " + t, 10, INK, ha="left")

    txt(ax, 67.5, 1.6, "That single broken arrow is the entire problem. Everything that follows is about getting a gradient past it.",
        12, INK, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 3 · value vs slope ═══════════════════════
def p_gradient(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 2, "A measurement is not a gradient",
         "\"but we ran the real plate and measured the output — why isn't that enough?\"")

    box(ax, 6, 62, 59, 18, "#999999", GREY_F, lw=1.4)
    txt(ax, 35.5, 76.4, "what the device gives you", 12.5, MUTE, w="bold")
    txt(ax, 35.5, 71.4, "a VALUE", 15, INK, w="bold")
    txt(ax, 35.5, 66.4, "\"with these knobs and this input,\nthis came out\"   —   one point", 10.5, INK)

    box(ax, 70, 62, 59, 18, DIGI, DIGI_F, lw=1.8)
    txt(ax, 99.5, 76.4, "what training needs", 12.5, DIGI, w="bold")
    txt(ax, 99.5, 71.4, "a SLOPE", 15, INK, w="bold")
    txt(ax, 99.5, 66.4, "\"if I nudged knob #4,192 a little,\nhow would the loss change?\"", 10.5, INK)

    txt(ax, 67.5, 58.4, "you cannot read a slope off a single point — a rate of change needs at least two measurements",
        12, INK, w="bold")
    txt(ax, 67.5, 55.2, "the same is true of your transformer: knowing the logits tells you nothing about $\\partial loss/\\partial W$ until you run backward",
        10.5, MUTE)

    rule(ax, 51.5)

    txt(ax, 6, 47.5, "You could get the slopes by measurement alone…", 13, INK, ha="left", w="bold")
    txt(ax, 10, 43.4, "measure the loss   →   nudge one knob   →   run the plate again   →   the difference IS that knob's derivative, exactly",
        11, INK, ha="left")
    txt(ax, 10, 39.6, "then repeat for the next knob.  One hardware run per parameter, per gradient step — millions of runs. Honest, and hopeless at scale.",
        11, BAD, ha="left", w="bold")

    txt(ax, 6, 34, "…so why is backprop cheap?  Because it knows the STRUCTURE.", 13, INK, ha="left", w="bold")
    txt(ax, 10, 29.9, "it knows every operation that produced the output, so it works out every slope analytically, in a single pass.",
        11, INK, ha="left")
    txt(ax, 10, 26.1, "measurement knows no structure. It sees a black box, so it must discover each slope separately, one poke at a time.",
        11, INK, ha="left")

    rule(ax, 22)

    box(ax, 6, 10, 59, 10, PHYS, PHYS_F, lw=1.7)
    txt(ax, 35.5, 17, "the device", 12, PHYS, w="bold")
    txt(ax, 35.5, 12.8, "truth  ✓          structure  ✗", 12, INK)

    box(ax, 70, 10, 59, 10, DIGI, DIGI_F, lw=1.7)
    txt(ax, 99.5, 17, "the simulation", 12, DIGI, w="bold")
    txt(ax, 99.5, 12.8, "truth  ✗          structure  ✓", 12, INK)

    txt(ax, 67.5, 6.2, "PAT takes from each what it actually has: the device supplies the values, the simulation supplies the slopes.",
        13, PARAM, w="bold")
    txt(ax, 67.5, 2.4, "Fog on a hillside: the altimeter says exactly where you are, the map says which way is down. Read the map at the altimeter's position.",
        10.5, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 4 · the four ways out ═══════════════════════
def p_options(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 3, "The four ways out",
         "two dead ends, one impossible, and one that works.")

    for x, name in ((44, "touches hardware\nduring training"), (70, "can the physics\nlayer learn?"),
                    (95, "cost per\ntraining step"), (117, "result")):
        txt(ax, x, 81, name, 10, MUTE, w="bold")

    rows = [
        ("A", "freeze it", BAD, "no", "✗   no", "cheap", "wastes the device", MUTE,
         "leave the knobs where they are; train only the digital parts around the plate"),
        ("B", "simulate both ways", BAD, "no", "✓   for the wrong device", "cheap", "79.25%", BAD,
         "train entirely in software, then set the real dials to the finished numbers"),
        ("C", "PAT", PARAM, "yes, every batch", "✓   yes", "one round trip\nper layer", "87.80%", PARAM,
         "run the real plate; let the simulator differentiate AT the values it measured"),
        ("D", "measure it directly", BAD, "yes, millions of times", "✓   yes", "one run per\nparameter", "impossible", MUTE,
         "nudge a knob, re-run the plate, read off the difference — exact, and unaffordable"),
    ]
    for i, (letter, name, c, hw, learn, cost, res, rc, what) in enumerate(rows):
        y = 70 - i * 13
        box(ax, 6, y - 6, 123, 12.5, c if c == PARAM else "#dddddd",
            PARAM_F if c == PARAM else ("#ffffff" if i % 2 else "#f7f7f7"),
            lw=2.0 if c == PARAM else 1.0)
        bullet(ax, 11.5, y + 2, letter, c, r=2.8, size=13)
        txt(ax, 16.5, y + 2, name, 12, c, ha="left", w="bold")
        txt(ax, 16.5, y - 3, what, 9.5, MUTE, ha="left", st="italic")
        txt(ax, 44, y + 1, hw, 10, INK)
        txt(ax, 70, y + 1, learn, 10, INK)
        txt(ax, 95, y + 1, cost, 10, INK)
        txt(ax, 117, y + 1, res, 14 if "%" in res else 10.5, rc, w="bold")

    rule(ax, 23)
    txt(ax, 6, 19.4, "Why B loses:  suppose the simulation is 0.5% off.  After n layers the gap is —",
        12.5, INK, ha="left", w="bold")

    gaps = [(1, "0.5%", INK), (2, "1.1%", INK), (5, "3.1%", WARN), (10, "8.3%", WARN), (20, "33%", BAD)]
    for i, (n, g, c) in enumerate(gaps):
        x = 18 + i * 21
        box(ax, x, 8.5, 17, 8, c if c != INK else "#999999",
            "#fdecec" if c == BAD else ("#fdf6e3" if c == WARN else GREY_F), lw=1.6 if c == BAD else 1.2)
        txt(ax, x + 8.5, 14.4, f"n = {n}", 9.5, MUTE)
        txt(ax, x + 8.5, 11, g, 14 if c == BAD else 12, c, w="bold")
        if i < 4:
            arrow(ax, x + 17.6, 12.5, x + 20.4, 12.5, ms=11)

    txt(ax, 67.5, 4.6, "errors compound, they do not add — every layer works on the previous layer's already-wrong answer.",
        11.5, INK)
    txt(ax, 67.5, 1.4, "and B never warns you: on its own simulator it reports 87.55%, a quarter-point above PAT's real score of 87.80%.   (Fashion-MNIST)",
        11.5, BAD, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 5 · what the simulator is ═══════════════════════
def p_sim(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 4, "What the simulator is",
         "a differentiable stand-in for the device. There are two ways to build one, and neither is exotic.")

    box(ax, 12, 68, 111, 13, DIGI, DIGI_F, lw=1.8)
    txt(ax, 67.5, 77.4, r"$f_{model}(x,\ \theta)$      →      a prediction of what the device will do", 13, INK)
    txt(ax, 67.5, 73.4, r"same inputs as the real device: the data $x$ AND the knob settings $\theta$", 10.5, MUTE)
    txt(ax, 67.5, 70.2, r"it must take $\theta$ as an input — otherwise it cannot tell you how turning a knob changes the output",
        10, PHYS, st="italic")

    rule(ax, 65)

    routes = [
        (6, "1", "Write the physics",
         "you model the mechanism",
         ["write down the equations of motion you\nbelieve govern the device",
          "plug in constants you measured —\nmasses, stiffnesses, damping",
          "integrate them numerically, in torch"],
         r"e.g.   $\ddot{q} = -\sin q + Q\sin q + e$",
         "your measured constants are off, and the\nequations were an idealisation to begin with"),
        (70, "2", "Fit a network to measurements",
         "you imitate the behaviour — no physics written down",
         ["poke the real device thousands of times,\nsweeping the inputs AND the knob settings",
          r"record every  $(x,\ \theta) \rightarrow y$  pair it produces",
          "fit a small MLP to them, by ordinary backprop"],
         r"the simulator IS a neural network:   $g(x,\ \theta) \approx y$",
         "a fit is never exact, and it degrades away\nfrom the settings you happened to sample"),
    ]
    for x, n, title, doing, bullets, example, wrong in routes:
        box(ax, x, 29, 59, 33, DIGI, "#ffffff", lw=1.6)
        bullet(ax, x + 6, 58.6, n, DIGI, r=2.6, size=12)
        txt(ax, x + 11, 58.6, title, 12.5, DIGI, ha="left", w="bold")
        txt(ax, x + 29.5, 54.8, doing, 9.5, MUTE, st="italic")
        for k, b in enumerate(bullets):
            txt(ax, x + 4, 50.4 - k * 4.6, "·   " + b, 9.5, INK, ha="left")
        txt(ax, x + 29.5, 37.6, example, 10.5, INK)
        ax.plot([x + 4, x + 55], [35.2, 35.2], color="#dddddd", lw=1.0, zorder=0)
        txt(ax, x + 29.5, 32.4, wrong, 9.5, PHYS)

    txt(ax, 67.5, 26, "both are wrong, and you can shrink the gap but never close it", 11, PHYS, w="bold")

    rule(ax, 23)

    txt(ax, 6, 19.6, "What both routes have in common", 12.5, INK, ha="left", w="bold")
    for k, t in enumerate([
        "built ONCE, offline, before PAT training starts — not learned during it",
        "it IS executed — you cannot differentiate a function without running it — but its answer is discarded",
        "the device produces every number the network actually uses; the simulator produces only slopes",
        "it never has to be good enough to USE. Only good enough to DIFFERENTIATE."]):
        txt(ax, 10, 15.6 - k * 3.7, "·   " + t, 11, INK, ha="left")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 6 · PAT ═══════════════════════
def p_pat(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 5, "PAT: the device forward, the simulator backward",
         "one substitution. Everything else in the training loop is untouched.")

    # ── forward
    txt(ax, 8, 79.5, "FORWARD", 13, PHYS, ha="left", w="bold")
    box(ax, 30, 69, 22, 8, "#999999", GREY_F, lw=1.3)
    txt(ax, 41, 73, "input $x$", 11)
    arrow(ax, 53, 73, 61, 73, lw=2.0)
    box(ax, 62, 66, 37, 14, PHYS, PHYS_F, lw=2.4)
    txt(ax, 80.5, 75.4, "THE REAL PLATE", 12.5, PHYS, w="bold")
    txt(ax, 80.5, 70.6, "run it, measure it", 10, INK)
    arrow(ax, 100, 73, 108, 73, lw=2.0)
    box(ax, 109, 69, 20, 8, "#999999", GREY_F, lw=1.3)
    txt(ax, 119, 73, r"output $\hat{y}$", 11)
    txt(ax, 37, 66.8, "this layer's inputs are saved", 9.5, PHYS, ha="left", st="italic")

    # ── the loss closes the loop, down the right-hand side
    arrow(ax, 119, 68.6, 119, 64.4, lw=2.0)
    box(ax, 101, 56, 28, 8, DIGI, DIGI_F, lw=1.8)
    txt(ax, 115, 60, r"loss    $L(\hat{y},\ label)$", 11, INK)
    txt(ax, 99, 61.4, "the true label enters here,\nand only here", 9.5, DIGI, ha="right", st="italic")
    arrow(ax, 119, 55.6, 119, 51.4, lw=2.0, c=DIGI)

    # ── the layer's INPUTS (not anything from inside the plate) go to the backward pass
    ax.plot([34, 34], [68.6, 59], color=PHYS, lw=2.2, zorder=3)
    ax.plot([34, 72], [59, 59], color=PHYS, lw=2.2, zorder=3)
    arrow(ax, 72, 59, 72, 54.4, c=PHYS, lw=2.2)
    txt(ax, 43, 62.6, "the same inputs, handed to the backward pass", 9.5, PHYS, ha="left", st="italic")
    txt(ax, 43, 60.2, "(for a deeper layer, that is the previous layer's measured output)", 9, MUTE, ha="left", st="italic")

    # ── backward
    txt(ax, 8, 48, "BACKWARD", 13, DIGI, ha="left", w="bold")
    box(ax, 109, 43, 20, 8, "#999999", GREY_F, lw=1.3)
    txt(ax, 119, 47, r"$\partial L\ /\ \partial \hat{y}$", 11)
    arrow(ax, 108, 47, 100, 47, lw=2.0, c=DIGI)
    box(ax, 62, 40, 37, 14, DIGI, DIGI_F, lw=2.4)
    txt(ax, 80.5, 49.4, "THE SIMULATOR", 12.5, DIGI, w="bold")
    txt(ax, 80.5, 44.6, "differentiate it,\nat those saved values", 10, INK)
    arrow(ax, 61, 47, 53, 47, lw=2.0, c=DIGI)
    box(ax, 30, 43, 22, 8, "#999999", GREY_F, lw=1.3)
    txt(ax, 41, 47, "gradient", 11)
    txt(ax, 41, 41, "→ $optimizer.step()$", 9.5, PARAM, st="italic")

    rule(ax, 36)

    txt(ax, 6, 33, "The whole method, in about ten lines of PyTorch", 12.5, INK, ha="left", w="bold")
    code = [("class func(torch.autograd.Function):", INK, ""),
            ("    def forward(ctx, *args):", INK, ""),
            ("        ctx.save_for_backward(*args)", PHYS, "save this layer's INPUTS — the real ones"),
            ("        return f_forward(*args)", PHYS, "the device runs here, with no graph recorded"),
            ("", INK, ""),
            ("    def backward(ctx, grad_output):", INK, ""),
            ("        args = ctx.saved_tensors", DIGI, "the inputs saved on the way through"),
            ("        torch.set_grad_enabled(True)", DIGI, "grad mode switches on only now"),
            ("        y = vjp(f_backward, args, v=grad_output)", DIGI, "y[0] = the simulator's guess — discarded"),
            ("        return y[1]", DIGI, "y[1] = the gradient — the only thing used")]
    for k, (line, c, note) in enumerate(code):
        yy = 29.5 - k * 2.5
        mono(ax, 8, yy, line, 10, c, "bold" if c != INK else "normal")
        if note:
            txt(ax, 68, yy, "←   " + note, 9.5, c, ha="left", st="italic")

    rule(ax, 5.5)
    txt(ax, 67.5, 2.4, "the gradient is approximate — it comes from a model that is wrong. It only has to point roughly downhill.",
        12, INK, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 8 · one batch, with numbers ═══════════════════════
def p_trace(pdf):
    """Numbers below are a real torch trace; see the docstring snippet at the end of this file."""
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 7, "One batch, with real numbers",
         "two layers. The device is 5% off from the simulator — that is the only difference between them.")

    box(ax, 6, 74, 123, 8, INK, GREY_F, lw=1.4)
    txt(ax, 24, 79.4, "the device", 10.5, PHYS, ha="right", w="bold")
    txt(ax, 27, 79.4, r"$f_{exp}(x, W) = 1.05 \cdot \tanh(Wx)$", 11.5, INK, ha="left")
    txt(ax, 78, 79.4, "←  the 1.05 is the manufacturing error", 9.5, PHYS, ha="left", st="italic")
    txt(ax, 24, 76.2, "the simulator", 10.5, DIGI, ha="right", w="bold")
    txt(ax, 27, 76.2, r"$f_{model}(x, W) = \tanh(Wx)$", 11.5, INK, ha="left")
    txt(ax, 78, 76.2, "←  it knows nothing about the 1.05", 9.5, DIGI, ha="left", st="italic")

    # ── forward
    txt(ax, 6, 70, "FORWARD   ·   every number here was MEASURED on the device", 11.5, PHYS, ha="left", w="bold")
    cells = [(8, 30, "$x_0$", "[ 0.500, -0.200, 0.800 ]", "the input"),
             (52, 32, "$x_1$", "[ 0.417, -0.372, 0.325 ]", "measured"),
             (98, 28, "$x_2$", "[ 0.385, -0.235 ]", "measured — the logits")]
    for x, w, sym, vec, note in cells:
        box(ax, x, 59, w, 8, PHYS, PHYS_F, lw=1.6)
        txt(ax, x + w / 2, 64.6, sym, 12, PHYS, w="bold")
        txt(ax, x + w / 2, 61.4, vec, 10, INK)
        txt(ax, x + w / 2, 57, note, 9, MUTE, st="italic")
    for x1, x2, lab in ((39, 51, "save $(x_0, W_1)$"), (85, 97, "save $(x_1, W_2)$")):
        arrow(ax, x1, 63, x2, 63, lw=2.0)
        txt(ax, (x1 + x2) / 2, 65.6, "device", 9, PHYS, w="bold")
        txt(ax, (x1 + x2) / 2, 60.4, lab, 8.5, PARAM)

    rule(ax, 54)

    # ── loss
    txt(ax, 6, 50.5, "THE LOSS   ·   ordinary, on your computer", 11.5, DIGI, ha="left", w="bold")
    loss = [(10, 26, "$softmax(x_2)$", "[ 0.650, 0.350 ]"), (40, 18, "the label", "0"),
            (62, 22, "$L$", "0.4303"), (88, 41, r"$g_2 = \partial L / \partial x_2$", "[ -0.350, 0.350 ]")]
    for x, w, sym, vec in loss:
        box(ax, x, 40, w, 7.5, DIGI, DIGI_F, lw=1.4)
        txt(ax, x + w / 2, 45.4, sym, 10.5, DIGI, w="bold")
        txt(ax, x + w / 2, 42.2, vec, 10.5, INK)
    txt(ax, 67.5, 37.6, "one number out, then plain calculus to get $g_2$. No physics, nothing specific to PAT.",
        9.5, MUTE, st="italic")

    rule(ax, 35)

    # ── backward
    txt(ax, 6, 31.5, "BACKWARD   ·   differentiate the SIMULATOR, at the values the device saw", 11.5, DIGI, ha="left", w="bold")
    layers = [(19.5, "layer 2", r"differentiate $f_{model}$ at $(x_1, W_2)$, multiply by $g_2$",
               "[ 0.367, -0.224 ]", r"gradient for $W_2$   and   $g_1$ = [ -0.115, 0.163, -0.257 ]"),
              (7.5, "layer 1", r"differentiate $f_{model}$ at $(x_0, W_1)$, multiply by $g_1$",
               "[ 0.397, -0.354, 0.310 ]", r"gradient for $W_1$   →   $optimizer.step()$")]
    for y, name, does, discarded, out in layers:
        box(ax, 6, y, 123, 10, DIGI, DIGI_F, lw=1.5)
        txt(ax, 10, y + 6.4, name, 11, DIGI, ha="left", w="bold")
        txt(ax, 25, y + 6.4, does, 10.5, INK, ha="left")
        txt(ax, 25, y + 2.6, "the simulator also produced " + discarded, 9.5, BAD, ha="left")
        txt(ax, 74, y + 2.6, "←  DISCARDED", 9.5, BAD, ha="left", w="bold")
        txt(ax, 126, y + 4.5, out, 10.5, PARAM, ha="right", w="bold")

    box(ax, 6, 0.2, 123, 6.2, BAD, "#fdecec", lw=1.8)
    txt(ax, 67.5, 4.4, "the device measured [ 0.385, -0.235 ].      The simulator guessed [ 0.367, -0.224 ].", 11.5, INK, w="bold")
    txt(ax, 67.5, 1.8, "These two are NEVER subtracted, compared, or combined. The right-hand one is deleted.", 11.5, BAD, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ P2.1 · the three notebooks ═══════════════════════
def p_notebooks(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 1, "The three notebooks are one argument",
         "read in file order they look disconnected. Read as 1 → 3 → 2 they are a proof.", PART_TWO)

    for x, name in ((36, "asks"), (72, "answers"), (104, "which part of Part One")):
        txt(ax, x, 81, name, 10, MUTE, w="bold")

    rows = [
        ("Example 1", "MNIST", "#666666",
         "can physics be a\nneural network?", "yes — 98%",
         "none of it. This is the CONTROL:\nno gap, and PAT is not used",
         "forward and backward are the same function, so there is nothing for PAT to fix.\nIt proves the architecture can learn, and deliberately proves nothing about hardware."),
        ("Example 3", "a toy scalar function", BAD,
         "so can I train in simulation\nand upload the weights?", "no",
         "Step 4 — why option B loses",
         "reality is $2x^{1.1}$, the simulator is $2.01x^{1.1}$: 0.5% off. Apply both 20 times and the gap is 33%.\nThen the same experiment on gradients: 33% for option B, 10.5% for PAT."),
        ("Example 2", "Fashion-MNIST", PARAM,
         "then how?", "PAT",
         "Steps 5 – 7, on a real task",
         "the full method, plus the counterfactual: a second network trained in simulation only,\nsame seed and same budget, then deployed on the device. 87.80% against 79.25%."),
    ]
    for i, (name, task, c, asks, ans, maps, note) in enumerate(rows):
        y = 57 - i * 22
        box(ax, 6, y, 123, 21, c, "#ffffff", lw=1.8)
        txt(ax, 10, y + 16.2, name, 13.5, c, ha="left", w="bold")
        txt(ax, 10, y + 12.4, task, 9.5, MUTE, ha="left", st="italic")
        txt(ax, 36, y + 14.6, asks, 10.5, INK)
        txt(ax, 72, y + 14.6, ans, 13, c, w="bold")
        txt(ax, 104, y + 14.6, maps, 10, MUTE)
        ax.plot([10, 126], [y + 8.6, y + 8.6], color="#e0e0e0", lw=1.0, zorder=2)
        txt(ax, 67.5, y + 5, note, 9.5, INK)

    txt(ax, 67.5, 8.5, "Example 1 read alone looks unimpressive. Read as step one of three, it is doing exactly its job.",
        11.5, INK, w="bold")
    txt(ax, 67.5, 4.8, "Example 3 is the load-bearing one: one number of slack (2.01 against 2.00) and one knob (depth), so every effect is checkable by hand.",
        11, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 7 · this repo ═══════════════════════
def p_repo(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 2, "This repo's network",
         "two physical layers — not a generic stack — and 2.6 million numbers that never leave your computer.", PART_TWO)

    steps = [
        (8, 64, 36, 12, "#999999", GREY_F, "the input", "a 28 × 28 image"),
        (48, 64, 36, 12, "#999999", GREY_F, "cut into patches", "16 patches, 7 × 7 = 49 pixels each"),
        (88, 64, 41, 12, PHYS, PHYS_F, "LAYER 1   ·   physics",
         "a 100-oscillator network\nthe SAME one runs all 16 patches"),
        (8, 40, 41, 12, "#999999", GREY_F, "stack them back up",
         "16 × 100 = 1600, plus 10 \"class\noscillators\" starting at rest = 1610"),
        (53, 40, 36, 12, PHYS, PHYS_F, "LAYER 2   ·   physics", "one 1610-oscillator network"),
        (93, 40, 36, 12, DIGI, DIGI_F, "read out",
         "the final positions of those\n10 class oscillators = the logits"),
    ]
    for x, y, w, h, ec, fc, title, body in steps:
        box(ax, x, y, w, h, ec, fc, lw=2.0 if ec == PHYS else 1.3)
        txt(ax, x + w / 2, y + h - 3.2, title, 10.5, ec if ec != "#999999" else MUTE, w="bold")
        txt(ax, x + w / 2, y + h / 2 - 2.4, body, 9.5, INK)
    for x1, x2, y in ((44.5, 47.5, 70), (84.5, 87.5, 70), (49.5, 52.5, 46), (89.5, 92.5, 46)):
        arrow(ax, x1, y, x2, y, ms=12)
    ax.plot([108, 108], [63.5, 58.5], color=INK, lw=1.6, zorder=3)
    ax.plot([108, 28.5], [58.5, 58.5], color=INK, lw=1.6, zorder=3)
    arrow(ax, 28.5, 58.5, 28.5, 52.5, ms=13)
    txt(ax, 67.5, 55.4, "both layers are physics — there is no digital layer in between", 10, PHYS, st="italic")

    rule(ax, 35)
    txt(ax, 6, 31.5, "$\\theta$  —  every trainable number in the network", 13, PARAM, ha="left", w="bold")

    tbl = [("$fc\\_small.weight$", "100 × 100", "10,000", "spring stiffness between every pair, layer 1"),
           ("$fc\\_small.bias$", "100", "100", "the push applied to each oscillator, layer 1"),
           ("$fc\\_large.weight$", "1610 × 1610", "2,592,100", "spring stiffnesses, layer 2"),
           ("$fc\\_large.bias$", "1610", "1,610", "the pushes, layer 2"),
           ("$output\\_fac$", "one number", "1", "a single output scale")]
    for i, (n, shape, cnt, meaning) in enumerate(tbl):
        y = 26 - i * 4.0
        if i % 2 == 0:
            box(ax, 8, y - 1.9, 121, 3.8, "#f7f7f7", "#f7f7f7", lw=0.5, z=0)
        txt(ax, 11, y, n, 10.5, INK, ha="left")
        txt(ax, 45, y, shape, 10.5, MUTE)
        txt(ax, 68, y, cnt, 10.5, INK, ha="right")
        txt(ax, 74, y, meaning, 10.5, PHYS, ha="left")
    txt(ax, 45, 5.6, "TOTAL", 11.5, PARAM, w="bold")
    txt(ax, 68, 5.6, "2,603,811", 13, PARAM, ha="right", w="bold")
    txt(ax, 74, 5.6, "ordinary floats on your computer — never inside the plate", 10.5, PARAM, ha="left", w="bold")
    txt(ax, 67.5, 1.4, "They change in exactly one place, and it is the same place as in any torch model: $optimizer.step()$.",
        11.5, INK, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 7 · the wiring ═══════════════════════
def p_wiring(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 3, "Which function runs, in which direction",
         "both layers are physics. The simulator never appears in the forward pass — it is only ever called on the way back.", PART_TWO)

    txt(ax, 6, 80, "The wiring — two lines, and this is the whole mapping", 12.5, INK, ha="left", w="bold")
    box(ax, 10, 64, 119, 12, INK, GREY_F, lw=1.4)
    for i, (name, dev) in enumerate((("f_pat_small", "f_exp_small"), ("f_pat_large", "f_exp_large"))):
        mono_parts(ax, 14, 72.4 - i * 4.2,
                   [(f"{name} = make_pat_func(", INK, "normal"), (dev, PHYS, "bold"),
                    (", ", INK, "normal"), ("f_model", DIGI, "bold"), (")", INK, "normal")], 11)
    arrow(ax, 44.7, 61.4, 44.7, 65.6, c=PHYS, lw=1.6)
    txt(ax, 44.7, 59.6, "1st argument  =  the FORWARD pass  =  the device", 10.5, PHYS, w="bold")
    arrow(ax, 73, 60.8, 55.5, 65.4, c=DIGI, lw=1.6)
    txt(ax, 75, 59.8, "2nd argument  =  the BACKWARD pass  =  the simulator", 10.5, DIGI, ha="left", w="bold")

    rule(ax, 57)

    txt(ax, 6, 53.5, "Where those get called  —  $PNN.forward$", 12.5, INK, ha="left", w="bold")
    code = [("def forward(self, x):", INK, "normal", ""),
            ("    x = self.rearrange(x)", INK, "normal", "16 patches of 49 pixels"),
            ("    ...", MUTE, "normal", ""),
            ("    x = f_pat_small(x, *self.fc_small.parameters())", PHYS, "bold", "LAYER 1  ·  physics"),
            ("    ...", MUTE, "normal", ""),
            ("    x = f_pat_large(x, *self.fc_large.parameters())", PHYS, "bold", "LAYER 2  ·  physics"),
            ("    return self.output_fac * x[:, -10:, 0]", INK, "normal", "the 10 class oscillators = the logits")]
    for i, (line, c, w, note) in enumerate(code):
        yy = 49 - i * 2.9
        mono(ax, 10, yy, line, 10, c, w)
        if note:
            txt(ax, 74, yy, "←   " + note, 9.5, c if c != MUTE else MUTE, ha="left", st="italic")

    rule(ax, 27)

    txt(ax, 6, 23.5, "So, per batch:", 12.5, INK, ha="left", w="bold")
    txt(ax, 48, 19.6, "the FORWARD pass runs", 10.5, PHYS, w="bold")
    txt(ax, 97, 19.6, "the BACKWARD pass runs", 10.5, DIGI, w="bold")
    rows = [(r"layer 1   ·   100 oscillators", r"$f_{exp\_small}$   the device", r"$f_{model}$   at  $(x_0,\ C_1,\ e_1)$"),
            (r"layer 2   ·   1610 oscillators", r"$f_{exp\_large}$   the device", r"$f_{model}$   at  $(x_1,\ C_2,\ e_2)$"),
            ("the loss", r"$F.cross\_entropy$", "ordinary autograd — nothing special")]
    for i, (stage, fwd, bwd) in enumerate(rows):
        y = 17 - i * 4.4
        if i % 2 == 0:
            box(ax, 8, y - 2.1, 121, 4.2, "#f7f7f7", "#f7f7f7", lw=0.5, z=0)
        txt(ax, 11, y, stage, 10.5, INK, ha="left")
        txt(ax, 48, y, fwd, 10.5, PHYS)
        txt(ax, 97, y, bwd, 10.5, DIGI)

    txt(ax, 67.5, 4.0, "$f_{model}$ appears zero times in $forward()$.    $f_{exp}$ appears zero times in $backward()$.",
        12, INK, w="bold")
    txt(ax, 67.5, 1.0, "Neither is a layer of the network — they are two directions through the same one.", 11, MUTE)

    pdf.savefig(fig); plt.close(fig)


import sys
from pathlib import Path

OUT = Path(__file__).with_name("pat-diagram.pdf")
PAGES = [p_summary, p_cast, p_loop, p_gradient, p_options, p_sim, p_pat, p_trace,
         p_notebooks, p_repo, p_wiring]
if "--png" in sys.argv:
    class P:
        n = 0
        def savefig(self, fig):
            fig.savefig(f"/tmp/pg{P.n}.png", dpi=110); P.n += 1
    p = P()
    for fn in PAGES:
        fn(p)
else:
    with PdfPages(OUT) as pdf:
        for fn in PAGES:
            fn(pdf)
print(f"wrote {OUT}" if "--png" not in sys.argv else "wrote /tmp/pg*.png")
