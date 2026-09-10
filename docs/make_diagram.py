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


def arrow(ax, x1, y1, x2, y2, c=INK, lw=1.7, ls="-", rad=0.0, z=3, ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def rule(ax, y, x1=6, x2=129, c="#d8d8d8"):
    ax.plot([x1, x2], [y, y], color=c, lw=1.0, zorder=0)


def head(ax, step, title, sub):
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
    cross(ax, 62, 54)
    arrow(ax, 96, 54, 68, 54, c="#555555", lw=1.6, ls=(0, (4, 2)))
    box(ax, 97, 50, 28, 8, DIGI, DIGI_F, lw=1.5)
    txt(ax, 111, 54, "loss", 11.5, INK)
    txt(ax, 67.5, 46.4, "backprop needs the derivative of every layer. A lump of metal has no code, so it has no derivative.",
        10.5, INK)
    txt(ax, 67.5, 43.4, "and you cannot measure your way out: a measurement is a value, not a slope. One poke per knob = 2.6 million runs.",
        10.5, MUTE)

    rule(ax, 40.5)

    txt(ax, 6, 36.5, "THE FIX", 11, MUTE, ha="left", w="bold")
    box(ax, 24, 27, 45, 8, PHYS, PHYS_F, lw=2.2)
    txt(ax, 46.5, 31, "FORWARD  →  the real plate", 12, PHYS, w="bold")
    box(ax, 76, 27, 49, 8, DIGI, DIGI_F, lw=2.2)
    txt(ax, 100.5, 31, "BACKWARD  →  a simulation", 12, DIGI, w="bold")
    txt(ax, 46.5, 24.2, "run it, measure it", 10, INK)
    txt(ax, 100.5, 24.2, "differentiated at the plate's measured values", 10, INK)
    txt(ax, 67.5, 20.6, "the simulation is wrong, so the gradient is approximate — but reality re-anchors the state at every layer,",
        10.5, INK)
    txt(ax, 67.5, 17.6, "so the error never compounds. The weights stay on your computer throughout and change only in $optimizer.step()$.",
        10.5, INK)

    rule(ax, 14.5)

    txt(ax, 6, 11, "THE RESULT", 11, MUTE, ha="left", w="bold")
    txt(ax, 40, 7.2, "PAT", 12, PARAM, ha="right", w="bold")
    txt(ax, 56, 7.2, "87.80%", 18, PARAM, w="bold")
    txt(ax, 92, 7.2, "trained on the simulation only", 12, BAD, ha="right", w="bold")
    txt(ax, 108, 7.2, "79.25%", 18, BAD, w="bold")
    txt(ax, 67.5, 2.4, "both measured on the real device.  The simulation-trained one reports 87.55% to itself — it never tells you anything is wrong.",
        10.5, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 2 · the loop, and the break ═══════════════════════
def p_loop(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 1, "The loop you know, and the one box that breaks it",
         "generic picture: a transformer, or any network. This repo's own network is different — see Steps 6 and 7.")

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
    txt(ax, 10, 39.6, "then repeat for the next knob.  2,603,811 hardware runs, per gradient step.  Honest, and hopeless at any real scale.",
        11, BAD, ha="left", w="bold")

    txt(ax, 6, 34, "…so why is backprop cheap?  Because it knows the STRUCTURE.", 13, INK, ha="left", w="bold")
    txt(ax, 10, 29.9, "it knows every operation that produced the output, so it works out all 2.6 million slopes analytically, in one pass.",
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
        txt(ax, x, 79.5, name, 10, MUTE, w="bold")

    rows = [
        ("A", "freeze it", BAD, "no", "✗   no", "cheap", "wastes the device", MUTE,
         "leave the knobs where they are; train only the digital parts around the plate"),
        ("B", "simulate both ways", BAD, "no", "✓   for the wrong device", "cheap", "79.25%", BAD,
         "train entirely in software, then set the real dials to the finished numbers"),
        ("C", "PAT", PARAM, "yes, every batch", "✓   yes", "one round trip\nper layer", "87.80%", PARAM,
         "run the real plate; let the simulator differentiate AT the values it measured"),
        ("D", "measure it directly", BAD, "yes, millions of times", "✓   yes", "2.6M runs\nper step", "impossible", MUTE,
         "nudge a knob, re-run the plate, read off the difference — exact, and unaffordable"),
    ]
    for i, (letter, name, c, hw, learn, cost, res, rc, what) in enumerate(rows):
        y = 68 - i * 15
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

    rule(ax, 25)
    txt(ax, 6, 21.4, "Why B loses:  suppose the simulation is 0.5% off.  After n layers the gap is —",
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
    txt(ax, 67.5, 1.4, "and B never warns you: on its own simulator it reports 87.55%, a quarter-point above PAT's real score of 87.80%.",
        11.5, BAD, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 5 · what the simulator is ═══════════════════════
def p_sim(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 4, "What the simulator is",
         "a piece of code you write that predicts the device. Nothing more exotic than that.")

    box(ax, 16, 64, 103, 14, DIGI, DIGI_F, lw=1.8)
    txt(ax, 67.5, 74.2, r"$f_{model}(x,\ Q,\ e)$      →      a prediction of what the device will do", 13, INK)
    txt(ax, 67.5, 70, "same inputs as the real device: the data, the coupling settings, the drives", 10, MUTE)
    txt(ax, 67.5, 66.8, r"in this repo that is  $\ddot{q} = -\sin q + Q\sin q + e$ , solved with RK4, 5 steps  —  your physics homework, in code",
        10, INK)

    box(ax, 16, 53, 103, 8.4, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 58.8, "it exists only to be differentiated", 12.5, PARAM, w="bold")
    txt(ax, 67.5, 55.4, "$vjp$ does evaluate it — you cannot differentiate a function without running it — but its prediction is discarded.  Only the gradient leaves.",
        10, INK)

    rule(ax, 49)

    box(ax, 6, 22, 59, 24, DIGI, "#ffffff", lw=1.5)
    txt(ax, 35.5, 42.8, "where it comes from", 12.5, DIGI, w="bold")
    for i, t in enumerate(["write down the physics you believe governs\nthe device",
                           "measure its constants as best you can —\nmasses, stiffnesses, damping",
                           "code it in torch, so autograd can\ndifferentiate it",
                           "or: poke the device with many inputs, record\nthe outputs, fit a small net to that"]):
        txt(ax, 10.5, 37.6 - i * 5.0, "·   " + t, 9.5, INK, ha="left")

    box(ax, 70, 22, 59, 24, PHYS, "#fffaf6", lw=1.5)
    txt(ax, 99.5, 42.8, "why it is always wrong", 12.5, PHYS, w="bold")
    for i, t in enumerate(["your measured constants are a bit off",
                           "the device has effects you never modelled —\nstray couplings, drift, manufacturing quirks",
                           "your equations were an idealisation anyway"]):
        txt(ax, 74.5, 37.8 - i * 5.4, "·   " + t, 9.5, INK, ha="left")
    txt(ax, 99.5, 24.8, "you can shrink the gap. You cannot close it.", 10.5, PHYS, w="bold")

    rule(ax, 18.5)

    txt(ax, 6, 15.2, "One confusing thing about this repo: there is no hardware, so the device is faked too",
        12.5, INK, ha="left", w="bold")
    mono(ax, 12, 11.2, "f_model     = make_ode_map(ode_model,     Nt, dt)", 10.5, DIGI, "bold")
    txt(ax, 82, 11.2, "←  the simulator", 10.5, DIGI, ha="left", st="italic")
    mono(ax, 12, 7.8, "f_exp_small = make_ode_map(ode_exp_small, Nt, dt)", 10.5, PHYS, "bold")
    txt(ax, 82, 7.8, "←  code pretending to be the metal plate", 10.5, PHYS, ha="left", st="italic")
    txt(ax, 6, 3.6, r"$f_{exp}$ is the same equation, deliberately spoiled: a 10% error in the nonlinearity ($\eta$) plus stray couplings $Q_{noise}$ the simulator knows nothing about.",
        10.5, INK, ha="left")
    txt(ax, 6, 0.8, "In the paper it was a speaker-driven metal plate, a nonlinear optical crystal, an analog circuit — real objects on a bench.",
        10.5, MUTE, ha="left")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 6 · PAT ═══════════════════════
def p_pat(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 5, "PAT: the device forward, the simulator backward",
         "one substitution, thirty lines of code. Everything else in the training loop is untouched.")

    txt(ax, 8, 79.5, "FORWARD", 13, PHYS, ha="left", w="bold")
    box(ax, 30, 68, 22, 10, "#999999", GREY_F, lw=1.3)
    txt(ax, 41, 73, "input $x$", 11)
    arrow(ax, 53, 73, 61, 73, lw=2.0)
    box(ax, 62, 66, 37, 14, PHYS, PHYS_F, lw=2.4)
    txt(ax, 80.5, 75.4, "THE REAL PLATE", 12.5, PHYS, w="bold")
    txt(ax, 80.5, 70.6, "run it, measure it", 10, INK)
    arrow(ax, 100, 73, 108, 73, lw=2.0)
    box(ax, 109, 68, 20, 10, "#999999", GREY_F, lw=1.3)
    txt(ax, 119, 73, "output $\\hat{y}$", 11)
    txt(ax, 41, 65.4, "saved for the backward pass", 9.5, PHYS, st="italic")

    arrow(ax, 72, 65.6, 72, 61.4, c=PHYS, lw=2.2)
    txt(ax, 101, 63.5, "handed down: the values the plate actually saw", 10, PHYS, ha="left", st="italic")

    txt(ax, 8, 53.5, "BACKWARD", 13, DIGI, ha="left", w="bold")
    box(ax, 62, 47, 37, 14, DIGI, DIGI_F, lw=2.4)
    txt(ax, 80.5, 56.4, "THE SIMULATOR", 12.5, DIGI, w="bold")
    txt(ax, 80.5, 51.6, "differentiate it,\nat those saved values", 10, INK)
    arrow(ax, 61, 54, 53, 54, lw=2.0, c=DIGI)
    box(ax, 30, 49, 22, 10, "#999999", GREY_F, lw=1.3)
    txt(ax, 41, 54, "gradient", 11)
    arrow(ax, 108, 54, 100, 54, lw=2.0, c=DIGI)
    box(ax, 109, 49, 20, 10, "#999999", GREY_F, lw=1.3)
    txt(ax, 119, 54, "from the loss", 9.5)

    rule(ax, 43.5)

    txt(ax, 6, 40, "pat.py  —  the whole method", 12.5, INK, ha="left", w="bold")
    code = [("class func(torch.autograd.Function):", INK, ""),
            ("    def forward(ctx, *args):", INK, ""),
            ("        ctx.save_for_backward(*args)", PHYS, "save this layer's INPUTS — the real ones"),
            ("        return f_forward(*args)", PHYS, "the device runs here, with no graph recorded"),
            ("", INK, ""),
            ("    def backward(ctx, grad_output):", INK, ""),
            ("        args = ctx.saved_tensors", DIGI, "the values the device actually saw"),
            ("        torch.set_grad_enabled(True)", DIGI, "grad mode switches on only now"),
            ("        y = vjp(f_backward, args, v=grad_output)", DIGI, "y[0] = the model's guess — discarded"),
            ("        return y[1]", DIGI, "y[1] = the gradient — the only thing used")]
    for i, (line, c, note) in enumerate(code):
        yy = 36 - i * 2.7
        mono(ax, 8, yy, line, 10, c, "bold" if c != INK else "normal")
        if note:
            txt(ax, 68, yy, "←   " + note, 9.5, c, ha="left", st="italic")

    rule(ax, 9)
    txt(ax, 67.5, 5.8, "the gradient is approximate — it comes from a model that is wrong. It only has to point roughly downhill.",
        12, INK, w="bold")
    txt(ax, 67.5, 2.2, "Darts: you watch where each dart actually lands, while your sense of how to adjust your arm stays imperfect. Corrected against a real outcome every throw, the flaws never pile up.",
        10.5, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 7 · this repo ═══════════════════════
def p_repo(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 6, "This repo, specifically",
         "two physical layers — not a generic stack — and 2.6 million numbers that never leave your computer.")

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
    txt(ax, 6, 31.5, "$\\theta$  —  every trainable number in the model", 13, PARAM, ha="left", w="bold")

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
    head(ax, 7, "Exactly where the physics runs, and where the model runs",
         "both layers are physics. $f_{model}$ never appears in the forward pass — it is only ever called on the way back.")

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
PAGES = [p_summary, p_loop, p_gradient, p_options, p_sim, p_pat, p_repo, p_wiring]
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
