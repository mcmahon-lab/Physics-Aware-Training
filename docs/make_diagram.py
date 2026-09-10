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
    cells = [(8, 14, "batch $i_1$", "#999999", GREY_F, 1.2, "-", 10),
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
    txt(ax, 10, y + h + 1.8, "forward  →", 9.5, MUTE, ha="left", st="italic")


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
    txt(ax, 67.5, 29.4, "as a layer:  the input is what you SET on it,  the weights are its knobs,  the output is what you MEASURE — its natural nonlinearity plays the role of ReLU",
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

    txt(ax, 67.5, 1.6, "That single broken arrow is the entire problem.  Four ways to attack it — A, B, C, D — then three experiments.",
        12, INK, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 2 · the answer ═══════════════════════
def p_optC(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 4, "The answer — PAT, drawn on the same picture",
         "forward runs the REAL plate. When the gradient reaches the plate on the way back, it detours through a simulator.")

    chain(ax, 66.5, 10.5, ("the real plate\nthe physics", PHYS, PHYS_F, 2.4, "-"))

    arrow(ax, 50, 66, 50, 56.8, c=PHYS, lw=2.0)
    txt(ax, 48, 62.6, "its input $i$, saved", 8.5, PHYS, ha="right", st="italic")
    txt(ax, 129, 78.8, "the dataset's label enters here ↓", 8.5, DIGI, ha="right", st="italic")

    ax.plot([122, 82], [60, 60], color="#555555", lw=1.6, ls=(0, (4, 2)), zorder=3)
    ax.plot([82, 82], [60, 52], color="#555555", lw=1.6, ls=(0, (4, 2)), zorder=3)
    arrow(ax, 82, 52, 75.5, 52, c=DIGI, lw=2.0)
    box(ax, 44, 47.5, 31, 9, DIGI, DIGI_F, lw=2.2)
    txt(ax, 59.5, 54.2, "a SIMULATOR", 11, DIGI, w="bold")
    txt(ax, 59.5, 50.6, "a digital stand-in that behaves like the plate — often\nliterally a small neural net, fitted beforehand (Step 3)", 8, INK)
    arrow(ax, 43.5, 52, 37, 52, c=DIGI, lw=2.0)
    ax.plot([37, 37], [52, 60], color="#555555", lw=1.6, ls=(0, (4, 2)), zorder=3)
    arrow(ax, 37, 60, 12, 60, c="#555555", lw=1.6, ls=(0, (4, 2)))
    txt(ax, 101, 62.4, "backward — normal, until the plate", 8.5, MUTE, st="italic")
    txt(ax, 24, 57.4, "…and on to Layer 1, which receives an ordinary-looking\nincoming gradient — it cannot tell there was a detour", 7.5, MUTE)

    txt(ax, 67.5, 45.6, "NOT part of the network: built once, frozen, output discarded — asked only for its slope, AT those saved inputs", 9.5, DIGI, w="bold")
    txt(ax, 67.5, 42.8, "“simulator” names how it is BUILT — able to predict the plate, which is what makes its slopes trustworthy. In PAT it never simulates; the code just calls it $f_{backward}$.",
        8.5, MUTE, st="italic")

    rule(ax, 41)

    txt(ax, 27, 37.2, "forward", 11, PHYS, ha="right", w="bold")
    txt(ax, 30, 37.2, "happens ON THE DEVICE — physically run, measured. Nothing is imagined.", 10.5, INK, ha="left")
    txt(ax, 27, 33.4, "backward", 11, DIGI, ha="right", w="bold")
    txt(ax, 30, 33.4, "happens ON YOUR COMPUTER — the simulator is code; the device sits idle", 10.5, INK, ha="left")
    txt(ax, 27, 29.6, "update", 11, PARAM, ha="right", w="bold")
    txt(ax, 30, 29.6, "$optimizer.step()$ on your computer, exactly as always — $\\theta$ never lives in the plate", 10.5, INK, ha="left")

    box(ax, 6, 18.5, 123, 8, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 24.2, "✓   the error cannot compound: the forward pass re-anchors every layer to reality.", 11.5, PARAM, w="bold")
    txt(ax, 67.5, 20.8, "all that survives is the simulator's small per-layer slope error — and that does not snowball with depth.", 10, INK)

    txt(ax, 67.5, 13.6, "next: the same thing layer by layer, showing every copy.  Then the last honest alternative, and three experiments.",
        11, MUTE, st="italic")
    txt(ax, 6, 9.8, "colour = where a number comes from:", 8.5, MUTE, ha="left", w="bold")
    txt(ax, 40, 9.8, "measured on the DEVICE", 8.5, PHYS, ha="left", w="bold")
    txt(ax, 66, 9.8, "computed on the COMPUTER", 8.5, DIGI, ha="left", w="bold")
    txt(ax, 94, 9.8, "the weights θ", 8.5, PARAM, ha="left", w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 3 · alternative A ═══════════════════════
def p_optA(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 2, "Alternative A — don't train the physical layer",
         "freeze the knobs and let the digital layers do all the learning. The simplest idea, and it half-works.")

    chain(ax, 66, 10.5, ("the plate\nFROZEN", "#777777", GREY_F, 2.2, "-"))
    arrow(ax, 122, 61, 70, 61, c="#555555", lw=1.6, ls=(0, (4, 2)))
    cross(ax, 64, 61)
    txt(ax, 57, 61, "the gradient still stops at the plate", 10.5, BAD, ha="right", w="bold")
    txt(ax, 67.5, 56.4, "so every layer BEFORE the plate is stranded too — no gradient can ever reach Layer 1", 10.5, INK)

    rule(ax, 52.5)

    box(ax, 6, 30, 59, 20, DIGI, "#ffffff", lw=1.5)
    txt(ax, 35.5, 46.6, "why it half-works", 12, DIGI, w="bold")
    txt(ax, 10, 42.2, "·   a fixed, random, nonlinear mixing of the data\n     is genuinely useful", 10, INK, ha="left")
    txt(ax, 10, 36, "·   put the physics FIRST and train only a digital\n     readout after it — this is reservoir computing", 10, INK, ha="left")

    box(ax, 70, 30, 59, 20, PHYS, "#fffaf6", lw=1.5)
    txt(ax, 99.5, 46.6, "why it is not enough", 12, PHYS, w="bold")
    txt(ax, 74, 43, "·   the physics never adapts to your task", 10, INK, ha="left")
    txt(ax, 74, 38.6, "·   all learning is pushed into digital layers —\n     the exact compute the device was built to avoid", 10, INK, ha="left")
    txt(ax, 74, 33.4, "·   millions of knobs sit at random settings, unused", 10, INK, ha="left")

    box(ax, 6, 17.5, 123, 7.5, BAD, "#fdecec", lw=1.8)
    txt(ax, 67.5, 21.2, "✗   VERDICT: usable, but it wastes the device — the physical layer is a random mixer, not a trained computer.", 11.5, BAD, w="bold")

    txt(ax, 67.5, 12.6, "the knobs themselves have to learn.  Next: replace the plate with a model of it.", 11, MUTE, st="italic")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 4 · alternative B ═══════════════════════
def p_optB(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 3, "Alternative B — train against the simulator",
         "then copy the finished weights onto the plate, once. The plate leaves the loop entirely — everything flows, and that is the trap.")

    chain(ax, 66.5, 10.5, ("the SIMULATOR\nin the plate's slot", DIGI, DIGI_F, 2.4, "-"))

    arrow(ax, 122, 61, 12, 61, c="#555555", lw=1.6, ls=(0, (4, 2)))
    txt(ax, 67.5, 63.6, "backward — flows perfectly: every box is differentiable. Nothing breaks, training converges…", 9.5, INK)
    txt(ax, 67.5, 57.8, "…because BOTH passes run the simulator. The real plate is not in the loop at all — when training ends, $\\theta$ is copied onto it, once.", 10, BAD, w="bold")

    txt(ax, 22, 55.9, "during B's training every number is COMPUTED —", 8.5, DIGI, ha="left", st="italic")
    txt(ax, 72, 55.9, "nothing is MEASURED until training is already over.", 8.5, PHYS, ha="left", st="italic")
    rule(ax, 54.5)

    txt(ax, 6, 51.2, "the simulator, properly:", 11, DIGI, ha="left", w="bold")
    txt(ax, 6, 48.4, r"$f_{model}(x,\ \theta)$  —  differentiable torch code that predicts the plate: the physics written out, or a small net FITTED to the device —", 9.5, INK, ha="left")
    txt(ax, 6, 45.8, r"trained once, beforehand, on thousands of measured $(x,\ \theta) \rightarrow y$ pairs. In that tiny side-training the device's own outputs are the TARGETS.", 9.5, INK, ha="left")
    txt(ax, 6, 43.2, "the paper's “differentiable digital model”. Either way it is slightly WRONG — constants off, effects unmodelled. Shrinkable, never closable.", 9, PHYS, ha="left", st="italic")

    rule(ax, 42)

    txt(ax, 6, 38.8, "and slightly wrong is fatal at depth — suppose it is off by just 0.5%:", 11, BAD, ha="left", w="bold")
    gaps = [(1, "0.5%", INK), (2, "1.1%", INK), (5, "3.1%", WARN), (10, "8.3%", WARN), (20, "33%", BAD)]
    for i, (n, g, c) in enumerate(gaps):
        x = 18 + i * 21
        box(ax, x, 28, 17, 8, c if c != INK else "#999999",
            "#fdecec" if c == BAD else ("#fdf6e3" if c == WARN else GREY_F), lw=1.6 if c == BAD else 1.2)
        txt(ax, x + 8.5, 33.8, f"n = {n}", 9.5, MUTE)
        txt(ax, x + 8.5, 30.4, g, 13 if c == BAD else 11.5, c, w="bold")
        if i < 4:
            arrow(ax, x + 17.6, 32, x + 20.4, 32, ms=11)
    txt(ax, 67.5, 24.6, "errors COMPOUND — every layer works on the previous layer's already-wrong answer. The weights land tuned for a machine that does not exist.", 9.5, INK)

    box(ax, 6, 14.5, 123, 7, BAD, "#fdecec", lw=1.8)
    txt(ax, 67.5, 18, "✗   VERDICT: fails SILENTLY — from inside the simulator everything looks excellent. The drop only appears on the hardware, after training is done.", 10.5, BAD, w="bold")

    txt(ax, 67.5, 9.8, "PAT — next page — keeps this exact simulator, demoted to one job:  B trusted its OUTPUTS, PAT uses only its SLOPES.", 11, PARAM, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 5 · alternative D ═══════════════════════
def p_optD(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 6, "Alternative D — measure the gradient on the device itself",
         "no simulator, no assumptions. Honest, exact — and seeing why it is unaffordable is what makes PAT click.")

    chain(ax, 66.5, 10.5, ("the real plate\nthe physics", PHYS, PHYS_F, 2.4, "-"))

    arrow(ax, 118, 61, 15, 61, c=BAD, lw=1.8, ls=(0, (4, 2)))
    txt(ax, 67.5, 63.6, "there is NO backward pass — instead: nudge ONE knob, run the whole forward chain AGAIN, compare the two losses", 9.5, BAD, w="bold")
    txt(ax, 67.5, 57.8, "the difference between the two losses is that knob's derivative — exact, measured, no model anywhere", 10, INK)

    rule(ax, 54.5)

    txt(ax, 6, 51.2, "why it takes two runs per knob:", 11, DIGI, ha="left", w="bold")
    txt(ax, 6, 47.8, "one run gives a VALUE — “with these knobs, this came out”.  Training needs a SLOPE, per knob — “how would the loss move if knob #4,192 moved?”", 9.5, INK, ha="left")
    txt(ax, 6, 44.8, "a slope cannot be read off a single point. Each one costs a fresh run of the entire chain:", 9.5, MUTE, ha="left")

    steps = [("run the device", "loss = 0.4131"), ("nudge knob #1 by $\\epsilon$", "every other knob untouched"),
             ("run it again", "loss = 0.4126"), ("that knob's derivative", "(0.4126 − 0.4131) / $\\epsilon$ — EXACT")]
    for k, (t, sub) in enumerate(steps):
        x = 8 + k * 31
        box(ax, x, 34.5, 27, 7.5, PHYS if k == 3 else "#999999", PHYS_F if k == 3 else GREY_F, lw=1.4)
        txt(ax, x + 13.5, 39.9, t, 9.5, INK, w="bold")
        txt(ax, x + 13.5, 36.6, sub, 8.5, MUTE)
        if k < 3:
            arrow(ax, x + 27.6, 38.2, x + 30.4, 38.2, ms=11)
    txt(ax, 67.5, 30.6, "…now repeat, from the top, for knob #2.  And #3.  And every one of the millions — for EVERY training step.", 10.5, BAD, w="bold")

    box(ax, 6, 21, 123, 7, BAD, "#fdecec", lw=1.8)
    txt(ax, 67.5, 25.7, "✗   VERDICT: perfectly honest, completely unaffordable — one hardware run per knob, per step.", 11, BAD, w="bold")
    txt(ax, 67.5, 22.8, "note the colour: in D even the SLOPE is orange — measured. That is its honesty, and its cost.", 8.5, PHYS, st="italic")

    box(ax, 6, 5, 123, 13.5, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 15, "A wasted the device.     B trusted a wrong simulator.     D cannot afford exactness.", 11, INK, w="bold")
    txt(ax, 67.5, 11.2, "PAT takes D's instinct — trust only real measurements — and B's simulator, demoted to slopes.", 11.5, PARAM, w="bold")
    txt(ax, 67.5, 7.6, "that is PAT — Step 4.  One refinement question remains (next page), then the experiments.", 10, PARAM)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 5 · layer by layer ═══════════════════════
def p_layers(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 5, "PAT, layer by layer — what lives where, what is copied",
         "two physical layers, one training step. Every crossing is plain numbers; everything else stays put.")

    ax.plot([84, 84], [27.5, 80], color="#999999", lw=1.8, ls=(0, (5, 3)), zorder=2)
    txt(ax, 42, 81.4, "YOUR COMPUTER", 11.5, DIGI, w="bold")
    txt(ax, 108, 81.4, "THE DEVICE", 11.5, PHYS, w="bold")

    box(ax, 6, 74.5, 72, 4.8, PARAM, PARAM_F, lw=1.8)
    txt(ax, 42, 76.9, r"$\theta_1,\ \theta_2$  —  ALL the weights live here, always", 10.5, PARAM, w="bold")

    def layer_row(y, n, xin, th, xout):
        box(ax, 6, y, 72, 10.5, DIGI, "#ffffff", lw=1.4)
        txt(ax, 10, y + 8.3, f"LAYER {n} — computer side", 10, DIGI, ha="left", w="bold")
        txt(ax, 10, y + 5.9, f"have  {xin}  and  {th}      →  send both across", 9, INK, ha="left")
        txt(ax, 10, y + 3.9, f"save  ({xin}, {th})  for the backward pass", 9, DIGI, ha="left")
        txt(ax, 10, y + 1.9, f"receive  {xout}  —  the measurement", 9, PHYS, ha="left", w="bold")
        box(ax, 88, y, 41, 10.5, PHYS, PHYS_F, lw=2.0)
        txt(ax, 108.5, y + 8.3, f"LAYER {n} — the physics", 10, PHYS, w="bold")
        txt(ax, 108.5, y + 5.7, f"knobs set from {th},  state set from {xin}\nthe physics runs;  {xout} is measured off", 8.5, PHYS)
        txt(ax, 108.5, y + 1.6, "stores nothing, learns nothing", 8, MUTE, st="italic")
        arrow(ax, 78.5, y + 7.6, 87.5, y + 7.6, c=PHYS, lw=2.2)
        txt(ax, 83, y + 9.2, "COPY →", 7.5, PHYS, w="bold")
        arrow(ax, 87.5, y + 2.6, 78.5, y + 2.6, c=PHYS, lw=2.2)
        txt(ax, 83, y + 4.2, "← COPY", 7.5, PHYS, w="bold")

    layer_row(62.5, 1, r"$i_1$", r"$\theta_1$", r"$o_1$")
    txt(ax, 42, 61.5, r"$i_2 = o_1$  —  layer 2's input IS layer 1's measured output: the JOINT", 8, PARAM, w="bold")
    layer_row(50, 2, r"$i_2$", r"$\theta_2$", r"$o_2$")

    ax.plot([3.6, 3.6], [50, 73], color=PHYS, lw=2.2, zorder=3)
    ax.text(1.6, 61.5, "FORWARD", size=10, color=PHYS, ha="center", va="center",
            weight="bold", rotation=90, zorder=4)
    ax.plot([3.6, 3.6], [28.5, 48.5], color=DIGI, lw=2.2, zorder=3)
    ax.text(1.6, 38.5, "BACKWARD", size=10, color=DIGI, ha="center", va="center",
            weight="bold", rotation=90, zorder=4)

    box(ax, 6, 44, 72, 4.5, DIGI, DIGI_F, lw=1.4)
    txt(ax, 42, 46.2, r"loss  =  $cross\_entropy(o_2,\ label)$   —   computer only, nothing crosses", 9.5, INK)
    txt(ax, 129, 75.8, "forward: the computer does bookkeeping — the DEVICE does the computing", 8.5, PHYS, ha="right", st="italic")

    box(ax, 6, 28.5, 72, 13.5, DIGI, DIGI_F, lw=1.8)
    txt(ax, 10, 39.6, "BACKWARD — entirely on the computer", 10, DIGI, ha="left", w="bold")
    txt(ax, 10, 36.8, r"simulator's slope of layer 2, at saved $(i_2, \theta_2)$   →   grad $\theta_2$, and the wish about $o_1$", 9, INK, ha="left")
    txt(ax, 10, 34.2, r"simulator's slope of layer 1, at saved $(i_1, \theta_1)$   →   grad $\theta_1$", 9, INK, ha="left")
    txt(ax, 10, 31.4, r"$optimizer.step()$   →   $\theta_1, \theta_2$ change — here, and only here", 9.5, PARAM, ha="left", w="bold")

    box(ax, 88, 28.5, 41, 13.5, "#999999", GREY_F, lw=1.2)
    txt(ax, 108.5, 36.4, "the device is IDLE", 10.5, MUTE, w="bold")
    txt(ax, 108.5, 33, "nothing is sent to it,\nnothing comes back", 9, MUTE)

    txt(ax, 6, 27.1, "colour = where a number comes from:", 8.5, MUTE, ha="left", w="bold")
    txt(ax, 40, 27.1, "measured on the DEVICE", 8.5, PHYS, ha="left", w="bold")
    txt(ax, 66, 27.1, "computed on the COMPUTER", 8.5, DIGI, ha="left", w="bold")
    txt(ax, 94, 27.1, "the weights θ", 8.5, PARAM, ha="left", w="bold")
    rule(ax, 26)

    txt(ax, 6, 23, "Copied over, per layer, per batch:", 11, INK, ha="left", w="bold")
    txt(ax, 10, 19.6, r"→   $\theta_l$, written into the dials     +     the layer's input state", 10.5, PHYS, ha="left")
    txt(ax, 10, 16.4, "←   one measured output vector", 10.5, PHYS, ha="left")
    txt(ax, 10, 13, "never copied:   the label   ·   the loss   ·   any gradient   ·   the simulator   ·   autograd", 10.5, BAD, ha="left", w="bold")
    txt(ax, 10, 9.8, r"next batch, the same $\theta$ is written to the dials AGAIN — the device kept nothing, so nothing needs copying back.", 9.5, MUTE, ha="left", st="italic")
    txt(ax, 10, 6.6, r"why TWO layers? PAT's advantage lives at the joint: layer 2's slope is taken at the MEASURED $o_1$, not the simulator's guess of it.",
        9, PARAM, ha="left", w="bold")
    txt(ax, 10, 4.0, "One layer has no joint — and B would do almost as well.", 9, PARAM, ha="left", w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 7 · case E ═══════════════════════
def p_optE(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 7, "Case E — PAT, but wrap the whole device as ONE block",
         "the natural refinement question: why per-layer at all? Run the device end to end, measure once, use one big simulator.")

    txt(ax, 6, 80, "FORWARD — real, end to end, measured ONCE at the exit", 10.5, PHYS, ha="left", w="bold")
    box(ax, 8, 68, 13, 7, "#999999", GREY_F, lw=1.2)
    txt(ax, 14.5, 71.5, "batch $i_1$", 9.5)
    arrow(ax, 21.6, 71.5, 23.4, 71.5, ms=11)
    box(ax, 24, 64.5, 54, 13, PHYS, PHYS_F, lw=2.4)
    txt(ax, 51, 74.8, "THE WHOLE DEVICE", 11.5, PHYS, w="bold")
    box(ax, 28, 66.5, 14, 5.5, PHYS, "#ffffff", lw=1.3)
    txt(ax, 35, 69.2, "layer 1", 9.5, INK)
    arrow(ax, 42.6, 69.2, 49.4, 69.2, ms=11)
    box(ax, 50, 66.5, 14, 5.5, PHYS, "#ffffff", lw=1.3)
    txt(ax, 57, 69.2, "layer 2", 9.5, INK)
    txt(ax, 51, 65.6, r"$o_1$ exists physically — but is never read", 8, PHYS, st="italic")
    arrow(ax, 78.6, 71.5, 81.4, 71.5, ms=11)
    box(ax, 82, 68, 20, 7, "#999999", GREY_F, lw=1.2)
    txt(ax, 92, 71.5, "$o_2$ — measured", 9.5, PHYS, w="bold")
    arrow(ax, 102.6, 71.5, 105.4, 71.5, ms=11)
    box(ax, 106, 68, 23, 7, DIGI, DIGI_F, lw=1.4)
    txt(ax, 117.5, 71.5, r"loss $(o_2,\ label)$", 9.5)

    ax.plot([117.5, 117.5], [67.5, 53.5], color="#555555", lw=1.5, ls=(0, (4, 2)), zorder=3)
    arrow(ax, 117.5, 53.5, 90.8, 53.5, c=DIGI, lw=1.8)
    box(ax, 34, 49, 56, 9, DIGI, DIGI_F, lw=2.2)
    txt(ax, 62, 55, "ONE SIMULATOR OF THE WHOLE STACK", 10.5, DIGI, w="bold")
    txt(ax, 62, 51.4, r"slope taken at $(i_1,\ \theta_1,\ \theta_2)$ — the block's edge", 9, INK)
    arrow(ax, 33.4, 53.5, 24.6, 53.5, c=DIGI, lw=1.8)
    box(ax, 6, 50, 18, 7, "#999999", GREY_F, lw=1.2)
    txt(ax, 15, 53.5, r"grads for $\theta_1$, $\theta_2$", 8.5)

    txt(ax, 67.5, 45.4, r"but inside, its slope must chain ITSELF:   $J_2(\tilde{o}_1) \cdot J_1(i_1)$   —   $\tilde{o}_1$ is its OWN guess. Compounding returns, inside the block.",
        10, BAD, w="bold")

    rule(ax, 42)

    box(ax, 6, 22.5, 59, 17.5, PARAM, "#ffffff", lw=1.5)
    txt(ax, 35.5, 36.6, "PROS", 12, PARAM, w="bold")
    txt(ax, 10, 33.2, "·   the forward pass and the loss are still REAL —", 9.5, INK, ha="left")
    txt(ax, 10, 30.8, "     B's silent failure cannot happen", 9.5, INK, ha="left")
    txt(ax, 10, 28, "·   only the final output needs measuring — the ONLY option", 9.5, INK, ha="left")
    txt(ax, 10, 25.6, "     when intermediates are physically unreachable", 9.5, INK, ha="left")

    box(ax, 70, 22.5, 59, 17.5, BAD, "#ffffff", lw=1.5)
    txt(ax, 99.5, 36.6, "CONS", 12, BAD, w="bold")
    txt(ax, 74, 33.2, r"·   a measured $o_1$, if you can get one, goes UNUSED", 9.5, INK, ha="left")
    txt(ax, 74, 30.4, "·   gradient quality decays toward B's as the block deepens —", 9.5, INK, ha="left")
    txt(ax, 74, 28, "     the joints inside the block have no anchors", 9.5, INK, ha="left")
    txt(ax, 74, 25.2, "·   one big simulator is harder to build than several small ones", 9.5, INK, ha="left")

    box(ax, 6, 13.5, 123, 6.5, WARN, "#fdf6e3", lw=1.8)
    txt(ax, 67.5, 16.7, "△   VERDICT: legitimate — even necessary — when you cannot measure between layers. Wasteful when you can.", 11, WARN, w="bold")

    txt(ax, 67.5, 8.6, "the spectrum:    B (no real anchors)   →   E (anchored at the block's edges)   →   C (anchored at every joint)", 11, INK)
    txt(ax, 67.5, 4.6, "the rule underneath PAT:  wrap the physics at the finest granularity you can measure.", 12, PARAM, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 6 · experiment 1 ═══════════════════════
def p_ex1(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 8, "Experiment 1 — can physics learn?",
         "MNIST. The control: an oscillator network with NO reality gap — forward and backward are the same function. PAT is not used.", PART_TWO)

    chain(ax, 66.5, 10.5, ("oscillator network\n(no gap, by design)", PARAM, PARAM_F, 2.4, "-"))

    arrow(ax, 122, 61, 12, 61, c="#555555", lw=1.6, ls=(0, (4, 2)))
    txt(ax, 67.5, 63.6, "backward — flows: in this notebook the simulator IS the device. One function plays both roles.", 9.5, INK)
    txt(ax, 67.5, 58.8, "so the orange/blue distinction collapses here — measured and computed are the same number, by design", 8.5, MUTE, st="italic")

    rule(ax, 56.5)

    txt(ax, 6, 53.2, "the physical layer, mapped onto layer vocabulary:", 11, PHYS, ha="left", w="bold")
    rows = [("input", "pull each pendulum to a starting angle — the 14×14 image, loaded as angles"),
            ("weights", "the spring stiffness between every pair, plus a steady push per pendulum — the knobs"),
            ("compute", "let go and wait half a second — everything swings and tugs on everything else"),
            ("output", "where the 10 designated “class” pendulums ended up — the logits"),
            ("activation", "sin — nobody chose it, it falls out of gravity")]
    for k, (a, b) in enumerate(rows):
        y = 49.4 - k * 3.6
        txt(ax, 22, y, a, 10.5, PHYS, ha="right", w="bold")
        txt(ax, 25, y, b, 10.5, INK, ha="left")

    rule(ax, 30)

    box(ax, 6, 21, 123, 7.5, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 24.8, "✓   RESULT: 98% on MNIST — with sin, a “terrible” activation.  The only real requirement is: don't be linear.", 11.5, PARAM, w="bold")

    box(ax, 6, 6.5, 123, 12.5, "#999999", GREY_F, lw=1.4)
    txt(ax, 67.5, 16.2, "what it proves:  the architecture can learn — despite sin, symmetric weights, and “let time pass” as the only operation.", 10.5, INK, w="bold")
    txt(ax, 67.5, 12.8, "why the no-gap setup is the point:  it is the CONTROL. It clears the architecture as a suspect —", 10, INK)
    txt(ax, 67.5, 9.6, "so when training fails in Experiments 3 and 2, only the GAP is left to blame. One variable at a time.", 10, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 7 · experiment 3 ═══════════════════════
def p_ex3(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 9, "Experiment 3 — what a tiny gap does",
         "B and C, raced on a toy:  reality is  f(x) = 2.00·x¹·¹ ,  the simulator is  2.01·x¹·¹  —  0.5% off, applied 20 times like 20 layers.", PART_TWO)

    txt(ax, 6, 79.6, "①  the forward gap — reality and the simulator, drifting apart:", 11.5, INK, ha="left", w="bold")
    gaps = [(1, "0.5%", INK), (2, "1.1%", INK), (5, "3.1%", WARN), (10, "8.3%", WARN), (20, "33%", BAD)]
    for i, (n, g, c) in enumerate(gaps):
        x = 18 + i * 21
        box(ax, x, 68, 17, 8, c if c != INK else "#999999",
            "#fdecec" if c == BAD else ("#fdf6e3" if c == WARN else GREY_F), lw=1.6 if c == BAD else 1.2)
        txt(ax, x + 8.5, 73.8, f"n = {n}", 9.5, MUTE)
        txt(ax, x + 8.5, 70.4, g, 13 if c == BAD else 11.5, c, w="bold")
        if i < 4:
            arrow(ax, x + 17.6, 72, x + 20.4, 72, ms=11)
    txt(ax, 67.5, 64.6, "0.5% per layer compounds to 33% at n = 20 — every application works on the previous one's already-wrong answer", 9.5, INK)

    rule(ax, 61)

    txt(ax, 6, 57.6, "②  now the GRADIENTS, same setup — how far off is each method at n = 20?", 11.5, INK, ha="left", w="bold")
    bars = [(50, "true gradient", "#999999", 1.2, "0% — the reference. This is D's ideal, free here only because the “device” is a formula"),
            (43, "B — in-silico", BAD, 66, "33% off — exactly as wrong as its forward pass"),
            (36, "PAT", PARAM, 21, "10.5% off  =  1.005²⁰ − 1 — ONLY the per-layer slope error. Nothing compounded.")]
    for y, name, c, w, note in bars:
        txt(ax, 24, y + 1.7, name, 10.5, c if c != "#999999" else MUTE, ha="right", w="bold")
        box(ax, 26, y, w, 3.4, c, "#fdecec" if c == BAD else ("#e7f3ea" if c == PARAM else GREY_F), lw=1.4)
        txt(ax, 28 + w, y + 1.7, note, 9.5, INK, ha="left")

    txt(ax, 67.5, 30.4, "PAT does not FIX the gradient — it stops the error compounding:  exponential in depth  →  linear in depth.", 11, INK, w="bold")

    rule(ax, 27)

    box(ax, 6, 17.5, 123, 8, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 23.2, "✓   RESULT: with the same 0.5% modelling error, B's gradient is 33% wrong — PAT's stays at the irreducible 10.5% floor.", 11, PARAM, w="bold")
    txt(ax, 67.5, 19.6, "trained on a toy target, in-silico's loss then drifts away from reality while PAT tracks it — Experiment 2 repeats this at scale.", 9.5, INK)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 8 · experiment 2 ═══════════════════════
def p_ex2(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, 10, "Experiment 2 — PAT vs B",
         "Fashion-MNIST. Same seed, same data order, same budget. The only difference is which function the forward pass runs through.", PART_TWO)

    chain(ax, 68, 10, ("the spoiled ODE\nplays the device", PHYS, PHYS_F, 2.4, "-"))
    txt(ax, 129, 79.8, "spoiled = 10% nonlinearity error + stray couplings the simulator knows nothing about", 8.5, PHYS, ha="right", st="italic")

    arrow(ax, 50, 67.5, 50, 60.3, c=PHYS, lw=2.0)
    ax.plot([122, 82], [63.5, 63.5], color="#555555", lw=1.4, ls=(0, (4, 2)), zorder=3)
    ax.plot([82, 82], [63.5, 56.5], color="#555555", lw=1.4, ls=(0, (4, 2)), zorder=3)
    arrow(ax, 82, 56.5, 75.8, 56.5, c=DIGI, lw=1.8)
    box(ax, 44, 53, 31, 7, DIGI, DIGI_F, lw=2.0)
    txt(ax, 59.5, 56.5, "the clean ODE  =  the simulator", 9, DIGI, w="bold")
    arrow(ax, 43.5, 56.5, 37, 56.5, c=DIGI, lw=1.8)
    ax.plot([37, 37], [56.5, 63.5], color="#555555", lw=1.4, ls=(0, (4, 2)), zorder=3)
    arrow(ax, 37, 63.5, 12, 63.5, c="#555555", lw=1.4, ls=(0, (4, 2)))
    txt(ax, 104, 65.7, "PAT's backward detour, as in Steps 4–5", 8.5, MUTE, st="italic")
    txt(ax, 30, 60.9, "B instead runs the clean ODE in the plate's slot, both directions", 8.5, MUTE, st="italic")

    rule(ax, 49.5)

    txt(ax, 64, 45.6, "tested on the DEVICE", 10.5, PHYS, w="bold")
    txt(ax, 64, 43.2, "what you actually get", 8.5, MUTE)
    txt(ax, 101, 45.6, "tested on its own SIMULATOR", 10.5, DIGI, w="bold")
    txt(ax, 101, 43.2, "what you would report", 8.5, MUTE)
    txt(ax, 42, 38, "PAT", 12, PARAM, ha="right", w="bold")
    txt(ax, 64, 38, "87.80%", 16, PARAM, w="bold")
    txt(ax, 101, 38, "86.55%", 16, INK, w="bold")
    txt(ax, 42, 31.5, "B — simulate both ways", 11, BAD, ha="right", w="bold")
    txt(ax, 64, 31.5, "79.25%", 16, BAD, w="bold")
    txt(ax, 101, 31.5, "87.55%", 16, WARN, w="bold")
    txt(ax, 64, 27.2, "↑ reality: PAT wins by 8.55 points", 9.5, PARAM, w="bold")
    txt(ax, 101, 27.2, "↑ in here, B looks BETTER — the inversion is total", 9.5, BAD, w="bold")

    rule(ax, 24)

    box(ax, 6, 15, 123, 7.5, PARAM, PARAM_F, lw=1.8)
    txt(ax, 67.5, 18.8, "✓   RESULT: PAT's training-time numbers ARE its deployment numbers.  B never warns you — it hands you 87.55% and ships 79.25%.", 10.5, PARAM, w="bold")

    txt(ax, 67.5, 10.4, "bonus finding: PAT's weights score HIGHER on their own device (88.85%) than on the clean model (86.60%) — they absorbed that unit's defects.", 9.5, INK)
    txt(ax, 67.5, 7.4, "so trained weights belong to ONE physical machine: you ship the training procedure, not a weight file.", 9.5, MUTE)

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 9 · learnings ═══════════════════════
def p_learn(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, None, "Learnings", "the whole deck, one line per page.", ("SUMMARY", INK))

    rows = [
        ("A", BAD, "freeze the physical layer",
         "half-works (reservoir computing) — but the knobs never learn, so the device is a random mixer, not a computer."),
        ("B", BAD, "train against the simulator",
         "0.5% of modelling error compounds to 33% over 20 layers — and it fails SILENTLY: everything looks excellent until deployment."),
        ("C", PARAM, "PAT",
         "run REALITY forward; ask the simulator only for SLOPES, taken at reality's values. The error stops compounding."),
        ("D", BAD, "measure the gradient directly",
         "exact, but one hardware run per knob per step — unaffordable. This is why C borrows B's simulator for its slopes."),
        ("E", WARN, "wrap the whole device as one block",
         "works — real forward, one simulator — but measured intermediates go unused. Wrap at the finest granularity you can measure."),
        ("1", INK, "MNIST — the control",
         "physics CAN learn: 98% with sin as the activation. Any nonlinearity works — “don't be linear” is the whole requirement."),
        ("3", INK, "the toy function",
         "the argument in three numbers: forward gap 33%, B's gradient 33%, PAT's gradient 10.5% = 1.005²⁰ − 1."),
        ("2", INK, "Fashion-MNIST",
         "PAT 87.80% vs B 79.25% on the device — while B reports 87.55% to itself. Ship the procedure, not the weights."),
    ]
    for k, (n, c, title, line) in enumerate(rows):
        y = 76.5 - k * 7.7
        bullet(ax, 10, y, n, c, r=2.6, size=12)
        txt(ax, 15, y + 1.6, title, 12, c, ha="left", w="bold")
        txt(ax, 15, y - 2, line, 10.5, INK, ha="left")
        if k < 7:
            ax.plot([6, 129], [y - 4.2, y - 4.2], color="#eeeeee", lw=1.0, zorder=0)

    box(ax, 6, 3, 123, 8, PARAM, PARAM_F, lw=2.0)
    txt(ax, 67.5, 7, "an approximate gradient applied to a real measurement beats an exact gradient applied to an imaginary one.", 12.5, PARAM, w="bold")

    pdf.savefig(fig); plt.close(fig)


# ═══════════════════════ 11 · taxonomy ═══════════════════════
def p_tax(pdf):
    fig = plt.figure(figsize=(13.5, 9.5)); ax = canvas(fig)
    head(ax, None, "Taxonomy — the vocabulary, one entry each",
         "come back here whenever a word stops making sense.", ("REFERENCE", INK))

    def header(x, y, t):
        txt(ax, x, y, t, 10, MUTE, ha="left", w="bold")
        return y - 3.6

    def entry(x, y, term, c, lines):
        txt(ax, x, y, term, 9.5, c, ha="left", w="bold")
        for i, ln in enumerate(lines):
            txt(ax, x + 1.5, y - 2.4 - i * 2.3, ln, 8.5, INK, ha="left")
        return y - 2.4 - len(lines) * 2.3 - 1.3

    y = header(6, 80.5, "THE TRAINING LOOP")
    y = entry(6, y, "logits", DIGI, ["the network's raw output scores, one per class"])
    y = entry(6, y, "label", DIGI, ["the dataset's human-written right answer — used only in the loss"])
    y = entry(6, y, "loss  (cross_entropy)", DIGI, ["ONE number: how wrong the logits are against the label"])
    y = entry(6, y, "forward pass", DIGI, ["data flowing through the layers to produce the logits"])
    y = entry(6, y, "backward pass", DIGI, ["the gradient walking back from the loss, layer by layer"])
    y = entry(6, y, "gradient / slope", DIGI, ["per knob: which direction, and how strongly, turning it would",
                                              "reduce the loss. A rate of change — never a value"])
    y = entry(6, y, "autograd", DIGI, ["torch's bookkeeper — records every operation on the way forward,",
                                       "replays the recording backwards"])
    y = entry(6, y, "θ  (theta)", PARAM, ["all the trainable numbers, as floats on your computer.",
                                          "Physically: the device's dial settings"])
    y = entry(6, y, "optimizer.step()", PARAM, ["the ONLY place θ ever changes — each knob nudged along its gradient"])

    y = header(6, y - 1.5, "ALSO SEEN")
    y = entry(6, y, "in-silico", MUTE, ["“in software” — trained entirely against the simulator: option B"])
    y = entry(6, y, "i₁, o₁, i₂, o₂", PHYS, ["layer inputs and outputs.  i₂ = o₁ is the JOINT;",
                                            "every o is MEASURED on the device, and saved as the next i"])

    y = header(70, 80.5, "THE PLAYERS")
    y = entry(70, y, "the device / the plate  (f_exp)", PHYS, ["the physical object playing a layer — you can run it and",
                                                               "measure it; nothing else. In this repo: a spoiled ODE"])
    y = entry(70, y, "the simulator  (f_model / f_backward)", DIGI, ["code that predicts the device — built once, frozen, NOT part of",
                                                                    "the network; PAT uses only its slopes. The paper calls it the",
                                                                    "“differentiable digital model”"])
    y = entry(70, y, "the gap", BAD, ["the DISAGREEMENT between device and simulator. Not an object:",
                                      "zero in Ex1, a 0.5% coefficient in Ex3, deliberate spoilage in Ex2"])
    y = entry(70, y, "the joint", PARAM, ["where one layer's output becomes the next layer's input — the only",
                                          "place error compounds. PAT's rule: only measurements cross joints"])
    y = entry(70, y, "saved inputs", PHYS, ["the (x, θ) a physical layer received going forward, kept so the",
                                            "simulator's slope can be taken AT them"])
    y = entry(70, y, "vjp", DIGI, ["vector–Jacobian product: run a function once, get its slope back.",
                                   "Returns (prediction, gradient); PAT discards the prediction"])

    y = header(70, y - 1.5, "THE PHYSICS")
    y = entry(70, y, "ODE", PHYS, ["an equation giving rates of change, never outcomes.",
                                   "“Integrating” = stepping it forward in small time slices —",
                                   "hardware does that for free; the simulator does it in code"])
    y = entry(70, y, "nonlinearity / activation", PHYS, ["the bend between layers — without one, stacked layers collapse",
                                                         "into a single matrix. ANY bend works; sin is fine"])

    pdf.savefig(fig); plt.close(fig)


import sys
from pathlib import Path

OUT = Path(__file__).with_name("pat-diagram.pdf")
PAGES = [p_loop, p_optA, p_optB, p_optC, p_layers, p_optD, p_optE, p_ex1, p_ex3, p_ex2, p_learn, p_tax]
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
