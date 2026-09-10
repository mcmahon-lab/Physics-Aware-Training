# PAT — questions and answers

The questions I actually asked while learning this, answered in a line or three.
The detail lives in `pat-diagram.pdf` (its last page is a glossary).

Notation: `i₁ → [layer 1] → o₁`, then `i₂ = o₁ → [layer 2] → o₂`. That equality is
the **joint**. In the PDF, colour marks provenance: orange = measured on the device,
blue = computed on the computer, green = the weights θ. B's training pages are all
blue, D's are all orange; PAT is orange values with blue slopes.

## The big picture

**Q. Is this a transformer with one special layer?**
No transformer — it's any layer stack where one layer is a physical object instead
of code. Everything else (loss, optimizer, autograd, the loop) is untouched.

**Q. Forward through the physics, then backprop the output?**
Forward: yes — run the device, measure. Backward: the gradient detours through a
simulator differentiated **at the layer's saved inputs**, then continues down.

**Q. What do 87.80% vs 79.25% mean?**
PAT vs simulation-trained, tested *on the device*. Tested on the simulator, the
losing method reports 87.55% — it looks *better* from inside. Silent failure.

## Loss and data

**Q. Is the loss comparing the physics output with the model's prediction?**
No. `cross_entropy(measured logits, dataset label)`. Prediction from the device,
truth from the dataset. The simulator's output is never compared with anything.

**Q. Is the training set the device's output?**
Only for the side-training that *fits the simulator* (device outputs = targets).
The main training uses an ordinary labeled dataset (device outputs = predictions).

## The weights

**Q. What is θ, and where does it live?**
All the trainable numbers — floats on your computer, meaning dial settings. Written
into the device's knobs every batch; changed only by `optimizer.step()`. The device
stores nothing. "Copy θ over once at the end" is option B, the one that fails.

## The simulator

**Q. What is it? Is it just a neural net?**
Differentiable code that predicts the device — physics equations, or literally a
small MLP fitted beforehand to measured `(x, θ) → y` pairs. Built once, frozen, not
part of the network; output discarded, only slopes used.

**Q. "Simulator" sounds like option B — why the name in PAT?**
The name is its qualification (it can predict the plate — that's what makes its
slopes trustworthy), not its PAT job. `pat.py` calls it `f_backward`.

**Q. Why built-once / frozen / output-discarded / slopes-only / at-saved-inputs?**
Each clause blocks a relapse into B: outputs reaching the loss, a moving reference,
guesses replacing measurements, value errors compounding, slopes taken at drifted
points. Values compound; slopes are spent by one update and reset.

## The gradient

**Q. We have the device's output — why an approximate gradient?**
A measurement is a value; training needs a slope per knob, and a slope can't be
read off one point. Normal nets get slopes cheap because autograd sees their
internals; the device shows nothing.

**Q. So measure the slopes (option D)?**
Exact — nudge, rerun, compare — but one hardware run per knob per step. Millions.
That's why PAT borrows the simulator's slopes instead.

**Q. Why does vjp return the model's output in the backward pass?**
You can't differentiate a function without running it. `y[0]` is the by-product
prediction (discarded); `y[1]` is the gradient (kept).

**Q. "…on to Layer 1 as if nothing happened"?**
The slope call yields the knob gradient *and* an input gradient, which is handed
down. Layers below receive an ordinary-looking gradient — the interface carries no
provenance, so the detour is invisible. That's why PAT is ten lines.

**Q. Why a gradient about o₁ at all — o₁ is the real value?**
True ≠ good: o₁ is a perfect *measurement* of behaviour that is still *bad at the
task*. The wish g_o₁ criticises the behaviour, not the measurement — and it exists
only because θ₁ touches the loss solely through o₁: no wish about o₁, no grad θ₁.
Nothing is applied to o₁ itself; the wish converts to a knob update, and the next
batch's o₁ is the device's new, equally real reply.

## Layers, joints, depth

**Q. Why two physical layers in the diagrams?**
One shows the mechanism; two is the minimum with a **joint** — where layer 2's
slope is taken at layer 1's *measured* output (i₂ = o₁) instead of the simulator's guess. B and PAT use
the identical simulator; the only difference is what crosses each joint. Error
compounds only at joints, so PAT's rule: only measurements cross joints.

**Q. Why not wrap the whole device as one block? (case E)**
You can — same ten lines, `n = 1`, and it beats B because the forward pass and loss
stay real. But the block's simulator must chain through its *own* guessed
intermediates, so interior joints get no anchor and the gradient decays toward B's
as the block deepens. E is right when intermediates are physically unreachable,
wasteful when they aren't. Hence the rule: **wrap the physics at the finest
granularity you can measure** — a layer is a measurable unit.

**Q. In backward, is o₁ on the device or the computer?**
The computer. It was *born* on the device (measured), crossed once at the readout,
and during backward it is just a saved float — the device is idle. Same storage slot
B uses; only the birth certificate differs, and that is what buys 10.5% over 33%.

**Q. Doesn't layer 2's simulator have its own i₂?**
No — inputs are given to functions, not made by them. We fill that argument slot with
the saved measurement. What the simulator does generate is its own *output* guess,
and that is the thing discarded. A simulator filling its own input slot is exactly
what B and E do wrong.

**Q. Why does layer 2 need o₁ at all?**
Because a slope depends on where you stand, and every physical layer is bent by
construction ("don't be linear"). Same complaint from the loss gives a different —
sometimes opposite-signed — note depending on which value actually arrived. Ordinary
backprop stashes activations for exactly this reason; PAT just stashes measured ones.

**Q. Does θ₁ ignore layer 2 then?**
The reverse: everything above arrives compressed into one vector, the wish `g_o₁`,
which layer 2 wrote *while evaluating at the measured o₁*. θ₁ has no target of its
own — the dataset labels only the final answer — so the note is its only direction.
θ₁'s own slope call uses its own saved `(i₁, θ₁)`; o₁ reaches it only through the note.

**Q. What is an ODE?**
An equation giving rates of change, never outcomes ("acceleration, right now").
The layer's output = what the rule does when time passes; hardware integrates it
for free, the simulator in small code steps. Ex2's "clean ODE" is the un-spoiled
copy of the device's equation, playing the simulator.

**Q. "Any nonlinearity works"?**
Stacked linear layers collapse into one matrix, so a bend is mandatory — but *any*
bend: `sin` breaks every folklore rule and hits 98%. That licenses building layers
from any physics that has a bend.

## The experiments

**Q. How does Ex1 prove anything with no gap?**
It's the control: it clears "the architecture is too weak" as a suspect, so later
failures can only be blamed on the gap.

**Q. Is Ex3 option B? Why race B and C?**
Ex3 races B against C — the only two gradient *estimators* (A computes none, D is
the answer key, affordable only on a toy). Result: 33% vs 10.5% = `1.005²⁰ − 1`.

**Q. In Ex1, is the gap a function or a net?**
Neither — the gap is the *disagreement* between device and simulator. Zero in Ex1,
0.5% in Ex3, deliberate spoilage in Ex2, unknowable in a real lab.

## One-liners

- The device says **where** to differentiate; the simulator says **how steep**.
- Error compounds at joints; PAT stations a measurement at every joint.
- Ship the procedure, not the weights — trained weights belong to one machine.
- An approximate gradient applied to a real measurement beats an exact gradient
  applied to an imaginary one.
