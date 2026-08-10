# Notes on physics-aware training

What I want to keep. Not a tutorial — `docs/introduction.md` defines PNNs and
PAT, and the paper's
[Supplementary Section 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04223-6/MediaObjects/41586_2021_4223_MOESM1_ESM.pdf#page=3)
has the formalism. These are the things that aren't written down anywhere else.

---

## The three notebooks are one argument

| | question | answer |
|---|---|---|
| Ex 1 | can physics be a neural network? | yes, 98% on 14x14 MNIST |
| Ex 3 | so can I simulate it and upload the weights? | no |
| Ex 2 | then how? | PAT |

Example 1 is the control. Its header says so: no simulation-reality gap, forward
and backward functions identical, PAT not used. It proves the architecture can
learn, and deliberately proves nothing about training real devices. Read alone
it looks unimpressive; read as step one of three it's doing exactly its job.

## The physics is a constraint list, not a component

Example 1 contains no physics. It's RK4 arithmetic on a CPU — nothing swings,
and `pat.py` is never imported. Formally it's a Neural ODE.

The physics enters as a set of things you are **not allowed to do**:

- the nonlinearity is `sin` because that's what a pendulum does — not a choice
- `J` must be symmetric because a spring pulls both ends equally
- inputs must be things you can physically *set* (initial angles)
- outputs must be things you can physically *measure* (final angles)
- the only compute primitive is **let time pass**

In an ordinary network I pick `Wx + b` and ReLU because they're convenient. In a
PNN the device hands me its transfer function and I only get to turn its knobs.

MNIST is a yardstick, not the content — it exists so a vibrating metal plate can
be compared against a conventional net on equal terms.

## `f` is the seam

```python
f = make_ode_map(ode, Nt, dt)   # simulation today; a lab instrument tomorrow
```

`f(x, C, e)` takes inputs and parameters, returns outputs. Point it at hardware
and nothing else in the notebook changes. That's the whole reason `pat.py` takes
`f_forward` as an argument. In the paper they did exactly this with a
speaker-driven metal plate, a nonlinear optical crystal, and an analog circuit.

## Nonlinearity is a generic resource

`sin` is a bad activation by every criterion ML uses: saturating, non-monotonic,
and periodic — two different states give identical output, so information is
destroyed. Nobody would design it. It falls out of gravity acting on a mass.

98% anyway. The only real requirement is:

> don't be linear

Replace `sin q` with `q` and every layer becomes affine, layers compose to a
single matrix, and the network collapses to logistic regression. Everything the
activation-function literature argues about — monotonicity, unboundedness,
gradient behaviour — is optimization convenience, not a requirement to compute.

This is what makes the field possible. If a specific activation were needed I'd
have to find hardware that happens to implement it, and almost nothing does.
Because any nonlinearity works, the substrate list is enormous: light in a
crystal, sound in a plate, current in a transistor, spins in a magnet.

## Differentiable splits into two things

- **mathematically differentiable** — a derivative exists
- **algorithmically differentiable** — I can compute it

Autograd needs the second: it records primitive ops as they execute and applies
the chain rule. A laser has a Jacobian and cannot tell me what it was.

> Differentiability is a property of the function.
> Backprop needs a property of the implementation.

So the two departures in `pat.py`: the forward pass is **never differentiated**
(it could block on an oscilloscope read), and `f_backward` is a **different
function** from `f_forward`. The second is the real break — ordinary autodiff's
correctness rests on the backward pass being the exact adjoint of the forward.
PAT discards that knowingly and argues the error is benign.

The alternatives exist and don't scale: finite differences cost O(n+p) hardware
calls per backward pass; evolutionary methods worse. PAT keeps backprop's cost
profile — one forward per layer.

## How the two methods converge

Both of them converge. That's the part worth holding onto: in-silico doesn't
diverge, oscillate, or fail loudly. It settles, smoothly, on the wrong answer.

They're minimising different things. In-silico's forward pass is the model, so
the loss it descends is the *model's* loss, and it finds that minimum accurately.
PAT's forward pass is the device, so the loss it descends is the real one; its
gradient is only approximate, which changes how fast it arrives, not where it
stops. A gradient that's consistently wrong by a scale factor still has its zero
in the same place. The endgame is messier — the supplementary reports parameters
drifting near the optimum rather than settling, and blames residual Jacobian
mismatch — but the direction is right.

Hence the silence. In-silico reports a good number and means it. The number is
about a machine that isn't the one I have.

Measured on Fashion MNIST. Both networks got the same initialisation, the same
data order and the same three epochs; the only difference is which function the
forward pass ran through.

```
                    on the device   on its own model
PAT                     87.80%           86.55%
in-silico               79.25%           87.55%
```

8.55 points lost at deployment. And in-silico's self-report, 87.55%, lands within
a quarter-point of PAT's *real* accuracy, 87.80% — looking only at the simulator,
the two methods are indistinguishable. There's no warning, just a number that
looks fine.

## What you ship is the procedure, not the weights

The committed PAT weights, unchanged, run through both maps on 2000 validation
images:

```
on the device (f_exp_*)    88.85%
on the model  (f_model)    86.60%
they agree on              93.35% of images
```

Identical parameters, identical inputs, one image in fifteen classified
differently. Note which way round it falls: the weights do *better* on the device
they were trained against than on the clean model. They absorbed that device's
particular defects, which makes them worth more on it than a "correct" set would
be.

So the trained parameters belong to one physical instance. Another unit off the
same line has its own defects and needs its own run. What gets distributed is the
training procedure, not a weight file — and recalibrating a drifted device is the
same operation as training one.

---

## Where the abstraction is thin

I abstract the maths rather than follow it. Everything above is my mental picture
of what the equations *do* — not a restatement of them, and I couldn't derive any
of it. That's a deliberate level to work at, and I think the picture is sound,
but it means these notes are only ever as good as the picture.

**The gradient equations.** I've read Supplementary (16)–(19) but not worked
through them. I can say *what* PAT does — physical forward pass, model Jacobians
evaluated at the true intermediate states `x[l]` rather than the model's `x̃[l]`
— but I couldn't derive the backward pass or re-obtain those equations myself.

**Why the correction is quantitatively that size.** Example 3's gradient gap at
n = 20. Its model's coefficient is off by 0.5% (`2.01` against `2.00`), so every
layer's Jacobian is wrong by a factor of 1.005. In-silico lands at 33% there,
PAT at `1.005^20 - 1 = 10.5%` — exactly that per-layer error compounding, with
nothing added. I've seen it reproduced, not shown it.

**The VJP machinery.** `torch.autograd.functional.vjp(f_backward, args,
v=grad_output)` is the whole backward pass in `pat.py`. I take on faith that it
computes `grad_outputᵀ · J_model` at the saved inputs; I haven't looked at how,
or why a vector-Jacobian product is the right primitive rather than the full
Jacobian.

**The solver.** RK4, `Nt = 5`, `t_end = 0.5`. The notebook asserts 5 steps
converges. I haven't checked what breaks at fewer, or why RK4 rather than
something cheaper.

**How the code actually computes things.** `C2Q` folds a second calculation into
the diagonal of the coupling matrix, and the 16 image patches all pass through
one shared oscillator network.

**What PAT costs in a real hardware loop.** Everything here is simulated, so a
"device" forward pass is a matmul. On real hardware it's a round trip — set the
parameters, trigger, measure, transfer back — once per layer per batch, and that
round trip, not the physics, would set training throughput. In simulation PAT
costs 1.7x in-silico per step (74 ms against 43 ms per batch) because it solves
the ODE twice. On a device the ratio would be set by instrument latency instead,
and I have no idea what that number is for any real system.

---

## Settled

**~~The experiment this repo never runs.~~** Done. It lives in Example 2 as
"Part 2: the in-silico counterfactual", precomputed, with
`run_counterfactual = True` to reproduce. Numbers are in the notebook. What it
leaves open:

**Does the gap survive 20 epochs?** The ablation ran 3 epochs a side. In-silico's
on-experiment accuracy had nearly flattened (78.90 → 79.25) while its on-model
accuracy was still climbing, so the gap reads as structural rather than PAT merely
starting faster — but 3 epochs is 3 epochs, and the committed checkpoint took 20.

The distinction matters: "PAT reaches somewhere in-silico can't" is the paper's
claim; "PAT gets there faster" would be a convergence-speed result and wouldn't
justify the method. Three epochs cannot tell those apart — early on, faster looks
identical to better.

**Not running this.** It's ~30 min of compute for a secondary question, and the
3-epoch result already says what I needed it to say. Left unanswered rather than
half-answered.

---

## Appendix: what I changed

Fork of `mcmahon-lab/Physics-Aware-Training`. Grouped by intent; `git log` has
them in order.

**Getting it to run.** The repo pins torch 1.7.1 and pytorch-lightning 0.9.0, both
from 2020, and neither imports on anything current.

- *Make .gitignore self-contained* — cover `ml_dataset/`, `.venv/` and
  `training_logs/new_results/`, none of which the original knew about.
- *Fix Example 3 crash on current torch/matplotlib* — `plt.annotate(s=)` became
  `text=` in matplotlib 3.3, and a tensor was kept in a list that only needed its
  value.
- *Run Example 1 / Example 2 without pytorch-lightning* — `PNN` becomes a plain
  `nn.Module`. The forward pass is untouched, so the committed checkpoints still
  load and reproduce.
- *Make requirements.txt installable again* — which none of the above had touched.
  It was also missing `torchvision` and `pandas`, both imported by Examples 1 and
  2, so the manifest was incomplete before I got here.
- *Add a lockfile* — the pins are what rotted, so `requirements.txt` now carries
  lower bounds: what the project *needs*, resolvable years from now. The exact set
  it was verified against lives in `requirements.lock` (136 packages, generated
  with `uv pip compile --universal`, so it isn't macOS-only). Install loosely and
  get a working repo, or `pip install -r requirements.lock` and get precisely my
  environment. Both paths verified to resolve in a clean Python 3.12.

  Worth being explicit that this was a judgement call rather than a fix. Exact
  pins are reproducible but perishable — `torch==1.7.1` was a good pin in 2020 and
  became unsatisfiable the moment Apple Silicon shipped. Bounds are durable but
  drift. The lockfile is how you get both, and it matters less here than usual
  anyway: the published numbers come from the committed checkpoints, not from the
  library versions.
- *Stop telling the reader to install pytorch-lightning* — both notebooks opened
  their architecture section by instructing the reader, in bold, to
  `pip install pytorch-lightning==0.9.0`. Nothing imports it and it doesn't
  install. Anyone following the instructions failed before reaching a single
  working cell.

**Fixing what was wrong or dead.**

- *Fix mislabelled comment for f_exp_large* — a comment calling it the digital
  model. It's the experiment. Copy-paste rot, and actively misleading in a repo
  whose entire point is the difference between those two.
- *Fail loudly when train_flag = True* — it had become a silent `NameError` once
  Lightning was gone. Now it fails with an explanation.
- *Replace Example 1's / Example 2's dead pytorch-lightning trainer block with a
  note* — recording the hyperparameters
  that produced the committed checkpoints. Every Lightning API those blocks called
  has since been deleted upstream.

**Adding the missing experiment.**

- *Add PAT vs in-silico comparison to Example 2* — Part 1 runs the committed
  weights through both worlds; Part 2 trains a second network in silico and
  deploys it on the experiment. Precomputed, with `run_counterfactual = True` to
  reproduce. A `world()` context manager swaps the globals `PNN.forward` reads, so
  no forward-pass logic is duplicated.
