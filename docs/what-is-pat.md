# What is Physics-Aware Training?

A standalone explainer. No code, no repository, no physics background assumed —
only that you know roughly what a neural network is and that training one involves
a loss and backpropagation.

---

## The short version

You can build a neural network layer out of a physical object instead of code — a
vibrating metal plate, light through a crystal, current through a circuit. It computes
fast and cheap. The problem is that you cannot backpropagate through a lump of metal.

Physics-Aware Training solves this by splitting the two directions of training:
**run the real object forwards, and use a simulation of it only to get the gradient.**
The gradient is approximate, because the simulation is imperfect. It works anyway,
because it is applied to what the object actually did rather than to what a model
guessed it would do.

That is the entire idea. The rest of this document is why each piece is necessary.

---

## 1. A layer can be a physical object

Strip a neural network layer down to its job. It takes numbers in, mixes them together
with adjustable strengths, bends the result through something non-straight, and hands
numbers out. In code that's `ReLU(Wx + b)`.

Nothing about that job requires a chip.

Take a row of pendulums, connected to each other by springs, and match the parts up:

| what a layer needs | what the pendulums do |
| --- | --- |
| numbers in | how far you pull each pendulum back before letting go |
| adjustable mixing | the stiffness of the spring between each pair — these are the weights |
| a bias | a small steady push on each pendulum |
| the computation | **let go and wait** — everything swings and tugs on everything else |
| numbers out | where each pendulum ends up |

That is a layer. Not an analogy for one — you can do arithmetic with it.

### The bend comes for free

A pendulum pulled back a little pulls back with a force proportional to the angle.
Pulled back a lot, it doesn't — the force flattens off and eventually reverses. That
bend is a fact about gravity, not a design decision.

And that bend is the activation function. You don't implement it. You don't choose it.

This is what makes the whole field possible. If a specific activation were required —
ReLU, say — you would have to find hardware that happens to implement ReLU, and almost
nothing does. But **any** bend works. The only real requirement is *don't be perfectly
straight*: a linear layer stacked with another linear layer collapses into a single
matrix, and the network can only draw straight lines.

Because any nonlinearity will do, the list of candidate substrates is enormous: sound
in a plate, light in a crystal, current in a transistor, spins in a magnet.

### Why bother

The plate does its layer in the time sound takes to cross it, using the energy you put
in to ring it, and it does every one of its degrees of freedom at once because that is
simply what happens when things are connected. No multiply-accumulate, no memory
traffic, no clock. That is the prize.

### The catch

You don't get to design the function. The device hands you whatever physics it has and
you only get to turn its knobs. Your inputs must be things you can physically *set*.
Your outputs must be things you can physically *measure*.

And the thing that breaks everything: **you cannot differentiate it.** A metal plate has
no source code and no computational graph. You can run it and see what comes out. That
is all.

---

## 2. Why the obvious fixes don't work

Training needs a gradient for every layer's parameters. The physical layer has knobs and
no derivative. There are only a few ways out.

### Don't train that layer

Freeze the knobs wherever they happen to be and train the digital parts around it.

This does something — a fixed random nonlinear mixing is genuinely useful, and it's the
idea behind reservoir computing. But you built a physical computer and then forbade it
from learning, so you need large digital layers to compensate, which is the expensive
thing you were trying to avoid.

### Train against a simulation, then transfer the weights

This is the answer almost everyone reaches for, because it is how the rest of machine
learning works: train on your machine, ship the weights.

Write a simulation of the device. Train against it, where derivatives are easy. Then set
the real device's knobs to the trained numbers.

**The transfer itself works perfectly.** Every number arrives exactly as intended. The
problem is that they were the right numbers for the wrong machine.

Your simulation is never exact. Call it 0.5% off — a generous figure; real hardware is
usually characterised much worse than that. Watch what depth does to it:

| after n layers | error |
| ---: | ---: |
| 1 | 0.5% |
| 2 | 1.1% |
| 5 | 3.1% |
| 10 | 8.3% |
| 20 | **33%** |

Errors compound, they don't add. Layer 2 doesn't just contribute its own 0.5% — it takes
layer 1's already-wrong answer and works on that. So depth, the thing that makes deep
learning work, is exactly what makes an imperfect simulation useless.

**And it fails silently.** This is what makes it a trap rather than a bug. The loss curve
is textbook-healthy. Accuracy climbs. Your simulator reports a good number and means it —
about a machine that isn't the one you have. In a published reproduction on Fashion-MNIST,
a simulation-trained network reported 87.55% to itself and delivered 79.25% on the actual
device. Nothing during training hinted at the gap.

"Then improve the simulation." You can. You cannot make it exact, and depth amplifies
whatever remains. Halving your modelling error moves the cliff a few layers further out.

### Measure the gradient on the hardware

You can get a gradient without any model at all. Measure the loss. Nudge one knob. Run the
device again. The difference is that knob's derivative — genuinely exact, no assumptions.

Then do the next knob. A modest physical network has millions of them, and you need this
**per gradient step**. It is honest and it is hopeless.

This last option is worth understanding, because it explains what is actually scarce.

---

## 3. A measurement is not a gradient

Here is the confusion worth clearing up, since it's the one that makes PAT look
unnecessary.

You ran the real device. You measured the output. Why isn't that the gradient?

Because **the device gives you a value, and training needs a slope.**

- A measurement says: *with these knobs and this input, this came out.* One point.
- A gradient asks: *if I nudged knob #4,192 slightly, how would the loss change?* A rate.

You cannot read a rate of change off a single point. You need at least two.

This isn't special to physics. Knowing your transformer's logits tells you nothing about
the gradient of the loss with respect to its weights, either. You still have to run the
backward pass. The difference is that in a normal network the backward pass is *possible*,
because the framework knows every operation that produced the output.

That is the real asset backpropagation has: **structure**. Knowing the chain of operations
lets it derive millions of slopes analytically, in a single pass. Measurement has no
structure — it sees a black box, so it has to discover each slope separately, one poke at
a time. That is why measurement costs one hardware run *per parameter*, and backprop costs
one pass *total*.

So:

| | truth | structure |
| --- | --- | --- |
| the device | ✓ real behaviour | ✗ opaque |
| a simulation | ✗ imperfect | ✓ fully differentiable |

The device has truth without structure. The simulation has structure without truth.

---

## 4. Physics-Aware Training

Take what each one actually has.

**Forward pass: run the real device.** Feed in the data, let the physics happen, measure
the output. Write those measured values down. Compute the loss from them — the loss is
therefore exactly right, because it came from reality.

**Backward pass: use the simulation.** The gradient walks back from the loss and reaches
the physical layer, where it has nothing to travel through. So hand it to the simulation
instead — and require the simulation to compute its derivative **at the values the device
actually produced**, not at its own prediction of what the device would have done.

That last clause is the whole method.

### Why it works

The simulation's derivative *formula* is still wrong — 0.5% off, at every layer. That error
doesn't go away.

But the *compounding* does. In pure simulation training, layer 2 works on layer 1's drifted
output, layer 3 on layer 2's, and the drift snowballs into 33%. Under PAT, the forward pass
re-anchors the state to reality at every single layer. The drift never gets started.

What remains is only the fixed per-layer formula error, applied once per layer — linear in
depth instead of exponential. In the same toy setup as the table above, 33% becomes about
10%.

### Why an approximate gradient is acceptable

Gradient descent does not need the exact gradient. It needs a direction that mostly points
downhill.

Think about what a scale error does. If your gradient is consistently 10% too large, you
take steps 10% too big. You arrive by a slightly different path, in a different number of
steps — but you are descending the same landscape, and **the bottom of the valley is in the
same place.** A gradient wrong by a scale factor has its zero exactly where the true one
does.

Simulation-only training doesn't have a scaled gradient. It has a gradient pointing down a
*different valley* — the simulator's. It finds that valley's bottom accurately. That bottom
just isn't where the device's is.

### The analogy that holds up

You're throwing darts. **Simulation-only training** is practising with your eyes closed,
imagining where each dart lands. You get very good at the imaginary board, and nothing warns
you. **PAT** is throwing real darts and actually looking. Your reasoning about how to adjust
your arm is still imperfect — that's the simulation, and it stays imperfect — but every
throw is corrected against a real outcome, so the flaws in your reasoning never accumulate.

Or, if you prefer: you're on a hillside in fog. The altimeter tells you exactly where you
are but not which way is down. The map tells you the shape of the terrain instantly but is
slightly wrong. PAT reads the map *at the position the altimeter reports*, and re-checks
every step.

---

## 5. What it costs, and what it changes

**The hardware is in the loop.** Every batch means a round trip: write the parameters and
input into the device, trigger it, measure, transfer back. On real instruments that round
trip — not the physics — is what sets training throughput.

**The gradient stays approximate.** Near the optimum, parameters tend to drift rather than
settle cleanly. The direction is right; the endgame is messier than real backpropagation.

**You still need a simulation.** PAT doesn't remove the modelling work. It removes the
requirement that the model be *good enough to train through*, which is a far weaker demand.

**The weights never live in the device.** They are ordinary floats on your computer,
written into the hardware as settings on every batch and read back out. The device stores
nothing, learns nothing, and is the same object after training as before. The optimizer
updates the weights in exactly the place it always did.

**You ship the procedure, not the weights.** This is the consequence people find most
surprising. A trained parameter set has absorbed *that particular device's* defects — it
performs better on the device it was trained against than a "correct" set would. So it
belongs to one physical unit. The next unit off the same production line has its own
quirks and needs its own training run. What you distribute is the training procedure. And
recalibrating a device that has drifted is the same operation as training one from scratch.

---

## 6. The one-sentence version

> An approximate gradient applied to a real measurement beats an exact gradient applied to
> an imaginary one.

---

## Source

Wright, Onodera, Stein, Wang, Schachter, Hu and McMahon, *Deep physical neural networks
trained with backpropagation*, Nature (2022).
[doi:10.1038/s41586-021-04223-6](https://doi.org/10.1038/s41586-021-04223-6) — the
supplementary material carries the derivation of the gradient estimator.
