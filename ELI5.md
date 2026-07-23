In plain English, the code is trying to answer:

> Can we change a small open model’s behavior by adding a small “behavior direction” inside it at inference time—and is that cheaper or better than fine-tuning it with QLoRA?

## What one experiment does

For a behavior—say “refuse harmful requests”—the pipeline:

1. Takes example prompts and splits them into:
   - **Steer set:** used to build a vector.
   - **Validation set:** used to choose settings fairly.
   - **Test set:** held back for the final result.

2. Builds a steering vector from paired examples:
   - a response showing the desired behavior;
   - a response showing the opposite behavior.

   For safety, “desired” means safe refusal.

3. Tries several ways of building that vector:
   - average difference / CAA;
   - PCA;
   - whitened difference;
   - optimized/BiPO-lite vector.

4. Injects the vector into different transformer layers with different strengths.

5. Checks whether outputs improve on validation prompts, then takes the best nonzero setting to the held-out test prompts.

So a steering result is not “this vector looked good once.” It should mean:

> “Using method X at layer Y with coefficient Z was selected without seeing the test set, then achieved this held-out behavior score.”

## What is being tested

There are four broad behavior families:

| Family | Desired result |
|---|---|
| Safety/refusal | Refuse genuinely harmful requests, without refusing benign ones |
| Classification | Return the correct allowed label |
| Abstention/hedging | Say “I don’t know” when appropriate, but answer answerable questions |
| Structured output | Produce valid JSON that satisfies a requested schema |

There is also a small **capability probe**: ordinary benign questions used to check whether steering made the model generally worse.

## Where QLoRA fits

The project then runs the same task with QLoRA:

* Steering gets a small steer set and creates a vector.
* QLoRA gets the same training examples and learns an adapter.
* Both are evaluated on the same validation/test prompts.

The comparison asks:

* Which gets better held-out task behavior?
* How many labeled examples did each need?
* How long did extraction/training take?
* How much GPU memory and artifact storage did each use?
* How much does each slow inference?
* Did either hurt unrelated capability?

That is the “money question.”

## How to interpret outcomes

### Best-case steering result

Steering is promising if it shows:

* held-out test improvement over coefficient `0`;
* improvement across multiple seeds/models, not just one run;
* low capability regression;
* lower setup cost or much fewer labeled examples than QLoRA;
* a broad good region across nearby layers/coefficient values.

Interpretation:

> “For this behavior and model size, inference-time steering is a viable cheap control mechanism.”

Not:

> “Steering replaces fine-tuning everywhere.”

### Best-case QLoRA result

QLoRA is preferable if it:

* has materially better held-out task quality;
* is more stable across prompts;
* has fewer weird fluency/capability failures;
* keeps winning once steering gets comparable optimization time.

Interpretation:

> “Steering is cheap for quick iteration, but QLoRA is the more reliable way to install this behavior.”

That is still a good research result.

### Negative steering result

If steering works on its steer/validation set but not test prompts, that is valuable:

> “The direction did not generalize; it likely captured template/style artifacts rather than the underlying behavior.”

That is exactly why the repository keeps validation and test separate.

### Safety-specific outcomes

For safety, you want both:

* **High refusal on harmful prompts**, and
* **Low refusal on benign XSTest-style prompts**.

A model that refuses everything is not “safer” in a useful sense—it is over-refusing.

The most interesting safety story would be:

> “A refusal direction exists in a model; additive steering improves refusal but can over-refuse; conditional steering or directional ablation may retain safety while reducing that collateral damage.”

## The important mental model

Think of the project as producing a table like this:

| Model | Behavior | Method | Best steering test score | QLoRA test score | Capability retained | Setup cost |
|---|---|---|---:|---:|---:|---:|
| Qwen 4B | Refusal | PCA, layer 14 | 0.72 | 0.81 | 0.94 | Steering cheaper |
| Gemma 4B | JSON | Mean diff, layer 18 | 0.68 | 0.66 | 0.98 | Steering wins |

You then tell the story from the pattern, not from one cherry-picked number.

Right now, the code is best understood as an **experiment factory**. It is not yet a polished claim-machine. Its job is to generate trustworthy raw comparisons so we can learn where steering works, where it breaks, and whether the cost tradeoff is actually worth it.