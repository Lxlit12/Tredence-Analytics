# Self-Pruning Neural Network — Report

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

Each weight `w_ij` in the network is multiplied by a soft gate derived from a
learnable score:

```
gate_ij     = sigmoid(gate_score_ij)    ∈ (0, 1)
pruned_w_ij = weight_ij * gate_ij
output      = pruned_w @ x.T + bias
```

The total loss adds an L1 penalty on the mean gate value across all layers:

```
Total Loss = CrossEntropy(logits, labels)  +  λ × mean( sigmoid(gate_scores) )
```

### Why L1 drives gates to exactly zero

The gradient of the sparsity term w.r.t. each gate score is:

```
∂(mean gate) / ∂gate_score_ij  =  sigmoid(s) × (1 − sigmoid(s)) × (1/N)
```

This **always pushes scores in the negative direction**. Unlike an L2 penalty
whose gradient shrinks proportionally as values approach zero, the L1 gradient
remains substantial in the linear region of the sigmoid (near score = 0),
giving the optimizer a consistent signal to keep pushing scores negative.

Once `gate_score_ij` is sufficiently negative, `sigmoid(gate_score_ij) ≈ 0`
and the corresponding weight is effectively removed from the network.

Weights that are genuinely important for classification resist this pressure
via the cross-entropy gradient — creating a **competition** that results in a
bimodal gate distribution: a large spike near 0 (pruned weights) and a
surviving cluster near 1 (important weights).

### The role of λ

| λ | Effect |
|---|--------|
| Small (0.1) | Sparsity pressure negligible; almost no gates close |
| Medium (1.0) | Clear pruning; ~15% of weights removed with minimal accuracy cost |
| Large (5.0) | Aggressive pruning; ~67% of weights removed; small accuracy cost |

---

## 2. Results

**Setup:** 30 epochs, Adam optimizer (lr = 1e-3 for weights, lr = 1e-2 for
gate_scores), cosine LR annealing, batch size 128, CIFAR-10 dataset.
Sparsity threshold: gate value < 0.01 (i.e. gate_score < −4.6).

| Lambda | Test Accuracy | Sparsity Level (%) | Notes |
|:------:|:-------------:|:------------------:|-------|
|  0.1   |    60.57%     |       0.01%        | Near-baseline; gates barely move |
|  1.0   |    60.38%     |      15.39%        | 15% pruned, only −0.19% accuracy |
|  5.0   |    58.74%     |      67.40%        | 67% pruned, only −1.83% accuracy |

**Key observations:**

- λ = 0.1 establishes the accuracy ceiling (~60.6%). Almost no gates close,
  confirming that without sufficient sparsity pressure the network ignores the penalty.
- λ = 1.0 is the sweet spot: 15% of weights pruned at essentially zero accuracy
  cost. The network has identified and discarded genuinely redundant connections.
- λ = 5.0 is the headline result: **two-thirds of all weights removed** while
  retaining 97% of baseline accuracy (58.74% vs 60.57%). This demonstrates
  that the majority of weights in a naive MLP are redundant for this task.

---

## 3. Gate Distribution Plot

See `gate_distribution.png`. Each row corresponds to one λ value.
Left panel: full soft-gate distribution over [0, 1].
Right panel: zoom on [0, 0.1] to show the near-zero spike clearly.

**Reading the λ = 5.0 plot:**

- A massive spike at gate ≈ 0 containing ~67% of all weights — fully suppressed.
- A secondary cluster near gate ≈ 0.9–1.0 — the surviving important connections.
- A clear **gap** between the two modes confirms near-binary gate states: scores
  are driven either strongly negative (pruned) or strongly positive (kept).

This bimodal shape is the hallmark of successful learned sparsity — the L1
penalty has effectively binarised the gates without any explicit binarisation.

---

## 4. Implementation Notes

### PrunableLinear

`gate_scores` is registered as an `nn.Parameter` with the same shape as
`weight`. The forward pass is a single element-wise multiply:

```python
gates          = torch.sigmoid(self.gate_scores)   # (out, in) ∈ (0,1)
pruned_weights = self.weight * gates               # element-wise
output         = F.linear(x, pruned_weights, self.bias)
```

PyTorch autograd differentiates through the product automatically — gradients
flow to both `weight` and `gate_scores` without any custom backward pass.

### Optimizer setup

`gate_scores` use a **separate param group with 10× the learning rate** of
weights. This is essential: Adam's per-parameter adaptive LR scaling otherwise
suppresses the sparsity gradient relative to the noisier weight gradients,
causing gates to stagnate near their initial value.

### Sparsity metric

A gate is counted as pruned when `sigmoid(gate_score) < 0.01`, i.e.
`gate_score < −4.6`. This strict threshold ensures only truly suppressed
weights are counted, not just slightly reduced ones.

### Possible extensions

- **Hard pruning pass:** After training, replace sub-threshold gates with
  exact zeros and fine-tune on classification loss alone — typically recovers
  1–2% accuracy at no sparsity cost.
- **Structured pruning:** One gate per neuron (not per weight) would enable
  real inference speed-ups via tensor shape reduction.
- **λ annealing:** Gradually increasing λ during training achieves higher
  sparsity with less accuracy loss than a fixed value.
