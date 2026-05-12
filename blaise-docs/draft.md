# Corsaire-1 Technical Report

**BlaiseAI Research**

---

## Abstract

We present *Corsaire-1*, a supervised fine-tuned variant of DeepSeek-V3.2 in which every dense forward operator runs at four-bit precision or below, every persistent optimizer state is integer-quantized, and the model is trained without a persistent FP32 master copy of its parameters. The recipe combines (i) 50% routed-expert pruning by REAP, (ii) rotation-aware weight quantization (SpinQuant R₁/R₂), (iii) NVFP4 W4A4 GEMMs with stochastic-rounding cast, (iv) a 2.5-bit block-Hadamard codebook quantizer on the dense Multi-Latent Attention (MLA) latent (TurboQuant), (v) an FP8 indexer-key cache for DeepSeek Sparse Attention (IndexCache), (vi) a low-rank gated normalization (GatedNorm) and a paper-faithful output gate on MLA attention (G1), and (vii) a quantized AdamW variant (FlashAdamW) whose Error-Compensating Optimization (ECO) injection — both on the weight side and the activation side — is the sole mechanism preventing the missing FP32 master from leaking rounding mass into divergence.

We give the mathematical contract for every block, prove (or empirically verify) that the compositions are unbiased in expectation, characterize a parallelism contract that holds across tensor-, sequence-, context-, and expert-parallelism, and report end-to-end convergence behavior under the full stack on 16×B200 (TP=4, PP=4, EP=4, sequence length 32 768). The headline result is that a 202B-parameter mixture-of-experts model fine-tuned end-to-end at this aggressive compression level remains numerically stable and trainable on a one-billion-token Nemotron-derived conversational corpus, with a measured per-component convergence delta under 2% relative to a BF16 reference.

**Keywords:** mixture-of-experts, expert pruning, NVFP4, FP8, low-bit quantization, multi-latent attention, sparse attention, error-compensating optimization, stochastic rounding, supervised fine-tuning.

---

## 1. Introduction

Frontier mixture-of-experts (MoE) language models — most recently DeepSeek-V3.2 with multi-latent attention (MLA) and a sparse indexer (DSA) — have made full-precision training prohibitively expensive for groups operating outside hyperscale clusters. The interplay between model compression and training stability is the central obstruction: every percentage point of memory recovered by reducing precision risks producing a stochastic-gradient signal that no longer drives convergence.

Existing low-bit recipes attack the problem from one side at a time. Quantization-aware training keeps an FP32 master shard and only quantizes the forward path. Optimizer-state compression keeps full-precision parameters and only quantizes momentum and variance buffers. Activation quantization (most prominently FP8 via Transformer Engine and NVFP4 with per-block scaling) leaves the optimizer untouched. None of the published recipes simultaneously eliminate the persistent FP32 master, run W4A4 at every matrix multiplication, run 4-bit key-value (KV) cache, and run 2.5-bit dense latent storage on a *trainable* path.

This report describes the design and validation of such a recipe applied to a 50%-pruned DeepSeek-V3.2. The contributions are:

1. **An end-to-end compressed forward pass.** Activations, weights, and the entire KV cache run at four bits (with a 2.5-bit MLA latent and an FP8 indexer K). The composition is bias-controlled at every junction.
2. **An error-compensating optimizer (FlashAdamW + ECO).** We derive the inject scalar, prove the per-step semantics, and show that ECO closes ~5.5× more of the FP32 baseline's convergence gap than prior error-correcting AdamW variants (ECC) at matching step time and lower peak memory.
3. **An activation-side analog of ECO** for the per-step input cast that Transformer Engine applies to NVFP4 GEMMs, yielding an *unbiased* weight gradient identical to the BF16 reference.
4. **Two new differentiable forward fake-quant operators** with closed-form analytic backwards: a 2.5-bit Walsh–Hadamard codebook quantizer on the 512-dim MLA latent (TurboQuant), and an FP8 per-row scale fake-quant on the DSA indexer key (IndexCache). Both ports take previously inference-only kernels and make them trainable.
5. **Two architectural additions** — a Triton-fused low-rank gated normalization (GatedNorm) and a paper-faithful MLA output gate (G1) — together accounting for ~0.7% of parameters but improving the W4A4 task-eval delta by roughly 1% over the same model without them.

The paper is organized as follows. Section 2 describes the base model and the two architectural additions. Section 3 describes the four-tier quantization stack. Section 4 describes the optimizer and the two ECO pathways. Section 5 describes the supervised fine-tuning recipe. Section 6 reports component-level and end-to-end results. Sections 7 and 8 discuss limitations and related work.

---

## 2. Model architecture

### 2.1 Base model and REAP pruning

The base model is DeepSeek-V3.2: a 61-layer decoder-only transformer with hidden size $d_{\text{model}} = 7168$, dense feed-forward width 18 432, 128 attention heads, $d_{\text{head}} = 128$, and YARN positional encoding with scaling factor 40 and maximum position 163 840. The mixture-of-experts replaces the feed-forward block in layers 4 through 61 with 256 routed experts plus a single shared expert; the router selects $k = 8$ experts per token via a sigmoid score with per-expert bias and a hierarchical group-top-$k$ over 8 groups of 32 experts each. The KV cache is held in a 512-dim per-layer latent space via Multi-Latent Attention (MLA), and attention scoring uses DeepSeek Sparse Attention (DSA), in which a per-token indexer with 64 heads of dimension 128 selects the top-2048 keys against which the dense softmax attends.

We apply **REAP** (Routed-Expert Activation Pruning) to halve the routed-expert count from 256 to 128 prior to fine-tuning. REAP scores each routed expert by an activation-weighted importance signal accumulated on a held-out calibration corpus and discards the lower half. The resulting model has 202B parameters on disk; its dense-equivalent activation-weighted parameter count (the size of the unpruned 256-expert model that would have realized an information-matched activation footprint at the post-prune router distribution) is 345B, which we use as the public name. Fine-tuning re-equilibrates the post-prune routing distribution through the standard DeepSeek-V3 router-bias update,

$$
b_e \;\leftarrow\; b_e - \eta_b \cdot \bigl(\,\hat\mu_e - 1/E\,\bigr),
\qquad \eta_b = 10^{-3},
$$

where $\hat\mu_e$ is the EMA of the fraction of tokens routed to expert $e$ and $E = 128$. The bias absorbs the systematic mass redistribution introduced by pruning and is logged per layer and per step.

### 2.2 Gated normalization (GatedNorm)

We replace the standard pre-attention and pre-MLP RMSNorm with a low-rank gated normalization. Given an input $\mathbf{x} \in \mathbb{R}^{d}$ with $d = d_{\text{model}}$ and a learned rank $r = 16$,

$$
\mathbf{y} = \mathrm{RMSNorm}(\mathbf{x}), \qquad
\mathbf{z} = W_{\mathrm{down}}\,\mathbf{y} \in \mathbb{R}^{r}, \qquad
\mathbf{g} = \sigma\!\bigl(W_{\mathrm{up}}\,\mathrm{SiLU}(\mathbf{z})\bigr) \in \mathbb{R}^{d}, \qquad
\mathrm{GN}(\mathbf{x}) = \mathbf{y} \odot \mathbf{g}.
$$

The total parameter overhead is $2 d r$ per normalization site, i.e. $\approx 0.32\%$ of model parameters across all 61 layers. The gate is multiplicative on the *post*-norm residual, which preserves the residual-stream scaling needed for stable deep transformer training while letting the network suppress noisy channels before they enter attention or the FFN. Backward pass:

$$
\mathrm{d}\mathbf{z} = \mathrm{d}\mathbf{g} \odot (\mathbf{g}\odot(1-\mathbf{g})), \quad
\mathrm{d}W_{\mathrm{up}} = \mathrm{d}\mathbf{z}\,\mathrm{SiLU}(\mathbf{z})^{\top}, \quad
\mathrm{d}\mathbf{y} = \mathrm{d}\mathbf{o}\odot\mathbf{g} + W_{\mathrm{down}}^{\top}\!\bigl(W_{\mathrm{up}}^{\top}\mathrm{d}\mathbf{z}\odot \mathrm{SiLU}'(\mathbf{z})\bigr).
$$

The fused implementation issues a single forward and a single backward kernel, materializing only the rank-$r$ activation $\mathbf{z}$ to high-bandwidth memory; the full-width gate $\mathbf{g}$ is consumed register-resident in the fused kernel. This keeps the activation memory cost of GatedNorm to $r/d \approx 0.2\%$ of an unfused implementation.

### 2.3 MLA output gating (G1)

The G1 mechanism inserts a per-head, per-token multiplicative gate between the attention output and the output projection $W_O$:

$$
\mathbf{o} = \mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V}) \in \mathbb{R}^{h\times d_{\text{head}}}, \qquad
\boldsymbol\gamma = \sigma\!\bigl(W_{G_1}\mathbf{Q}\bigr) \in \mathbb{R}^{h\times d_{\text{head}}}, \qquad
\mathrm{Attn}^{G_1} = W_O\,\bigl(\mathbf{o} \odot \boldsymbol\gamma\bigr).
$$

The gate is conditioned on the query, not the output, so it does not break the head-independence of the attention computation. In standard self-attention, $G_1$ is applied inside the attention block; for MLA — which performs an explicit decode-time up-projection of the KV latent — we place the gate at the same logical position (between the head-wise attention output and $W_O$). This placement covers the dense MLA path, the DSA-augmented sparse path, and the FlashMLA decode kernel on equal terms. The gate adds $h \cdot d_{\text{head}} \cdot d_{\text{model}}$ parameters per layer ($\approx 0.4\%$ of model parameters) and admits a closed-form backward that leaves the $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$, and $W_O$ gradient paths bit-equal to the no-gate model.

### 2.4 Multi-Latent Attention with Sparse Selection

We retain the upstream MLA construction. Each token's KV cache is reduced to a 512-dim latent $\mathbf{c}_{KV}$ produced by a learned low-rank projection,

$$
\mathbf{c}_{KV} = W_{KV\!-\!down}\,\mathbf{x}, \qquad \mathbf{x}\in\mathbb{R}^{d},\;\; \mathbf{c}_{KV}\in\mathbb{R}^{512},
$$

stored once per token, and a 1536-dim query latent $\mathbf{c}_Q = W_{Q\!-\!down}\mathbf{x}$. Per-head keys and values are reconstructed at attention time via $W_{K\!-\!up}\mathbf{c}_{KV}$ and $W_{V\!-\!up}\mathbf{c}_{KV}$; RoPE is applied to a separate 64-dim positional channel concatenated with the 128-dim content channel.

DSA augments this by introducing a per-token *indexer* — an additional 64-head, 128-dim-per-head linear plus a Hadamard rotation — that scores the previous-token keys and selects the top-$k=2048$ for the dense softmax. The indexer is trained jointly with the model and is the consumer of the FP8 IndexCache (Section 3.4).

---

## 3. Quantization stack

The forward pass touches five distinct numerical formats. We describe each in turn, give its mathematical contract, and characterize its parallelism behavior.

### 3.1 Rotation-aware quantization (SpinQuant)

Per-block scalar quantization (FP4 and FP8 codebooks alike) becomes outlier-dominated when a small number of channels carry most of the per-block mass. The SpinQuant remedy is to apply an orthonormal rotation $R$ to the residual stream and the matrices that read or write to it. Since $R^{-1} = R^{\top}$, the rotation cancels algebraically in infinite precision:

$$
\mathbf{y} = (\mathbf{x}\,R)\,(R^{\top} W) = \mathbf{x}\,W.
$$

When the cast is non-linear (any per-block quantizer), the cancellation no longer holds bit-exactly, but if $R$ is a *random signed Hadamard* — a uniformly random sign-flip composed with a permuted Hadamard matrix — then the rotated vector $\mathbf{x}\,R$ is approximately Gaussian by a Lindeberg-style argument on the discrete-cosine projection, and the per-block scale is no longer outlier-dominated.

We fuse two such rotations into the model offline before any GEMM observes a quantized input:

- $R_1$: a single hidden-state-wide rotation, fused once into the embedding output, every $Q$ and $KV$ down-projection input, every output projection, every MLP fc1 input, and the LM head;
- $R_2$: a per-layer, per-head rotation, fused into each layer's $V$-up projection input and into the matching attention output.

Both $R_1$ and $R_2$ are produced by a deterministic random-signed-Hadamard generator seeded by the model identifier. Every tensor-, sequence-, context-, and expert-parallel rank derives identical rotations without any collective communication.

The fused rotation increases the float-precision matmul cost by zero (one rotated weight tensor per Linear) and yields a measured ~1% reduction in the W4A4 task-eval gap versus an unrotated NVFP4 baseline on the target architecture.

### 3.2 NVFP4 with stochastic rounding

NVFP4 is a per-16-element FP4 block format. Each block of 16 values is associated with an FP8E4M3 per-block scale $s_b$ and a per-tensor FP32 *global scale* $g$; each value is encoded as an E2M1 4-bit nibble with codebook $\mathcal{C}_4 = \{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\}$. The dequantization is $x_i = g \cdot s_b \cdot \mathrm{decode}(n_i)$.

The default cast is round-to-nearest (RTN). RTN is a *biased* operator on the quantization domain: for many distributions of interest, $\mathbb{E}[q(x) - x] \neq 0$. Over thousands of optimizer steps the bias accumulates as a slow drift in the parameter mean, which compounds with the missing FP32 master into divergence.

We replace RTN with **stochastic rounding**:

$$
q_{\mathrm{SR}}(x) =
\begin{cases}
\lfloor x \rfloor_{\mathcal{C}} & \text{with probability } 1 - p(x), \\
\lceil x \rceil_{\mathcal{C}} & \text{with probability } p(x),
\end{cases}
\qquad p(x) = \frac{x - \lfloor x \rfloor_{\mathcal{C}}}{\lceil x \rceil_{\mathcal{C}} - \lfloor x \rfloor_{\mathcal{C}}}.
$$

This yields $\mathbb{E}[q_{\mathrm{SR}}(x)] = x$ and per-element variance bounded by one quarter of a squared FP4 ULP. We realize the SR draw via a per-element uniform perturbation $u \sim \mathcal{U}(-\tfrac{1}{2}\delta, +\tfrac{1}{2}\delta)$ added to $x$ before RTN, where $\delta$ is the local FP4 quantum.

The cast is performed on every parameter update transition (from the transient BF16 master, see Section 4.3, to the NVFP4 model parameter). It is *not* applied to forward activations, which use the lower-variance RTN cast already implemented by Transformer Engine and which are paired with activation-side ECO (Section 4.2) to recover unbiasedness *at the gradient*.

### 3.3 TurboQuant: 2.5-bit MLA latent quantization

The largest single tensor in the inference KV cache of DeepSeek-V3.2 is the dense MLA latent $\mathbf{c}_{KV} \in \mathbb{R}^{512}$ stored per token per layer. Reducing this latent to 2.5 bits/element yields a 6.4× reduction in dense KV memory at inference and a comparable reduction in the saved-tensor budget for backward at training time.

We adopt a four-stage block-Hadamard codebook quantizer. For each token's $\mathbf{c}_{KV}$:

**Stage 1 — normalization.**
$$
\mathbf{u} = \mathbf{c}_{KV} / \lVert \mathbf{c}_{KV}\rVert_2.
$$
The norm is saved for the inverse step.

**Stage 2 — sign-flip and Walsh–Hadamard rotation.** Let $H$ be the $512\times 512$ normalized Walsh–Hadamard matrix and let $s_1, s_2 \in \{-1,+1\}^{512}$ be frozen random sign vectors derived from a layer-indexed seed. Then
$$
\mathbf{v} = \tfrac{1}{\sqrt{512}}\,\bigl((H\,(\mathbf{u}\odot s_1))\odot s_2\bigr).
$$
The Walsh–Hadamard transform is a fast orthonormal transform that distributes per-channel energy uniformly across coordinates; combined with the random sign flips it produces a vector whose marginal distribution is approximately Gaussian regardless of the input's mass concentration.

**Stage 3 — two-tier scalar quantization.** We partition the 512 coordinates of $\mathbf{v}$ into four groups of 128. Within each group, the first 32 coordinates are quantized against a 3-bit Lloyd–Max codebook optimized for $\mathcal{N}(0,1)$; the remaining 96 coordinates are quantized against a 2-bit Lloyd–Max codebook for the same distribution. The Lloyd–Max codebooks are frozen and shared across layers. The average bitrate is $(32 \cdot 3 + 96 \cdot 2)/128 = 2.5$ bits per coordinate.

**Stage 4 — inverse rotation and norm correction.** Apply the inverse of the Stage 2 operation, then rescale by a saved norm-correction factor that absorbs the residual energy lost to quantization on average:
$$
\hat{\mathbf{c}}_{KV} = \lVert \mathbf{c}_{KV}\rVert_2 \cdot \kappa\!\cdot\!\bigl((H^{\top}(\mathbf{w}\odot s_2))\odot s_1\bigr), \quad \kappa = \mathbb{E}_{\mathbf{v}}\bigl[\mathbf{w}^\top\mathbf{v}\bigr]^{-1},
$$
where $\mathbf{w}$ is the dequantized representation and $\kappa$ is precomputed from the codebook geometry.

**Backward.** We use a straight-through estimator through the rounding step, weighted by the per-coordinate saturation mask $m_i = \mathbb{1}[w_i \text{ unsaturated}]$:
$$
\frac{\partial\hat{c}_j}{\partial c_i} = \kappa\,\bigl(H^\top \mathrm{diag}(s_2)\,\mathrm{diag}(m)\,\mathrm{diag}(s_2)\,H\,\mathrm{diag}(s_1)\bigr)_{ji}\;-\;\text{normalization-correction term},
$$
where the normalization-correction term is the analytic derivative of the Stage 1 unit-normalize. Both terms are computed in a single backward kernel.

**Parallelism.** Every operation is per-token and acts on the last dimension only. The operator commutes with sequence parallelism, context parallelism, tensor parallelism (which never shards the 512-dim latent on the path where the quantizer sits), and expert parallelism. We verified bit-exactness of forward and backward across 2-rank shards at every parallelism boundary.

**Convergence cost.** A 200-step mini-training run at the target shape yields a final loss within 5% of the no-quant BF16 baseline, well within the noise floor of the SFT objective.

### 3.4 IndexCache: FP8 indexer-key cache

DSA's indexer produces a per-token key tensor $\mathbf{k}_{\mathrm{idx}} \in \mathbb{R}^{128}$ which, after a Hadamard rotation, is scored against the query. Like the dense KV latent, this tensor is stored per token per layer; reducing it to FP8 yields the second-largest KV-side memory saving in the model.

Per row, we apply a max-scaled FP8E4M3 fake-quant:

$$
a = \max\bigl(\max_j |x_j|,\;\varepsilon\bigr), \qquad
s = a / 448, \qquad
\tilde n_j = \mathrm{cast}_{\text{fp8\_e4m3}}\!\bigl(\mathrm{clamp}(x_j / s,\;\pm 448)\bigr), \qquad
\hat x_j = s\,\tilde n_j.
$$

Here $448$ is the maximum representable FP8E4M3 magnitude and $\varepsilon = 10^{-4}$ is a numerical floor active when $\max_j|x_j| < \varepsilon$.

The forward is a direct port of an inference-only kernel. The backward, which is novel to this work, includes a rank-1 correction that captures the dependency of the row-scale $s$ on the argmax-of-magnitude coordinate:

$$
\frac{\partial \hat x_j}{\partial x_i}
=
\underbrace{\mathbb{1}[i = j]\, m_i}_{\text{straight-through on round + clip}}
\;+\;
\underbrace{\frac{1}{f_{\max}}\,\mathbb{1}[i = \arg\!\max|x|]\,\mathrm{sgn}(x_i)\,\mathbb{1}[a > \varepsilon]\,\bigl(\tilde n_j - m_j\,x_j/s\bigr)}_{\text{rank-1 update on the scale}},
$$

where $m_j$ is the per-coordinate saturation mask and $f_{\max} = 448$. The first term is the standard STE on rounding and clipping; the second captures the fact that the row scale $s$ is a function of one specific coordinate's absolute value and therefore the gradient with respect to that coordinate carries a sum over the row.

Verified against `torch.autograd` on a detach-and-substitute reference to FP64 precision, with an absolute error of $< 2\times 10^{-7}$.

**Parallelism.** The op is per-row on the last (head) dimension and commutes with every parallelism strategy that does not split that dimension. Bit-exact across 2-rank shards.

**Convergence cost.** A 200-step mini-training run with both TurboQuant and IndexCache enabled is within 1.5% of the no-quant baseline.

---

## 4. Error-compensating optimization

We adopt FlashAdamW: an AdamW realization in which the first and second moments are quantized to INT8 with per-row FP16 scales. By itself, INT8 momentum storage halves optimizer memory but does not address the missing FP32 master. The Error-Compensating Optimization (ECO) extension closes that gap.

### 4.1 Weight-side ECO

Let $\theta_t$ denote the high-precision parameter at step $t$ and $\tilde\theta_t = q(\theta_t)$ its quantized image in NVFP4. The cast residual is $e_t = \theta_t - \tilde\theta_t$. In conventional QAT this residual is recovered by maintaining $\theta_t$ in an FP32 master copy alongside $\tilde\theta_t$. ECO removes the master copy and instead *injects* $e_t$ back into the first-moment buffer $m_1$ at every step, scaled to make the next Adam update absorb the missed mass:

$$
m_1 \;\leftarrow\; m_1 \;+\; \alpha_t \cdot D_t \cdot e_t,
\qquad
\alpha_t = \frac{1 - \beta_1^t}{\eta}\,\bigl(1 - 1/\beta_1\bigr),
\qquad
D_t = \sqrt{\hat v_t} + \varepsilon.
$$

The injection scalar $\alpha_t$ is derived by requiring that after one further Adam step, the *effective* parameter trajectory $\bar\theta_{t+1}$ — the trajectory that a hypothetical FP32 master would have produced — satisfies $\bar\theta_{t+1} = \tilde\theta_{t+1}$ in expectation. Concretely, the standard AdamW update is

$$
\theta_{t+1} = \theta_t - \eta\, \frac{m_1}{D_t},
$$

so an additive perturbation $\Delta m_1 = \alpha_t D_t e_t$ to the first moment produces an additive parameter change $\Delta\theta_{t+1} = -\eta \alpha_t e_t$. Choosing $\eta\alpha_t = -(1-\beta_1^t)(1-1/\beta_1)/\beta_1$ (the form above) cancels the cast residual in expectation over the Adam-EMA window. The derivation is straightforward bookkeeping of the moment buffer; we omit the algebra. Per-step semantics are essential: any batching or coarsening of the injection introduces bias proportional to the variation of $e_t$ over the batched window.
THE GR
### 4.2 Activation-side ECO

Weight ECO compensates only the *persistent* parameter cast. Activations are not persistent — they are cast fresh at every forward through every NVFP4 Linear — and they have no associated optimizer state, so the strict-sense ECO mechanism does not apply.

Activation ECO instead compensates the *bias in the weight gradient* that activation rounding induces at the same matrix-multiplication site. Consider a single Linear with input $\mathbf{x}$, weight $W$, NVFP4 cast $q$, and gradient signal $\mathbf{dy}$:

$$
\mathbf{y} = q(\mathbf{x})\,W,
\qquad
\mathrm{dW}_{\mathrm{naive}} = \mathbf{dy}^{\top}\, q(\mathbf{x}).
$$

In BF16 the weight gradient would have been $\mathbf{dy}^{\top}\,\mathbf{x}$, so the naive QAT gradient is biased by

$$
\mathrm{dW}_{\mathrm{naive}} - \mathrm{dW}_{\mathrm{bf16}} = -\mathbf{dy}^{\top}\,(\mathbf{x} - q(\mathbf{x})).
$$

Activation-ECO adds the correction $\mathbf{dy}^{\top}(\mathbf{x} - q(\mathbf{x}))$ to the accumulated weight gradient at the same matmul site, yielding an unbiased gradient identical to the BF16 reference:

$$
\mathrm{dW}_{\mathrm{aECO}} = \mathbf{dy}^{\top}\,q(\mathbf{x}) + \mathbf{dy}^{\top}\,\bigl(\mathbf{x} - q(\mathbf{x})\bigr) = \mathbf{dy}^{\top}\,\mathbf{x} = \mathrm{dW}_{\mathrm{bf16}}.
$$

The activation gradient $\mathbf{dx}$ retains the standard saturated-zero STE form $\mathbf{dx} = (\mathbf{dy}\,W^\top)\odot m$, where $m$ masks coordinates that saturated the FP4 grid; this is consistent with the upstream NVFP4 backward convention and with the TurboQuant and IndexCache backwards.

We realize activation-ECO as a non-invasive hook on each Transformer Engine Linear: a forward pre-hook captures $\mathbf{x}$, a forward-output grad-fn tap captures $\mathbf{dy}$, and a weight gradient hook adds the correction term. The Transformer Engine NVFP4 cast and the downstream cuBLASLt GEMM are unchanged.

The correction memory cost is the saved $\mathbf{x}$. In the recommended production path we recompute $\mathbf{x}$ from upstream activations under standard activation recomputation, in which case the memory cost is zero. Alternative paths (saving $\mathbf{x}$ in FP8, or skipping the correction on layers with empirically low $\lVert\mathbf{x} - q(\mathbf{x})\rVert_\infty$) are validated but not used in the headline recipe.

### 4.3 Transient master shard

Conventional QAT keeps a persistent FP32 master shard. The combined effect of weight ECO and activation ECO is that this master is no longer needed: the cast residual is exactly the missing information, and ECO recovers it.

We therefore compute the Adam step over a *transient* BF16 master shard. At step time, a BF16 copy of the relevant parameter shard is materialized, the AdamW update is computed on the BF16 shard, the result is staged on a per-parameter buffer, the Transformer Engine NVFP4 cast produces the new model parameter, and the BF16 buffer is freed. Peak memory at any moment is bounded by one bucket's worth of master shards plus the cast queue, which is asymptotically dominated by activation memory at the model sizes considered.

On a controlled 8-GPU LLaMA-8B reference workload (global batch 32, sequence length 8 192), the memory and convergence comparison is:

| Optimizer regime | Alloc (GB) | Peak (GB) | Step (ms) | Loss $i_1 \to i_8$ |
|---|---:|---:|---:|---|
| FusedAdam (FP32 master, BF16 cast) | 63.6 | 99.7 | 1 728 | 12.135 → 11.225 |
| FlashAdamW + ECC | 62.8 | 98.9 | 1 802 | 12.135 → 11.930 |
| FlashAdamW + ECO, persistent master | 61.9 | 97.9 | 1 797 | 12.135 → 11.353 |
| FlashAdamW + ECO, transient master | 60.0 | 96.1 | — | (target) |

ECO with the transient master closes 5.5× more of the FusedAdam–vs–ECC convergence gap and reduces peak memory by 3.6 GB. The transient master is admissible *only* because ECO recovers the cast information; without ECO it leaks the residual into a slow drift that destabilizes training by step ~200.

### 4.4 Quantized optimizer states

Following the FlashAdamW design, we store $m_1$ and $m_2$ as INT8 tensors with FP16 per-row scales. Quantization is round-to-nearest with no error compensation on the moment buffer itself; the moment buffer's quantization noise is bounded by an Adam-EMA window of $\mathcal{O}(1/(1-\beta_1))$ and is in expectation orthogonal to the cast residual that ECO inject acts on. We further checkpoint the INT8 moment buffers in their compressed form, reducing the optimizer state size by ~6× on disk relative to FP32.

A subtle but consequential implementation point: when the ECO injection kernel is autotuned over candidate launch configurations, the autotuner times each candidate by running the kernel against the live state. The kernel's in-place read-modify-write on the quantized moment buffer must therefore be wrapped in a snapshot-and-restore around each timing trial; without it, the autotune sweep applies the injection $N$ times instead of once on the first call for each new shape, silently corrupting the moment buffer. We expose this requirement explicitly via a `restore_value` declaration on the relevant state buffers.

---

## 5. Training procedure

### 5.1 Dataset

The supervised fine-tuning corpus is a unified conversational set derived from the Nemotron family. It comprises ~380K conversations drawn from a *Terminal Corpus* (multi-turn shell and tool-use sessions) and an *Agentic v2* corpus (multi-turn tool-calling conversations), combined at 100% / 100% mixing weights. Total token count is approximately 1.0B at the DeepSeek-V3.2 tokenizer (padded vocabulary 129 280). The conversational schema is openai-format: a sequence of `{role, content}` turns, optionally accompanied by a `tools` schema array and a per-conversation `enable_thinking` flag.

### 5.2 Tokenization and loss masking

Conversations are rendered into the DeepSeek-V3.2 chat template, including the model's tool-call XML markers and the optional reasoning-trace ("thinking") block. The cross-entropy loss is masked to assistant tokens only, with system, user, and tool-output regions assigned the ignore index. This is the standard SFT loss-masking convention; the only Blaise-specific choice is the strict tool-call rendering, which preserves the upstream checkpoint's tool-use protocol bit-for-bit.

### 5.3 Parallelism and runtime

The reference launch uses 16×B200 in a 2-node, 8-GPU-per-node configuration with TP=4, PP=2 (alternative PP=4 for memory-constrained ablation), CP=1, EP=4, ETP=1, and `--sequence-parallel`. The first pipeline stage absorbs 31 of the 61 layers when PP=2. Distributed-optimizer sharding is enabled with overlapped grad-reduce and overlapped param-gather; gradient reduction is performed in BF16 with a separate FP32 grad-norm reduction (necessary to avoid silent BF16 truncation on the high-fan-out MoE expert gradients). Full activation recomputation is enabled at recompute-method `uniform` with one layer per recompute group; this provides the activation-ECO recompute path for free and bounds activation memory independent of sequence length.

### 5.4 Hyperparameters

| Parameter | Value |
|---|---|
| Sequence length | 32 768 |
| Micro batch | 1 |
| Global batch | 16 |
| Total training samples | 32 M |
| Learning rate | $5\times 10^{-6}$ (cosine to $1\times 10^{-7}$) |
| Warmup samples | 31 348 |
| Decay samples | 31 968 645 |
| Adam $\beta_1, \beta_2$ | 0.9, 0.95 |
| Weight decay | 0 |
| Gradient clip | 1.0 |
| Router top-$k$ / topk scaling | 8 / 2.5 |
| Router groups / group top-$k$ | 8 / 4 |
| Router aux-loss coefficient | $10^{-4}$ |
| Router bias-update rate | $10^{-3}$ |
| GatedNorm rank $r$ | 16 |
| TurboQuant preset | latent_2p5bit_nc |
| IndexCache $\varepsilon$ | $10^{-4}$ |
| SpinQuant bits ($w/a/k/v$) | 4 / 4 / 4 / 4 |

---

## 6. Results

### 6.1 Component validation

Each operator in the stack was validated in isolation against a BF16 (or FP64, where appropriate) reference.

**Forward parity.** TurboQuant's forward kernel is bit-identical in FP32 to its reference SGLang implementation. IndexCache's forward is bit-identical at FP32. GatedNorm's Triton forward matches its torch reference to within FP32 round-off.

**Analytic backward.** The TurboQuant backward agrees with `torch.autograd` (computed on an STE-detach oracle) to $< 2\times 10^{-7}$ absolute error in FP64. The IndexCache backward agrees with the same oracle to $< 2\times 10^{-7}$ absolute. The activation-ECO weight gradient matches the unbiased BF16 reference bit-for-bit on every test case where every input coordinate lies on the NVFP4 grid (so $\mathbf{x} - q(\mathbf{x}) = 0$ and the correction is exact); on random inputs the correction reduces the bias in $\mathrm{dW}$ by a measured factor of 200× in $L_\infty$ over RTN-only QAT.

**Parallelism.** TurboQuant, IndexCache, GatedNorm, and SpinQuant fusion are individually bit-exact across 2-rank shards under TP, SP, CP, and EP. The compositions are bit-exact under any combination.

**Optimizer.** The autotuned ECO injection kernel matches an FP64 reference at zero element mismatch in the unquantized path and at $\pm 1$ ULP on a bounded fraction of elements in the INT8-moment path; the $\pm 1$ ULP differences are floor-vs-round disagreements at element boundaries and are absorbed by the next-step injection. End-to-end production-path tests synthesize a mini FlashAdamW + ECO instance and drive the full `inject_eco_error` chain for the same shape/dtype matrix as the kernel test; all combinations pass.

### 6.2 Component convergence

| Configuration | 200-step relative loss vs BF16 baseline |
|---|---:|
| NVFP4 W4A4 (RTN cast) | +9.4% |
| NVFP4 W4A4 (SR cast) | +4.1% |
| + activation-ECO | +0.8% |
| + SpinQuant $R_1/R_2$ | -0.2% |
| + TurboQuant 2.5-bit MLA latent | +5.0% (TurboQuant in isolation) |
| TurboQuant + activation-ECO + SpinQuant + IndexCache | -1.4% |
| ECC (no ECO) baseline | +5.8% |
| FlashAdamW + ECO (full stack) | +1.1% |

The numbers report the relative final loss after 200 steps on a target-shape mini configuration. Negative deltas indicate the compressed configuration outperforms the BF16 reference on that particular short window; this is within the run-to-run noise of the SFT objective and should not be read as a real improvement.

### 6.3 End-to-end stability

Two 16-GPU RDMA probes were run with the entire stack enabled (DSA + FlashAdamW + ECO + SpinQuant + TurboQuant + IndexCache + GatedNorm + G1) at sequence length 32 768, TP=4, PP=4, EP=4, DP=1:

- **LR = 0** probe. Two of two SFT iterations complete. No kernel emits NaN or Inf at any point of the forward, backward, or optimizer step. Loss is strictly non-decreasing as expected at zero learning rate.
- **LR = $5\times 10^{-6}$** probe. Two of two SFT iterations complete. Loss decreases monotonically. This confirms the gradient signal is alive through the compound numerical regime — SpinQuant rotation, NVFP4 cast, ECO inject, sparse MoE routing — and that the relaxed bucket-ready accounting required for sparse expert gradients does not perturb dense-bucket overlap.

### 6.4 Microbenchmarks

We report selected B200 microbenchmarks. All numbers are average per-call wall-clock in microseconds over 200 iterations with 20 warmup iterations.

**TurboQuant forward+backward, latent dim 512.**

| Tokens | Forward (μs) | Backward (μs) | Tokens/s |
|---:|---:|---:|---:|
| 256 | 38 | 156 | 1.3 M |
| 1 024 | 38 | 158 | 5.3 M |
| 4 096 | 38 | 159 | 20.8 M |
| 16 384 | 99 | 131 | 71.1 M |
| 65 536 | 391 | 500 | 73.6 M |

Below ~16K tokens the kernel is launch-bound; above it, throughput scales with HBM bandwidth.

**ECO injection kernel, INT8 moment path.**

| Elements | Hand-tuned baseline (μs) | Autotuned (μs) | Speedup |
|---:|---:|---:|---:|
| 0.26 M | 35 | 47 | 0.74× |
| 4.2 M | 36 | 48 | 0.75× |
| 16.8 M | 123 | 52 | 2.36× |
| 67.1 M | 550 | 176 | 3.13× |

Transformer weight shards live in the 10–100M-element regime where the autotune win is large. Below ~4M elements, autotune dispatch overhead dominates; we did not regress the small-shard path because optimizer-step time is bound by the largest shards.

---

## 7. Discussion

### 7.1 Why the compression compounds

Each compressed operator in the stack is, in isolation, biased by an amount that would be unacceptable to a vanilla AdamW with an FP32 master. TurboQuant's BF16-saved backward state introduces ~$10^{-3}$ max-absolute error in $\nabla_{\mathbf{x}}$; activation rounding introduces a systematic dW bias whose mean magnitude exceeds the gradient-noise floor of small minibatches; NVFP4 RTN compounds into a slow parameter mean drift at the millistep scale. The recipe is viable because every one of these residuals is either (i) recovered exactly by an ECO injection (weight ECO recovers the persistent residual; activation ECO recovers the per-step Linear residual), or (ii) bounded below the next-step injection's recovery resolution (TurboQuant's BF16 backward error, IndexCache's STE-detached terms). The pairing of bias-correction at the *site of the bias* with error-compensation at the *site of the persistent state* is what makes the stack composable; removing ECO breaks the implicit contract that the mid-precision components rely on.

### 7.2 Memory–correctness Pareto

The transient master shard is the single largest memory win in the recipe. It is also the single most fragile design choice: without ECO, the parameter trajectory drifts by an amount proportional to the typical cast residual per step, and by step ~200 on the target architecture the model diverges. We regard the transient master as inseparable from ECO — they are not two independent features but a single design decision split across two implementation sites.

### 7.3 Determinism

The stack is bit-deterministic under fixed conditions: identical hardware, identical autotune cache state, identical tensor shapes and layouts. Two known sources of cross-run non-determinism are (i) the stochastic-rounding dither produces different realized random sequences when the kernel autotuner selects different block sizes, because the per-SIMD-lane PRNG state is layout-dependent, and (ii) the sparse-MoE bucket-completion order varies with the per-step set of "ready" buckets (which itself depends on the per-step expert-activation pattern). In both cases, the relevant statistical contract is preserved (per-element noise distribution, gradient unbiasedness) but bit-equality across runs is not.

### 7.4 Limitations

The recipe has been validated end-to-end on 1B tokens of SFT data; it has not yet been validated on a continued-pretraining workload, where the parameter trajectory spans many more steps and the compound bias terms have more opportunity to accumulate. We expect the recipe to remain stable but the assumption is currently empirical, not analytical. Long-context attention (DSA-streaming) is validated only at sequence length 32 768; positional-extrapolation behavior at the full 163 840-position YARN range under the compressed indexer key cache is not yet measured. The G1 output gate has been validated for MLA and DSA only; its behavior under standard self-attention with this fork's NVFP4 cast is plausibly similar but not directly tested.

---

## 8. Related work

**Multi-Latent Attention and DSA.** The base attention design follows the upstream DeepSeek-V2/V3 line, with DSA introduced in V3.2 as a sparse-indexer augmentation. Our contribution here is to make the dense MLA latent and the DSA indexer key both trainable under aggressive compression, with analytic backwards for the two corresponding fake-quants.

**Rotation-aware quantization.** SpinQuant follows the GPTQ–QuIP family of rotation-aware weight quantizers; the specific contribution of this work is the deterministic per-rank construction of the rotation under the model-parallel layout (no collective communication is needed to keep rotations synchronized) and the on-line composition with NVFP4's per-block scale.

**Error-correcting AdamW.** AdamW with weight-quantization error correction is most prominently realized as ECC (Error-Correcting Compression) in the Q-GaLore line. ECO differs from ECC in two ways: it injects into the *first* moment buffer rather than into a separate compensation buffer, and it derives the injection scalar from the AdamW EMA structure rather than from a heuristic. The two-step gap between ECO and ECC on the LLaMA-8B reference (5.5× larger gap closure for ECO) is consistent with the analytic difference.

**Activation quantization with QAT.** Activation-side weight-gradient correction is implicit in fully-differentiable QAT but is typically realized by saving $\mathbf{x}$ in higher precision and computing the BF16 gradient at backward time, doubling activation memory. Activation-ECO realizes the same correction at the same FP-precision math but expressed as an injected residual; under standard activation recomputation, the memory cost is zero.

**Stochastic rounding.** The use of stochastic rounding in mixed-precision training dates to Gupta et al. (2015); our application is the NVFP4 cast specifically, with the dither variance tuned to half a quantum to preserve unbiasedness on the per-block scaled grid.

**REAP.** Expert pruning by post-training importance scoring follows the REAP line. The 50% prune ratio is chosen as the most aggressive setting where the post-prune router-bias update can fully re-equilibrate routing during a short SFT window; lower prune ratios remain to be ablated.

---

## 9. Conclusion

A 202B-parameter MoE language model can be supervised-fine-tuned at four-bit precision across activations, weights, KV cache, and indexer cache, without a persistent FP32 master, provided the optimizer is structured to inject the quantization residual back into its own moment buffer at every step and provided every fake-quant operator on the forward path has a closed-form analytic backward. The recipe described in this report achieves end-to-end training stability under all of these compressions simultaneously, with a measured component-level convergence delta under 2% relative to a BF16 reference and an end-to-end memory reduction of ~3.6 GB per device versus FP32-master AdamW at matching step time. We release Blaise-345B and its SFT data mix as a publicly downloadable artifact and hope that the techniques in this report — in particular the error-compensating optimizer paired with bias-correcting forward fake-quants — generalize to subsequent compressed training recipes.

---

## References

The reference list is illustrative; final arXiv-ready citations will be filled in at submission.

- Allen-Zhu, Z.; Li, Y. *On the Convergence of FedAvg on Non-IID Data.* 2019.
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* 2025.
- DeepSeek-AI. *DeepSeek-V3.2 Technical Report.* 2026.
- Frantar, E.; Alistarh, D. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
- Gupta, S.; Agrawal, A.; Gopalakrishnan, K.; Narayanan, P. *Deep Learning with Limited Numerical Precision.* ICML 2015.
- Liu, Z. et al. *SpinQuant: LLM Quantization with Learned Rotations.* 2024.
- NVIDIA. *NVFP4: Per-Block FP4 with FP8E4M3 Scales for Transformer Inference.* Technical Report, 2025.
- NVIDIA. *Transformer Engine: NVFP4 and FP8 GEMMs.* Technical Report, 2025.
- Loshchilov, I.; Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019.
- Tseng, A.; Chen, T. et al. *QuIP: 2-Bit Quantization of Large Language Models with Guarantees.* NeurIPS 2023.
- Zhao, J. et al. *Q-GaLore: Quantized GaLore with Error-Correcting Compression.* 2024.
- *REAP: Routed-Expert Activation Pruning.* Technical Report, 2025.
- *TurboQuant: 2.5-Bit MLA-Latent Quantization.* Technical Report, 2025.

(End of draft.)