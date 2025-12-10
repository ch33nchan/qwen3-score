# Inference Scaling in Diffusion Models: A Mathematical Framework

We often talk about scaling laws in training—throw more data and compute at a model, and validation loss goes down predictably. But what about **inference-time compute scaling**? What if we could trade GPU hours for quality *at runtime*, after the model is already trained?

This is the core question we explore in **EACPS (Evolutionary Annealing with Candidate Potential Scoring)**. In this writeup, we treat the stochastic sampling process of diffusion models as a **discrete optimization problem** over the space of random seeds, and show how we can systematically improve output quality by scaling inference compute.

## The Fundamental Problem: Sampling vs. Optimization

### The Stochastic Nature of Diffusion

Consider a conditional diffusion model that maps noise $z_T \sim \mathcal{N}(0, I)$ to samples $x_0$ given conditioning $c$:

$$
x_0 = \mathcal{F}_\theta(z_T, c; s)
$$

where $s$ is the random seed that determines the noise trajectory. The model implicitly defines a distribution:

$$
p_\theta(x_0 \mid c) = \int p(z_T) \, p_\theta(x_0 \mid z_T, c) \, dz_T
$$

For a fixed conditioning $c$, each seed $s$ yields a **different sample** from this distribution. Critically, not all samples are created equal. The distribution $p_\theta(x_0 \mid c)$ has high variance—some seeds produce excellent results, others produce failures.

### The Variance Problem

Let $Q(x)$ be a quality metric (e.g., human preference score, FID, CLIP similarity). The expected quality under random sampling is:

$$
\mathbb{E}_{s \sim \mathcal{U}}[Q(\mathcal{F}_\theta(s, c))] = \int Q(x) \, p_\theta(x \mid c) \, dx
$$

But we don't want the *expectation*—we want samples from the **tail of the quality distribution**. We want to solve:

$$
s^* = \mathop{\mathrm{argmax}}_{s \in \mathcal{S}} Q(\mathcal{F}_\theta(s, c))
$$

This is a **discrete black-box optimization problem**. We cannot differentiate through $\mathcal{F}_\theta$ with respect to $s$ (the seed is discrete), and evaluating $Q(\cdot)$ is expensive (requires VLM inference or human evaluation).

## EACPS: Evolutionary Search Over the Seed Space

We treat the seed space $\mathcal{S}$ as a **combinatorial landscape** and apply evolutionary optimization.

### Stage 1: Global Exploration (Broad Search)

We begin with Monte Carlo sampling. Draw $K_{\mathrm{global}}$ seeds uniformly:

$$
\mathcal{S}_{\mathrm{global}} = \{s_1, s_2, \ldots, s_{K_{\mathrm{global}}}\}, \quad s_i \sim \mathcal{U}(0, 2^{32})
$$

For each seed, we generate a candidate and evaluate it:

$$
\mathcal{P}_{\mathrm{global}} = \left\{ (x_i, U_i) \mid x_i = \mathcal{F}_\theta(s_i, c), \, U_i = U(x_i) \right\}_{i=1}^{K_{\mathrm{global}}}
$$

where $U(x)$ is our **potential function** (more on this below). We then rank candidates by potential and select the top $M$ performers:

$$
\mathcal{S}_{\mathrm{elite}} = \mathop{\mathrm{top}}_M \{ s_i \mid U_i = U(\mathcal{F}_\theta(s_i, c)) \}
$$

**Key insight:** Even with random sampling, the maximum of $K_{\mathrm{global}}$ draws concentrates in the tail of the quality distribution. By the order statistics of i.i.d. samples:

$$
\mathbb{E}[\max_{i=1}^K Q_i] \approx \mu + \sigma \sqrt{2 \ln K}
$$

where $\mu, \sigma$ are the mean and standard deviation of $Q$. This means **logarithmic returns** to sampling more candidates.

### Stage 2: Local Refinement (Exploitation)

Now we exploit the structure of the seed space. We hypothesize that nearby seeds produce **correlated outputs**. This is not obvious—deterministic chaos in the diffusion ODE could destroy all correlation—but empirically, we observe that seed $s$ and seed $s + \delta$ (for small $\delta$) often share structural features (pose, composition, lighting), differing mainly in high-frequency details (texture, fine edges).

For each elite seed $s_{\mathrm{elite}}$, we spawn $K_{\mathrm{local}}$ children:

$$
\mathcal{S}_{\mathrm{local}}(s_{\mathrm{elite}}) = \{ s_{\mathrm{elite}} + \delta_j \}_{j=1}^{K_{\mathrm{local}}}, \quad \delta_j \in \{1, 2, \ldots, K_{\mathrm{local}}\}
$$

We evaluate each child and collect all candidates:

$$
\mathcal{P}_{\mathrm{local}} = \bigcup_{s \in \mathcal{S}_{\mathrm{elite}}} \left\{ (x_j, U_j) \mid x_j = \mathcal{F}_\theta(s + \delta_j, c) \right\}
$$

The final output is:

$$
x^* = \mathop{\mathrm{argmax}}_{x \in \mathcal{P}_{\mathrm{global}} \cup \mathcal{P}_{\mathrm{local}}} U(x)
$$

### Total Compute Budget

The total number of diffusion forward passes is:

$$
N_{\mathrm{total}} = K_{\mathrm{global}} + M \cdot K_{\mathrm{local}}
$$

For typical hyperparameters ($K_{\mathrm{global}} = 8, M = 3, K_{\mathrm{local}} = 4$), we evaluate $N = 8 + 3 \times 4 = 20$ candidates.

## The Potential Function $U(x)$: Multi-Objective Scoring

We define a **potential function** $U : \mathcal{X} \to \mathbb{R}$ that maps outputs to scalar quality scores. We use a weighted combination of sub-metrics:

$$
U(x) = \sum_{k=1}^K w_k \cdot v_k(x)
$$

where each $v_k(x) \in [0, 10]$ is a normalized score from a different evaluator (e.g., a VLM). In our implementation:

$$
U(x) = \alpha \cdot v_{\mathrm{consistency}}(x) + \beta \cdot v_{\mathrm{realism}}(x) + \gamma \cdot v_{\mathrm{identity}}(x)
$$

We use **learned weights** $\alpha, \beta, \gamma$ that prioritize photorealism:

$$
\alpha = 1.0, \quad \beta = 4.0, \quad \gamma = 1.5
$$

This is a **scalarization** of a multi-objective problem. We are solving:

$$
\max_{s} \left[ \alpha \cdot v_1(\mathcal{F}_\theta(s, c)) + \beta \cdot v_2(\mathcal{F}_\theta(s, c)) + \gamma \cdot v_3(\mathcal{F}_\theta(s, c)) \right]
$$

The weights encode our **preference model**—what trade-offs we accept between different quality dimensions.

## Theoretical Analysis: Scaling Laws

### Expected Maximum Order Statistics

Let $Q_1, Q_2, \ldots, Q_N$ be i.i.d. quality scores drawn from a distribution $F$ with CDF $F(q) = P(Q \leq q)$. The expected maximum is:

$$
\mathbb{E}[\max_i Q_i] = \int_0^\infty \left(1 - F(q)^N\right) dq
$$

For a Gumbel distribution (common in extreme value theory), this simplifies to:

$$
\mathbb{E}[\max_i Q_i] \approx \mu + \beta \ln N
$$

where $\beta$ is the scale parameter. This gives us **logarithmic scaling**:

$$
Q(N) = Q_{\mathrm{base}} + \beta \ln N
$$

Doubling the compute ($N \to 2N$) yields a constant additive improvement $\beta \ln 2 \approx 0.69\beta$.

### Empirical Validation

In practice, we observe:

$$
Q(N) \approx Q_0 + c \log(N + 1)
$$

where $c$ depends on the task and base model variance. For high-variance tasks (e.g., character consistency), $c$ is large, making inference scaling highly effective.

### Computational Complexity

Each candidate requires one full diffusion forward pass. For a model with $T$ denoising steps, $d$-dimensional latents, and transformer depth $L$:

$$
\mathrm{FLOPs} = N \cdot T \cdot L \cdot d^2
$$

With $N = 20$ candidates and $T = 15$ steps, we do $20 \times 15 = 300$ forward passes per task. This is expensive, but parallelizable across seeds (embarrassingly parallel).

## Architecture Diagram

```mermaid
graph TD
    Start([Input: Condition c, Base Model θ]) --> Stage0[Stage 0: Initialize]
    
    Stage0 --> PriorGen[Generate Prior x_prior]
    PriorGen --> GlobalStage[Stage 1: Global Exploration]
    
    GlobalStage --> SampleSeeds[Sample K_global seeds uniformly]
    SampleSeeds --> GenCandidates[Generate K_global candidates]
    
    GenCandidates --> Parallel1{Parallel Execution}
    Parallel1 --> Gen1[x_1 = F_θ s_1, c]
    Parallel1 --> Gen2[x_2 = F_θ s_2, c]
    Parallel1 --> GenK[x_K = F_θ s_K, c]
    
    Gen1 --> Score1[U_1 = U x_1]
    Gen2 --> Score2[U_2 = U x_2]
    GenK --> ScoreK[U_K = U x_K]
    
    Score1 --> GlobalPool[Pool: P_global = {x_i, U_i}]
    Score2 --> GlobalPool
    ScoreK --> GlobalPool
    
    GlobalPool --> Rank[Rank by Potential U_i]
    Rank --> SelectElite[Select Top M: S_elite]
    
    SelectElite --> LocalStage[Stage 2: Local Refinement]
    
    LocalStage --> ForEachElite{For each s ∈ S_elite}
    ForEachElite --> SpawnChildren[Spawn K_local children: s + δ_j]
    
    SpawnChildren --> Parallel2{Parallel Execution}
    Parallel2 --> GenLocal1[x'_1 = F_θ s+δ_1, c]
    Parallel2 --> GenLocal2[x'_2 = F_θ s+δ_2, c]
    Parallel2 --> GenLocalL[x'_L = F_θ s+δ_L, c]
    
    GenLocal1 --> ScoreLocal1[U'_1 = U x'_1]
    GenLocal2 --> ScoreLocal2[U'_2 = U x'_2]
    GenLocalL --> ScoreLocalL[U'_L = U x'_L]
    
    ScoreLocal1 --> LocalPool[Pool: P_local = {x'_j, U'_j}]
    ScoreLocal2 --> LocalPool
    ScoreLocalL --> LocalPool
    
    LocalPool --> MergeLoop{More elite seeds?}
    MergeLoop -->|Yes| ForEachElite
    MergeLoop -->|No| FinalSelection
    
    FinalSelection[Merge: P_total = P_global ∪ P_local] --> ArgMax[x* = argmax U x]
    ArgMax --> Output([Output: x*])
    
    subgraph "Potential Function U(x)"
        UFunc[U x = Σ w_k · v_k x]
        VLM1[v_realism: VLM Photorealism Check]
        VLM2[v_identity: VLM Face Similarity]
        VLM3[v_consistency: VLM Scene Match]
        
        UFunc --> VLM1
        UFunc --> VLM2
        UFunc --> VLM3
    end
    
    Score1 -.->|Uses| UFunc
    ScoreLocal1 -.->|Uses| UFunc
    
    style GlobalStage fill:#e1f5ff
    style LocalStage fill:#fff5e1
    style UFunc fill:#ffe1f5
    style Output fill:#90EE90
```

## Key Insights

1. **Seed space has structure**: Nearby seeds produce correlated outputs, enabling local search.
2. **Logarithmic returns**: Doubling inference compute yields $O(\log 2)$ improvement.
3. **Embarrassingly parallel**: All candidates can be evaluated in parallel on multiple GPUs.
4. **Post-training scaling**: No need to retrain—this works with any frozen diffusion model.
5. **Domain-agnostic**: The method generalizes beyond our specific task to any conditional generation problem.

## Comparison to Other Approaches

| Method | Compute Scaling | Requires Training | Parallel |
|--------|----------------|-------------------|----------|
| **EACPS (Ours)** | $O(\log N)$ | No | Yes |
| Best-of-N Sampling | $O(\log N)$ | No | Yes |
| Guidance Annealing | $O(1)$ | No | No |
| DPO/RLHF | $O(1)$ | Yes | No |
| Test-Time Training | Linear? | Yes | No |

Our method sits in the "zero-shot test-time optimization" category—we get better results by spending more compute at inference, without any model updates.

## Limitations and Future Work

1. **Compute cost scales linearly** with candidate count (unlike amortized methods like CFG).
2. **Seed correlation is empirical**, not theoretically guaranteed for all models.
3. **VLM scoring has systematic biases**—may reward certain aesthetics over ground truth quality.
4. **No gradient information**—we're doing black-box optimization, which is sample-inefficient.

Future directions:
- **Learned search policies**: Use RL to predict which seeds are promising (reduce $K_{\mathrm{global}}$).
- **Adaptive budgets**: Allocate more compute to harder examples.
- **Hierarchical search**: Multi-level seed refinement (not just local/global).

---

*This work demonstrates that inference-time compute scaling is a viable alternative to model scaling or post-training. By treating generation as optimization, we unlock monotonic quality improvements at runtime.*