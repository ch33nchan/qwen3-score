# Inference Scaling in Diffusion: An Evolutionary Approach to Character Inpainting

We often talk about scaling laws in training—throw more data and compute at a model, and loss goes down. But what about **inference scaling**? What if we could trade compute for quality *at runtime*?

This is the core philosophy behind **EACPS (Evolutionary Annealing with Candidate Potential Scoring)**. In this writeup, I'll explain how I'm treating image generation not as a one-shot probabilistic roll, but as a search problem over the latent trajectory space.

## The Problem: Stochasticity vs. Control

Standard diffusion models model the conditional distribution $p_\theta(x_0 \mid c)$, where $c$ is our condition (mask, prompt, reference image).

$$
x_0 = \mathcal{F}_\theta(z_T, c)
$$

Where $z_T \sim \mathcal{N}(0, I)$ is the initial Gaussian noise.

The problem is variance. For character consistency, the "mode" of this distribution is often broad. One seed yields a perfect likeness; another yields a distorted face. We don't want the *average* sample; we want the *maximum a posteriori* (MAP) estimate, or at least a sample from the high-density region of a quality-weighted posterior.

## The Algorithm: EACPS

I treat the generation process as an optimization problem where we search for the optimal noise seed $s^* \in \mathcal{S}$ that maximizes a Potential Function $U(x)$.

$$
s^* = \mathop{\mathrm{argmax}}_{s \in \mathcal{S}} U(\mathcal{F}_\theta(\mathrm{seed}(s), c))
$$

Since $\mathcal{F}_\theta$ (the diffusion process) is non-convex and expensive to differentiate through for discrete metrics (like "does this look like the character?"), I use a derivative-free evolutionary strategy.

### Stage 0: The Identity Prior (The "Warm Start")

Diffusion models struggle to "hallucinate" exact identity details (tattoos, specific scar shapes) from a text prompt alone. To fix this, I inject a strong prior before diffusion begins.

I use **InsightFace** to perform a rigid identity transfer in pixel space:

$$
x_{\mathrm{prior}} = \mathrm{FaceSwap}(I_{\mathrm{init}}, I_{\mathrm{char}})
$$

This $x_{\mathrm{prior}}$ serves as the conditioning signal. Instead of asking the diffusion model to *create* the identity, we ask it to *refine and harmonize* it. This collapses the search space significantly, ensuring we start in the correct "basin of attraction."

### Stage 1: Global Exploration (Monte Carlo Search)

We define a population of $K_{\mathrm{global}}$ candidates. Each candidate is a trajectory through the reverse diffusion process, determined by a random seed $s_i$.

$$
\mathcal{P}_{\mathrm{global}} = \{ x_i \mid x_i = \mathrm{QwenEdit}(x_{\mathrm{prior}}, \mathrm{prompt}, s_i) \}_{i=1}^{K_{\mathrm{global}}}
$$

Here, `QwenEdit` is our diffusion backbone. It takes the face-swapped image and "heals" the artifacts—blending skin tones, fixing lighting, and ensuring the expression matches the context.

#### The Potential Function $U(x)$

How do we evaluate $x_i$? I use a **Multi-Model Scorer** (VLM-based). We define a vector of utility functions $\mathbf{v}(x) = [v_{\mathrm{id}}, v_{\mathrm{realism}}, v_{\mathrm{prompt}}]$.

The potential $U(x)$ is a scalarization of these objectives:

$$
U(x) = \sum_{j} w_j \cdot v_j(x)
$$

In my implementation, $v_{\mathrm{id}}$ checks facial similarity, and $v_{\mathrm{realism}}$ (via Moondream/Gemini) checks for artifacts.

### Stage 2: Local Refinement (Evolutionary Annealing)

Once we evaluate the global population, we select the top $M$ candidates based on $U(x)$. These represent promising regions in the latent seed space.

We then perform "local annealing." We assume that the quality landscape of diffusion models is locally correlated with respect to the seed (i.e., seed 1000 is likely to be somewhat similar to seed 1001 in terms of structural composition, though high-frequency details differ).

For each best candidate $s_{\mathrm{best}}$, we spawn $K_{\mathrm{local}}$ children:

$$
\mathcal{P}_{\mathrm{local}} = \{ x'_j \mid x'_j = \mathrm{QwenEdit}(x_{\mathrm{prior}}, \mathrm{prompt}, s_{\mathrm{best}} + \delta_j) \}
$$

This allows us to fine-tune the high-frequency details—the glint in the eye, the texture of the skin—without losing the structural identity we found in Stage 1.

## Inference Scaling Laws

This approach demonstrates **test-time compute scaling**.

Let $Q(N)$ be the expected quality of the result given $N$ inference passes.

$$
Q(N) \propto \log(N)
$$

By increasing $K_{\mathrm{global}}$ and $K_{\mathrm{local}}$, we monotonically increase the probability of sampling the tail-end of the quality distribution. We are effectively trading GPU hours for a reduction in failure rate.

## Summary

1.  **Inject Prior:** Use InsightFace to anchor the identity.
2.  **Explore:** Sample the latent landscape broadly (Global Stage).
3.  **Evaluate:** Use VLMs to approximate the "human eye" potential function $U(x)$.
4.  **Exploit:** Anneal around the best solutions (Local Stage).

This is how we turn a stochastic generator into a reliable production pipeline.