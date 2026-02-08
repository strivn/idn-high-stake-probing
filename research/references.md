# Research References

## Core Papers

### Activation Oracles (Anthropic, 2025)
**URL:** https://alignment.anthropic.com/2025/activation-oracles/

**Key Innovation:** LLMs that accept neural activations as inputs and answer natural-language questions about them.

**Relevance:** Primary inspiration for project. Demonstrates generalist interpretation beyond task-specific probes.

**Key Takeaways:**
- Single model can answer multiple questions vs. one probe per property
- Generalizes to novel out-of-distribution questions
- Natural language interface for activation interpretation
- Successfully uncovers secret knowledge and misalignment from fine-tuning
- Limitations: expensive inference, potential confabulation, not mechanistic

---

### Persona Features Control Emergent Misalignment
**URL:** https://arxiv.org/abs/2506.19823

**Main Finding:** Misalignment operates through identifiable "persona features" in activation space, particularly a "toxic persona feature."

**Methodology:** Sparse autoencoders and model diffing to compare representations before/after fine-tuning.

**Practical Impact:** Shows that fine-tuning on just a few hundred benign samples can restore alignment.

**Relevance:** Demonstrates that interpretability can lead to practical control mechanisms.

---

### Analysing Moral Bias in Finetuned LLMs through Mechanistic Interpretability
**URL:** https://arxiv.org/abs/2510.12229v2

**Main Finding:** Moral biases (Knobe effect) can be localized to specific layers and surgically removed.

**Methodology:** Layer-patching analysis across three open-weights LLMs.

**Practical Impact:** Targeted interventions without full model retraining.

**Relevance:** Shows mechanistic interpretability enables precise bias mitigation.

---

### Persona Vectors: Monitoring and Controlling Character Traits in Language Models
**URL:** https://arxiv.org/abs/2507.21509

**Main Finding:** Personality/behavioral traits can be represented as directional patterns in activation space.

**Methodology:** Extract directional embeddings using natural-language descriptions.

**Practical Impact:**
- Predict personality shifts during fine-tuning
- Post-hoc intervention to steer behavior
- Identify problematic training data

**Relevance:** Practical tool for monitoring and controlling behavioral dimensions.

---

## Additional Resources

### Interpretability Perspectives

**A Pragmatic Vision for Interpretability**
- URL: https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability
- Focus: Practical, application-oriented approach to interpretability

**An Ambitious Vision for Interpretability**
- URL: https://www.lesswrong.com/posts/Hy6PX43HGgmfiTaKu/an-ambitious-vision-for-interpretability
- Focus: Comprehensive mechanistic understanding

### Tools & Datasets

**Gemma Scope 2**
- URL: https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/
- Resource: SAEs for understanding language model behavior

**Building and Training an LLM from Scratch**
- URL: https://beyondtheparrot.com/what-i-learned-building-and-training-an-llm-from-scratch-you-can-too/
- Resource: Practical guide to LLM fundamentals

---

---

## New Critical Papers (January 2025)

### Building Production-Ready Probes For Gemini
**URL:** https://arxiv.org/abs/2601.11516

**Main Innovation:** Developed robust activation probes that work in production despite distribution shifts (especially variable context lengths).

**Key Technical Contributions:**
- **Multimax architecture:** Handles context length distribution shifts
- Systematic robustness testing against jailbreaks, multi-turn conversations, adaptive red teaming
- Hybrid probe+classifier systems for efficiency
- Successfully deployed in Gemini production for misuse detection

**Relevance:** This is EXACTLY what we need - shows how to make interpretability tools production-ready and robust.

---

### The Assistant Axis: Situating and Stabilizing the Default Persona
**URL:** https://arxiv.org/abs/2601.10387

**Main Finding:** Language models have an identifiable "Assistant Axis" in activation space that controls helpful/harmless behavior.

**Key Insights:**
- Steering toward Assistant direction → helpful, stable behavior
- Steering away → mystical/theatrical responses, persona drift
- Persona instability correlates with meta-reflective conversations and emotional vulnerability
- Can use activation steering to prevent both natural drift AND adversarial jailbreaks

**Relevance:** Shows mechanistic approach to controlling model stability and preventing problematic outputs.

---

### Recursive Language Models (For Interest)
**URL:** https://arxiv.org/abs/2512.24601

**Main Innovation:** Process prompts 100x longer than context window by recursive decomposition.

**Why Interesting:** Enables handling very long contexts (e.g., entire codebases, long documents) with comparable or reduced compute cost.

---

## Key Tools & Frameworks

### Inspect AI
**Source:** UK AI Security Institute
**URL:** https://inspect.aisi.org.uk/

**What it is:** Open-source framework for LLM evaluations, adopted by Anthropic, DeepMind, Grok.

**Key Features:**
- 100+ pre-built evaluations
- Supports all major model providers (OpenAI, Anthropic, Google, local models)
- Built-in tools: bash, Python, web search, sandboxing
- Web-based visualization dashboard
- Specifically designed for AI safety evaluation

**Use cases:**
- Jailbreak testing
- Policy compliance checking
- Agentic task evaluation
- Multi-step reasoning tests

**Why relevant:** Production-grade evaluation framework for our experiments.

---

### TransformerLens
**Maintainer:** Neel Nanda (now Bryce Meyer)
**URL:** https://github.com/TransformerLensOrg/TransformerLens

**Capabilities:**
- Access model activations at each layer/position
- Hook-based activation manipulation and patching
- Circuit analysis for understanding computation flow
- Activation caching for efficiency

**Use cases:**
- Extract activations for probe training
- Patch activations to test interventions
- Trace information flow through models

---

### SAELens
**Maintainers:** Joseph Bloom, David Chanin
**URL:** https://github.com/decoderesearch/SAELens

**What it does:**
- Train Sparse Autoencoders (SAEs) on model activations
- Decompose computations into interpretable features
- Provides pre-trained SAEs for various models
- Generates feature dashboards (SAE-Vis)

**How it works with TransformerLens:**
- `HookedSAETransformer` attaches SAEs to TransformerLens models
- Can cache SAE feature activations
- Unified workflow: TransformerLens loads models, SAELens finds interpretable features

---

## To Explore

- [x] TransformerLens library for activation extraction
- [x] SAELens for sparse autoencoder work
- [x] Inspect AI for evaluation framework
- [ ] Gemini multimax probe architecture details
- [ ] Pre-trained SAEs for target models
- [ ] Inspect AI's built-in safety evaluations
- [ ] Assistant Axis replication code (if available)
- [ ] Existing hallucination detection benchmarks
