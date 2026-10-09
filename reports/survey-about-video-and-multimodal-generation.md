# Survey of Modern Video and Multimodal Generation Architectures

## TL;DR
Recent advancements in multimodal video generation are defined by a shift toward Diffusion Transformer (DiT) architectures, the integration of Multimodal Large Language Models (MLLMs) as orchestrating directors, and the transition from basic per-frame metrics to agentic, multi-dimensional diagnostic benchmarks.

## Background
Video generation has evolved from early frame-level models to sophisticated, high-fidelity synthesis systems. Key challenges include maintaining long-term temporal coherence, ensuring physical plausibility, and achieving cross-modal synchronization. The current landscape is dominated by DiT models, which offer superior scalability, and agentic workflows that leverage LLMs to manage complex, multi-shot narrative structures.

## Architectural Foundations in Video Synthesis
The transition from traditional UNet-based architectures to Diffusion Transformers (DiTs) has fundamentally improved high-fidelity video synthesis. DiTs provide more efficient scaling and better handling of complex dynamics. Innovations like serial-to-parallel diffusion (S2PD) optimize the trade-off between logical consistency and parallel computational speed [1]. Furthermore, addressing the performance degradation of sparse attention in large DiTs has led to the development of techniques like Meta-Cached Sparse Attention [2]. Temporal coherence, a core challenge in video generation, is improved through strategies like video token merging (VidToMe), which reduces memory footprints while ensuring cross-frame stability [3]. To solve latency issues in real-time applications such as talking-head generation, new distillation schemes enable high-quality output in single-step processes [4].

## MLLMs as Generative Directors
Multimodal LLMs are increasingly serving as central "directors" within generation pipelines. By interpreting high-level user instructions, they guide specialized generative modules, bridging the gap between abstract intent and concrete visual output. Projects like JavisGPT utilize an encoder-LLM-decoder architecture to handle joint audio-visual comprehension and generation, relying on fusion modules for spatio-temporal alignment [5]. For procedural tasks, models like WorldGuide operate as closed-loop systems, where the MLLM acts as an agent that decides, executes, and verifies task progression, enabling long-term goal-directed synthesis [6]. Similarly, approaches like the LLM Director decompose complex concepts into 3D representations, which are subsequently refined by diffusion models to ensure aesthetic and physical fidelity [7]. Broader "Any-to-Many" architectures have further integrated base LLMs to coordinate diverse modalities through specialized instruction templates [8].

## Diagnostic Evaluation and Benchmarking
The evaluation of video generation has moved beyond simplistic pixel-level metrics like FID and FVD. Modern frameworks are now model-centered and agentic, providing granular diagnostics. VBench evaluates T2V models across 16 dimensions [9]. For multimodal tasks, T2AV-Compass addresses the lack of standardized metrics in text-to-audio-video generation by using unified prompt sets to evaluate synchronization and instruction adherence [10]. Long-form video synthesis is particularly challenging, and DirectorBench serves as a diagnostic tool for minute-long, multi-shot narratives, identifying failures in coherence that standard metrics miss [11]. The industry is shifting toward these holistic, multi-dimensional diagnostic frameworks, which leverage MLLMs and agentic systems to assess physical plausibility and cross-modal synchronization in a perceptually grounded manner [12].

## Trends and Open Problems
The field is moving toward long-form, goal-directed generation where temporal consistency and physical grounding are critical. While DiTs and MLLM-directed pipelines have shown impressive results, significant challenges remain. Open problems include improving the physical reliability of long-duration content, reducing the high computational cost of inference, and developing benchmarks that can reliably measure subjective qualities like narrative coherence and emotional resonance without relying on potentially biased model-based judges.

## References
[1] S2PD: Serial-to-Parallel Diffusion. arxiv. https://arxiv.org/abs/2610.06847 (2026-10-05)
[2] MC-Sparse: Deconstructing and Closing the Dense-Sparse Attention Gap. hf-daily. https://huggingface.co/papers/2610.06801 (2026-10-05)
[3] VidToMe: Video Token Merging. hf-search. https://huggingface.co/papers/2312.10656 (2023-12-17)
[4] LeapTalk: Breaking the Latency-Quality Trade-off. arxiv. https://arxiv.org/abs/2608.00079 (2026-07-29)
[5] JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation. hf-search. https://huggingface.co/papers/2512.22905 (2025-12-28)
[6] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. hf-daily. https://huggingface.co/papers/2610.12459 (2026-10-08)
[7] Compositional 3D-aware Video Generation with LLM Director. hf-search. https://huggingface.co/papers/2409.00558 (2024-08-31)
[8] Spider: Any-to-Many Multimodal LLM. hf-search. https://huggingface.co/papers/2411.09439 (2024-11-14)
[9] VBench: Comprehensive Benchmark for T2V. hf-search. https://huggingface.co/papers/2604.08540 (2026-04-09)
[10] T2AV-Compass: Unified Evaluation for Text-to-Audio-Video. arxiv. https://arxiv.org/abs/2512.21094 (2025-12-24)
[11] DirectorBench: Diagnosing Long-Form Video Generation. arxiv. https://arxiv.org/abs/2605.30090 (2026-05-28)
[12] Generative AI Video Evaluation Survey. web. https://openaccess.thecvf.com/content/CVPR2026W/VGBE/papers/Safavigerdini_Generative_AI_Video_Evaluation_Survey_of_Metrics_Benchmarks_and_Trustworthiness_CVPRW_2026_paper.pdf (2026-06-01)
