# Survey of Modern World Models

## TL;DR
World models have evolved from passive video generators into interactive, closed-loop systems designed for embodied control and physical reasoning. Modern research emphasizes multimodal architectures that bridge pixel-space simulation with continuous action spaces, validated by task-centric, interactive benchmarks.

## Background
World models serve as the mental engines for autonomous agents, allowing them to simulate physical environments and predict the consequences of their actions before execution. Historically, these systems focused on visual consistency. Today, they are transitioning toward Generative Physical Artificial Intelligence (GPAI), which integrates foundational models with robotic embodiments to enable autonomous perception, reasoning, and action [1].

## Foundational Architectures
Modern world models are moving toward unified architectures that treat temporal perception, language, and action as a single, shared substrate [2]. Key developments include Robot Foundation Models (RFMs) that leverage multimodal tokenization to achieve cross-domain skill transfer [1]. Models like InternW0 represent this shift, utilizing asymmetric architectures with flow matching to jointly learn future visual dynamics and continuous robot control in asynchronous, multi-frequency environments [3]. Furthermore, researchers are re-evaluating capability emergence, focusing on the fundamental roles of predictive modeling and embodied coupling rather than just architectural scale [4].

## Video Generation and Embodied Control
Video simulation is no longer strictly open-loop; recent approaches integrate closed-loop control to allow models to adapt to real-time execution outcomes [5]. Advances in 4D Gaussian Splatting (4DGS) have improved temporal consistency, with frameworks like OuroWorld implementing deformation fields for seamless, infinite-duration scene generation [6]. For robotics, architectures such as Worldscape-MoE use Mixture-of-Experts (MoE) modules to unify heterogeneous control modalities, enabling the model to manage diverse physical inputs within a single Transformer-based backbone [7]. Similarly, models like SPW-Nav utilize panoramic video generation to interpret movement instructions in real-time, bridging the gap between high-level navigation goals and pixel-space streaming [8].

## Evaluation and Benchmarking
The evaluation of world models has shifted away from static metrics like PSNR or LPIPS, favoring interaction-centric, closed-loop frameworks. Benchmarks such as WBench and Omni-WorldBench provide multi-turn, interactive test cases to assess an agent’s performance across long-horizon scenarios [9][10]. Specialized diagnostic tools, including PhyGround, have introduced taxonomy-driven evaluations of physical reasoning, often utilizing vision-language models to judge agent-object interactions [11]. Recent work underscores that for robotics applications, long-horizon consistency and controllability are primary determinants of success, with policy-based evaluations like GigaWorld-1 and World-in-World prioritizing end-to-end task completion metrics [12][13].

## Trends and Open Problems
The field is currently moving toward "dual-loop" architectures that balance fast sensory perception with slower, high-level reasoning [3]. While significant progress has been made in procedural generation, a primary challenge remains the robust integration of asynchronous control signals with complex physical simulations [3]. Future efforts are expected to refine long-horizon temporal consistency and improve the data-efficiency of training world models across diverse embodiments [14].

## References
[1] A Comprehensive Review of Generative Physical Artificial Intelligence. arxiv. https://arxiv.org/abs/2609.18111 (2026-09-16)
[2] Generalist Open-World Temporal Perception. arxiv. https://arxiv.org/abs/2609.06823 (2026-09-06)
[3] InternW0: A Foundational Physical World Model for Efficient Real-World Interactions. arxiv. https://arxiv.org/abs/2609.27656 (2026-09-23)
[4] Seven Sources of Physical AI Capability Formation. arxiv. https://arxiv.org/abs/2609.09627 (2026-09-09)
[5] WorldGuide. hf-daily. https://huggingface.co/papers/2610.12459 (2026-10-08)
[6] OuroWorld. hf-daily. https://huggingface.co/papers/2610.12461 (2026-10-08)
[7] Worldscape-MoE. hf-search. https://huggingface.co/papers/2607.03964 (2026-07-04)
[8] SPW-Nav. hf-daily. https://huggingface.co/papers/2610.08941 (2026-10-06)
[9] Omni-WorldBench. hf-search. https://huggingface.co/papers/2603.22212 (2026-03-23)
[10] WBench. hf-search. https://huggingface.co/papers/2605.25874 (2026-05-25)
[11] PhyGround. hf-search. https://huggingface.co/papers/2605.10806 (2026-05-11)
[12] GigaWorld-1. hf-search. https://huggingface.co/papers/2607.02642 (2026-07-02)
[13] World-in-World. hf-search. https://huggingface.co/papers/2510.18135 (2025-10-20)
[14] LongVie 2. hf-search. https://huggingface.co/papers/2512.13604 (2025-12-15)
