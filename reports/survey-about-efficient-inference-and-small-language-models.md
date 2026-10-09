# Survey on Efficient Inference for Small Language Models

## TL;DR
Efficient inference for Small Language Models (SLMs) focuses on reducing computational and memory bottlenecks through architectural innovations, hardware-aware compression, and system-level co-design. Emerging techniques like batch-aware expert routing, recurrent transformer blocks, and asynchronous speculative decoding allow sub-7B models to deliver real-time performance on resource-constrained edge hardware.

## Background
The demand for local, on-device AI has spurred the development of Small Language Models (SLMs). Unlike frontier models, SLMs aim to optimize the trade-off between parameter count, latency, and memory footprint. Challenges in deploying these models include limited VRAM capacity, thermal constraints of mobile hardware, and the need for high-performance inference in real-time, embodied intelligence scenarios.

## Architectural Innovations
Recent architectural advancements prioritize parameter efficiency. Mixture-of-Experts (MoE) models reduce computation by activating only a subset of experts per token; however, standard implementations can suffer from memory-transfer bottlenecks. Batch-aware expert selection methods optimize which experts are loaded across concurrent requests [1]. Recurrent transformer blocks offer another pathway to efficiency by applying a single block repeatedly, achieving model depth without increasing the parameter count [2]. Furthermore, specialized architectures like those used in sub-billion parameter models have demonstrated that compact designs can outperform general-purpose models in specific domains [3]. To mitigate memory constraints, techniques that coordinate inference across tiered storage (SSD, RAM, and VRAM) allow large-capacity models to operate effectively on hardware with limited onboard memory [4].

## Model Compression for Edge Deployment
Compression remains essential for edge deployment, with quantization and pruning serving as core techniques. Standard quantization methods, such as INT4, are widely used for reducing memory footprint, while outlier-aware dual-precision quantization has shown significant improvements in balancing accuracy and speed [5]. Structured pruning techniques, such as OTOV3, provide substantial reductions in computational complexity (MACs) and parameter counts [6]. Knowledge distillation is frequently leveraged alongside these methods to ensure that smaller, compressed models retain the performance of their larger counterparts [7][8]. Integration of these techniques, often referred to as layer-wise unified compression, has been shown to deliver multiple-fold speedups and memory reductions on edge devices [5].

## Hardware-Aware Inference and Energy Efficiency
Hardware-aware optimization is critical for sustained edge inference. Speculative decoding has been adapted through asynchronous, task-level architectures, such as AHASD, which mitigates the inefficiencies associated with fluctuating draft lengths on mobile NPU-PIM systems [9]. Energy efficiency is a primary concern; dynamic voltage and frequency scaling (DVFS) tailored for SLMs has been introduced to improve device longevity and thermal management [10]. Sustained inference poses significant challenges, including the thermal instability of mobile GPUs [11]. To address this, hybrid attention architectures and shared-KV cache strategies are being developed to minimize redundant memory access and computation during decoding, particularly for real-time applications [12].

## Trends and Open Problems
The field is shifting toward holistic co-design, where model architecture, hardware selection, and system management (such as thermal-aware tier routing) are treated as a single optimization problem. Despite progress, several open problems persist. Thermal management remains a major barrier, as sustained generation can lead to runtime instability and hardware degradation on mobile devices [11]. Additionally, there is a need to refine the integration of heterogeneous hardware, particularly in bridging the latency gaps between diverse memory tiers, to support more reliable, high-performance edge deployment.

Speculative decoding has been adapted through asynchronous, task-level architectures, such as AHASD [9], which mitigates the inefficiencies associated with fluctuating draft lengths on mobile NPU-PIM systems [9].

## References
[1] BASE: Batch-Aware Selection of Experts. arxiv. https://arxiv.org/abs/2609.36222 (2026-09-28)
[2] One Block, Multiple Depths. hf-daily. https://huggingface.co/papers/2610.12448 (2026-10-08)
[3] FLOORA: Domain-Specific Language Model. arxiv. https://arxiv.org/abs/2609.36064 (2026-09-28)
[4] SSD-LLaMA. arxiv. https://arxiv.org/abs/2609.18110 (2026-09-16)
[5] EDGE-LLM: Enabling Efficient Large Language Model Adaptation on Edge Devices. web. https://dl.acm.org/doi/10.1145/3649329.3658473 (2024-11-07)
[6] Edge AI: Evaluation of Model Compression Techniques for .... web. https://arxiv.org/html/2409.02134v1 (2024-09-02)
[7] A Survey on Model Compression for Large Language Models. hf-search. https://huggingface.co/papers/2308.07633 (2023-08-15)
[8] Compact Language Models via Pruning and Knowledge Distillation. hf-search. https://huggingface.co/papers/2407.14679 (2024-07-19)
[9] AHASD: Asynchronous Heterogeneous Architecture for LLM Adaptive Drafting Speculative Decoding on Mobile Devices. arxiv. https://arxiv.org/abs/2604.25326 (2026-04-28)
[10] DVFS for Small Language Model Inference on Mobile Edge Devices. arxiv. https://arxiv.org/abs/2609.13153 (2026-07-08)
[11] HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference. arxiv. https://arxiv.org/abs/2609.30270 (2026-07-14)
[12] IronLLM: Forging Compact Edge-Native Language Models for Real-Time Embodied Intelligence. hf-search. https://huggingface.co/papers/2609.36860 (2026-09-29)
