# GPU Environment Spec and Comparison

This document provides a comprehensive comparison between the local baseline GPU (Nvidia GeForce RTX 2060) and the potential target GPUs for training the DailyTalk model: **Nvidia Tesla T4**, **Nvidia L4**, and **Nvidia RTX PRO 6000 (Ada Generation)**.

---

## Hardware Comparison Matrix

The table below summarizes the key technical specifications and recommendations for each GPU:

| GPU Model | Architecture | VRAM | Memory Bandwidth | CUDA Cores | Tensor Cores | Native BF16 Support | Recommended Batch Size | Estimated Training Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RTX 2060** *(Baseline)* | Turing (TU106) | 6 GB | 336 GB/s | 1,920 | 240 | No (Fallback to FP32) | 4 - 8 | **1.0x** *(Very Slow / VRAM-bottlenecked)* |
| **Tesla T4** *(Cloud)* | Turing (TU104) | 16 GB | 320 GB/s | 2,560 | 320 | No (Fallback to FP32) | 16 | **2.5x** *(Standard Cloud Entry)* |
| **L4** *(Ada Lovelace)* | Ada Lovelace (AD104) | 24 GB | 300 GB/s | 7,424 | 232 (Gen 4) | **Yes** *(Highly Optimized)* | 32 - 48 | **6.5x** *(Highly Recommended)* |
| **RTX PRO 6000** *(Ada)* | Ada Lovelace (AD102) | 48 GB | 960 GB/s | 18,176 | 568 (Gen 4) | **Yes** *(Highly Optimized)* | 64 - 128 | **12.0x+** *(Ultimate Power)* |

---

## Detailed GPU Analysis

### 1. Nvidia GeForce RTX 2060 (Local Baseline)
*   **VRAM Limit**: At **6 GB**, this GPU is severely VRAM-constrained for Text-to-Speech (TTS) training. Since TTS models (like FastSpeech2/CompTransTTS) process multi-dimensional spectrograms, long audio sequences will trigger Out-of-Memory (OOM) errors even with small batch sizes.
*   **Batch Size Constraint**: Practically limited to a batch size of **4 or 8**. Training with such small batch sizes makes gradients noisy and can lead to unstable alignment learning (since unsupervised alignment needs stable prior-to-mel matching).
*   **Precision**: Supports FP16 (Automatic Mixed Precision), but does not support BF16.
*   **Verdict**: **Not suitable for production training.** Highly recommended only for quick syntax/script verification dry-runs.

### 2. Nvidia Tesla T4
*   **VRAM**: **16 GB** is a substantial upgrade from 6 GB, giving enough headroom to train with the default batch size of **16** comfortably.
*   **Performance Limits**: It is based on the legacy Turing architecture. While it supports FP16 mixed precision, it **lacks native BF16 (bfloat16) hardware execution**. Trying to run BF16 on a T4 will cause PyTorch to fallback to FP32 software calculation, making training extremely slow.
*   **Precision Strategy**: **Must use FP16 AMP** (`--use_amp` in the script). Loss scaling via PyTorch's `GradScaler` is required to prevent underflow.
*   **Verdict**: A decent, budget-friendly cloud option. Good for running standard training, but noticeably slower than Ada Lovelace architectures.

### 3. Nvidia L4 (Highly Recommended Value)
*   **VRAM**: **24 GB** provides substantial memory headroom. You can scale the batch size to **32** or even **48** safely.
*   **Modern Features**: Based on the Ada Lovelace architecture, the L4 features 4th-generation Tensor Cores. It includes **native support for BF16 and FP8**.
*   **Precision Strategy**: **Highly recommend BF16** over FP16. BF16 has the same dynamic range as FP32, eliminating gradient underflow/overflow issues and removing the need for a dynamic `GradScaler`.
*   **Verdict**: **The optimal value-to-performance choice** for cloud training. Running BF16 on L4 provides exceptionally fast, mathematically stable training.

### 4. Nvidia RTX PRO 6000 Ada (Ultimate Performance)
*   **VRAM**: **48 GB** of ECC-protected VRAM. This is a massive amount of memory that eliminates any fear of OOMs.
*   **Throughput**: Boasts 18,176 CUDA Cores and 960 GB/s bandwidth. It can train at batch sizes of **64 to 128** with ease.
*   **Precision Strategy**: **Native BF16** is fully supported and runs at maximum hardware throughput.
*   **Verdict**: **The absolute fastest choice.** If budget is not an issue, this GPU will complete the 900,000 steps multiple times faster than the L4 or T4, allowing for large-batch optimization.

---

## Mixed-Precision Strategies: FP16 vs. BF16

To maximize the compute power of these GPUs, understanding mixed precision is crucial.

### FP16 (Float16) Mixed Precision
*   **How it works**: Uses 16-bit floating point representation (1 sign bit, 5 exponent bits, 10 fraction bits).
*   **Pros**: Supported on all listed GPUs (RTX 2060, T4, L4, RTX PRO 6000). Speeds up execution and cuts VRAM usage in half.
*   **Cons**: Narrow dynamic range (due to only 5 exponent bits). Easily causes underflow (gradients become 0) or overflow (gradients become `NaN`).
*   **Mitigation**: Requires PyTorch's `GradScaler` to dynamically scale losses up and down to fit into the float16 range.

### BF16 (Bfloat16) Mixed Precision
*   **How it works**: Brain Floating Point 16-bit representation (1 sign bit, 8 exponent bits, 7 fraction bits). It matches the exponent range of FP32 while reducing precision.
*   **Pros**:
    *   **No underflow/overflow**: Since it shares the exact same exponent range as FP32, gradients do not explode or vanish.
    *   **No GradScaler Needed**: Simplifies the training loop, saves computation, and guarantees stability.
*   **Cons**: Only supported natively on modern hardware (**L4** and **RTX PRO 6000**).
*   **Mitigation**: None needed! It is a drop-in, highly stable format.

---

## Batch-Size Scaling and Gradient Accumulation

In deep learning, larger batch sizes improve gradient estimation stability and allow for higher learning rates. However, scaling batch sizes too far can exceed VRAM or alter optimization dynamics.

### Dynamic Recommendation Table
If you scale the batch size, you should adjust the learning rate and gradient accumulation steps to match:

| Selected GPU | Target Batch Size | Learning Rate Scaling | Gradient Accumulation (`grad_acc_step`) |
| :--- | :---: | :---: | :---: |
| **RTX 2060** | 4 | Keep original (0.001) | 4 *(Simulates batch size 16)* |
| **Tesla T4** | 16 | Original (0.001) | 1 *(Direct step)* |
| **L4** | 32 | Scale by 1.5x (0.0015) | 1 *(Direct step)* |
| **RTX PRO 6000** | 64 | Scale by 2.0x (0.002) | 1 *(Direct step)* |

> [!TIP]
> **Simulating Large Batches on L4**: If you train on an L4 and want a robust batch size of 64 but want to save VRAM, configure `batch_size: 32` and `grad_acc_step: 2` in `train.yaml`. This performs two forward passes before updating weights, simulating an effective batch size of 64 perfectly.
