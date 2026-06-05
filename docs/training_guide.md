# Training Guide and Reference

This document outlines how to configure, start, resume, and monitor training of the DailyTalk model.

---

## 1. Configurations (`config/DailyTalk/train.yaml`)

Before launching training on your selected GPU, adjust the core hyperparameters inside `config/DailyTalk/train.yaml`:

```yaml
seed: 1234
dist_config:
  dist_backend: "nccl"
  dist_url: "tcp://localhost:54321"
  world_size: 1
path:
  ckpt_path: "./output/ckpt/DailyTalk"
  log_path: "./output/log/DailyTalk"
  result_path: "./output/result/DailyTalk"
optimizer:
  batch_size: 16                  # Update this according to your GPU's VRAM (16 default for T4)
  betas: [0.9, 0.98]
  eps: 0.000000001
  weight_decay: 0.0
  grad_clip_thresh: 1.0
  grad_acc_step: 1                # Increase to simulate larger batch sizes (e.g., 2)
  warm_up_step: 4000
  anneal_steps: [300000, 400000, 500000]
  anneal_rate: 0.3
step:
  total_step: 900000              # Total training iterations
  log_step: 100                   # Iterations between printing loss logs
  synth_step: 1000                # Iterations between synthesizing evaluation audio
  val_step: 1000                  # Iterations between validation evaluations
  save_step: 25000                # Iterations between checkpoint saves
```

---

## 2. Launching Training

Training is launched using the `uv` environment runner.

### Running with FP16 Automatic Mixed Precision (AMP)
Highly recommended for **RTX 2060** and **Tesla T4**:
```bash
uv run python3 train.py --dataset DailyTalk --use_amp
```

### Running in Standard FP32 (Full Precision)
```bash
uv run python3 train.py --dataset DailyTalk
```

---

## 3. Native BF16 Optimization for Ada Lovelace GPUs (L4 / RTX PRO 6000)

If you run training on an **Nvidia L4** or **RTX PRO 6000**, using **BF16 (Bfloat16)** mixed precision is much more stable and efficient than FP16. BF16 has the same exponent size as FP32, meaning gradients do not overflow or underflow, eliminating the need for loss scaling.

To enable native BF16, you can modify `train.py` to bypass the `GradScaler` and set autocast to `bfloat16`.

### How to Modify `train.py` for BF16:

1.  **Add a `--use_bf16` argument** in the script arguments block (around line 245):
    ```python
    parser.add_argument("--use_bf16", action="store_true", help="Use native BF16 mixed precision")
    ```

2.  **Modify GradScaler initialization** (around line 83):
    ```python
    # Disable GradScaler if BF16 is active, as BF16 does not require dynamic loss scaling
    use_scaler = args.use_amp and not getattr(args, "use_bf16", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ```

3.  **Update Autocast context block** (around line 128):
    ```python
    # Set autocast to bfloat16 or float16 dynamically
    dtype = torch.bfloat16 if getattr(args, "use_bf16", False) else torch.float16
    with amp.autocast("cuda", enabled=(args.use_amp or getattr(args, "use_bf16", False)), dtype=dtype):
    ```

### Launch Command with BF16:
```bash
uv run python3 train.py --dataset DailyTalk --use_bf16
```

---

## 4. Resuming Training from Checkpoint

If training gets interrupted or you want to resume from a saved checkpoint, use the `--restore_step` argument:

```bash
uv run python3 train.py --dataset DailyTalk --use_amp --restore_step <STEP_NUMBER>
```

For example, to resume from step 150,000:
```bash
uv run python3 train.py --dataset DailyTalk --use_amp --restore_step 150000
```

> [!NOTE]
> Ensure the checkpoint file `150000.pth.tar` is present in the checkpoint output directory (`output/ckpt/DailyTalk/`).

---

## 5. Monitoring Training with TensorBoard

You can monitor training loss curves, alignments, and synthesized speech quality in real-time.

### Launching TensorBoard
Run this command from your terminal:
```bash
uv run tensorboard --logdir output/log
```

Then open your browser and navigate to `http://localhost:6006`.

### Key Metrics to Monitor:
1.  **Total Loss**: Should show a steady downward trend.
2.  **Mel Loss & PostNet Loss**: Measure how well the model is learning to reconstruct the spectrograms.
3.  **Binarization Loss**: In unsupervised duration modeling, this measures the transition from soft alignments to hard binarized durations (should drop to 0 after around 18,000 steps).
4.  **Synthesized Audios**: Play reconstructed vs. prediction audios under the `Audio` tab to hear quality improvements as steps progress.
