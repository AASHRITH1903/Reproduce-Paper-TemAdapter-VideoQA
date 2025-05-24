# Reproducing the results of paper Tem-adapter (Video Question-Answering)

This repository explores the reproduction and improvement of the [Tem-adapter](https://arxiv.org/pdf/2308.08414) architecture for Video Question Answering (VideoQA) using the [SUTD-TrafficQA dataset](https://github.com/SUTDCV/SUTD-TrafficQA). The project involves replication of results using released checkpoints, training from scratch, and extending the architecture with a custom cross-attention layer.

---

## Setup

- Dataset: Download the [SUTD-TrafficQA](https://github.com/SUTDCV/SUTD-TrafficQA) dataset and place it in the ```data/``` folder
- Released Checkpoint: [Drive Link](https://drive.google.com/drive/folders/1SplEKEjrp-Uw-PxziyBHvUuU-yQ0YevX)
- Reproduced Checkpoint: [Drive Link](https://drive.google.com/drive/folders/1HLZ5SMFfdEljQsT8jPoytoIDXDmczdwg?usp=sharing)

---

## Results

### Replication with Released Checkpoint

| Source             | Validation Accuracy |
|--------------------|---------------------|
| Original (paper)   | 46.00%              |
| Reproduced (ckpt)  | 46.00%              |

✔️ Exact match with the published results using the official checkpoint.

---

### Training from Scratch

| Metric             | Value               |
|--------------------|---------------------|
| Sum loss           | 0.127               |
| Avg loss           | 0.34                |
| CE loss            | 33.28               |
| Recon loss         | 0.0067              |
| Average Accuracy   | 98.20%              |
| Validation Accuracy| 45.37%              |

⚠️ Minor drop (~0.63%) from original likely due to smaller batch size and different GPU.

<!-- ---

### 🔬 Extended Architecture: Cross-Attention Summarizer

Modifications:
```python
old_video_embed = mean(frame_embeds)

new_video_embed = mean(frame_embeds) + cross_attn(
    query=query_embed,
    keys=frame_embeds,
    values=frame_embeds
) -->
