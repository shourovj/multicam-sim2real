# Cosmos on the Swarm Servers: Hardware Check and Smoke Test (Nov 9, 2026)

**Owner:** SP Sharan · **Status:** results of one smoke-test session. It answers two open items in the [lit-review synthesis](lit-review/2026-11-09-1750-lit-review-synthesis.md) (§4, §8): which swarm GPUs can run Cosmos, and whether Cosmos3-Nano-Sim-Transfer runs on an A100. It also checks the Cosmos3-Nano choice in a coworker's v0 walkthrough, which runs text-to-video, image-to-video and depth transfer from one checkpoint.

## 1. Bottom line

- **Both Cosmos 3 models run on the swarm5 A100s.** Each produced a depth-to-video clip from NVIDIA's own sample inputs, and the output follows the depth map.
- **Cosmos3-Nano-Sim-Transfer is the practical Cosmos option this week.** It takes 2 minutes per clip on one shared A100, despite a model card that lists only Hopper and Blackwell.
- **The base Cosmos3-Nano model works but is slow here.** Transfer mode cannot use CPU offload, so it needs about 64 GiB on one GPU or a split across two. It took 31 minutes per 720p clip while sharing GPUs with other users.
- **swarm1–4 cannot run any Cosmos model.** Their RTX A5000s have 24 GB, and swarm1 and swarm2 also have glibc 2.31.
- **swarmllm (4× RTX 6000 Ada, 48 GB, idle) should run base Nano with offload, but is untested.** It also has RT cores, which matters for Isaac Sim (§6).

## 2. Servers

Checked on Nov 9, 2026 around 21:00 CST.

| Server | GPUs | Driver / CUDA | glibc | Notes |
|---|---|---|---|---|
| **swarm5** | 8× A100 80 GB PCIe (Ampere, sm 8.0) | 580.159 / 13.0 | 2.35 | Shared with three other users. Most cards had only 17–45 GB free. No NVLink. The root disk is 97% full (about 500 GB left) |
| swarm1 | 7× RTX A5000 24 GB | 535 / 12.2 | 2.31 | |
| swarm2 | 8× RTX A5000 24 GB | 580 / 13.0 | 2.31 | The disk is full (4 GB left) |
| swarm3, swarm4 | 8× RTX A5000 24 GB each | 580 / 595 | 2.35 | |
| **swarmllm** | 4× RTX 6000 Ada 48 GB (sm 8.9) | 595.99 / 13.2 | 2.35 | Idle. Has RT cores and supports FP8 |

## 3. Which Cosmos model fits where

Requirements come from NVIDIA's model cards and framework docs. Only the two Cosmos 3 models on swarm5 were tested (§4); the other cells are inferred from the documented memory floors.

| Model | swarm5 (A100 80 GB) | swarmllm (Ada 48 GB) | swarm1–4 (A5000 24 GB) |
|---|---|---|---|
| **Cosmos3-Nano** (16B) | **Tested, works.** Transfer needs about 64 GiB, so it used two GPUs | Probably works with offload for text-to-video and image-to-video. Transfer can't offload (§5), so it would need two GPUs | No |
| **Cosmos3-Nano-Sim-Transfer** (16B) | **Tested, works** on one GPU | Untested. Ada isn't listed | No |
| Cosmos-Transfer2.5-2B | Should fit (65.4 GB on one GPU), but only on an empty card | No, 48 GB is too small | No |
| Cosmos-Predict2.5-2B | Should fit (about 33 GB) | Should fit | No |
| Cosmos-Transfer1-7B | Should fit | Probably fits (28 GB minimum with offload) | No |

**Documented requirements:**
- Cosmos 3 needs CUDA ≥12.8, glibc ≥2.35 and an Ampere or newer GPU, and runs in BF16.
- Cosmos-Transfer2.5 needs driver ≥570.124 and glibc ≥2.35.

**Sources:**
- [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) and [Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer) model cards
- Framework [setup](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md) and the [Cosmos3 support matrix](https://github.com/NVIDIA/cosmos-framework/blob/main/.agents/skills/cookbooks/cosmos3/nim/support-matrix.md)
- [Cosmos-Transfer2.5 setup](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md)

## 4. Smoke test results

Each model ran one depth-transfer clip from NVIDIA's sample assets, one seed, on swarm5. The other users' jobs were running on the same GPUs.

| | Cosmos3-Nano-Sim-Transfer | Cosmos3-Nano (base) |
|---|---|---|
| Input | HF `example2` depth video + saved prompt (a person dancing outdoors) | Cookbook depth sample + prompt (a city street) |
| GPUs | 1 A100 (GPU 5) | 2 A100s (GPUs 0 and 3), FSDP `--dp-shard-size 2` |
| Our GPU memory | About 35 GB | About 42 GB on each card |
| Output | 121 frames, 832×480, 30 fps | 121 frames, 1280×720, 30 fps |
| Sampling | 4 distilled steps, guidance 1 | 50 UniPC steps, guidance 3 |
| Wall time | **119 s** | **31 min** (29 min sampling). Steps took 35–145 s depending on the other users' load |
| Visual check (frame 60) | Pose and layout match the depth map | Trees, cars and street layout match the depth map |

These are NVIDIA's sample scenes, not ours, so the runs show that the models work on this hardware, not how well they handle drones or overhead convoys. The timings are an upper bound on shared, busy GPUs.

## 5. Problems found and fixes

1. **The repo pins uv to exactly 0.11.29.** Run `uvx uv@0.11.29 sync --all-extras --group=cu130-train` rather than downgrading the global uv.
2. **Layerwise CPU offload doesn't support transfer.** With `--enable-layerwise-offload`, transfer fails with "Text K/V prefill requires single-GPU two-way text-KV eligible inference". The 31–35 GiB offload floors in NVIDIA's tables therefore don't apply to transfer.
3. **Multi-GPU jobs on swarm5 hang without `NCCL_P2P_DISABLE=1`.** The A100s have no NVLink and connect only over PCIe through the CPU. The first two-GPU run sat at 100% CPU for 15 minutes without loading weights. With P2P disabled it ran normally.
4. **The guardrail model is gated.** Our Hugging Face account isn't approved for `nvidia/Cosmos-Guardrail1`, so both runs used `--no-guardrails`, as NVIDIA's Sim-Transfer guide does.

## 6. What this changes

- **Condition D:** Cosmos3-Nano-Sim-Transfer is a better fit than Cosmos-Transfer2.5. It runs on one shared A100, it's trained on simulator renders (Transfer2.5 is reported to degrade on them), and its licence has no guardrail clause.
- **Licence:** Cosmos 3 uses [OpenMDW 1.1](https://openmdw.ai/license/1-1/), which has no guardrail clause. The NVIDIA Open Model License still covers Transfer2.5, Predict2.5 and Transfer1, and ends if guardrails are disabled. So the synthesis's guardrail concern (§5) applies to those models, not to Cosmos 3. This is a reading of the licences, not legal advice; the compliance check is still open.
- **The coworker's plan:** one base Nano checkpoint for text-only, first-frame and depth arms works on A100s. At about 30 minutes per 720p clip under current load, though, a matrix of many clips needs an empty A100, swarmllm, or lower resolution and fewer steps.
- **Isaac Sim:** it was ruled out because the A100s lack RT cores. swarmllm's RTX 6000 Ada cards have them, so it may be back on the table there. Untested.

## 7. Still open

- Sim-Transfer and Nano on our own BlenderProc drone and convoy depth, including how a small, distant drone shows up against the sky in depth.
- swarmllm: base Nano with offload, Sim-Transfer on Ada, and an Isaac Sim pre-flight.
- Access approval for Cosmos-Guardrail1, needed for Cosmos-Transfer2.5 or for any run that keeps guardrails on.
- Speed of base Nano on an empty A100 at 480p.

## 8. Reproducing

Everything is outside this repo, in `~/code/sim2real-videogen/` on swarm5:

| Path | Contents |
|---|---|
| `cosmos-framework/` | [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) at `5950e7b`, with its environment installed |
| `cosmos-logs/run-nano-depth.sh` | Base Nano run. Uses `GPU` and `NPROC` env vars; set `NCCL_P2P_DISABLE=1` for two GPUs |
| `cosmos-logs/run-simtransfer-depth.sh` | Sim-Transfer run, following NVIDIA's inference guide. Uses the `GPU` env var |
| `cosmos-outputs/` | The two generated videos and their control inputs |

The checkpoints (about 65 GB) are in the Hugging Face cache, `~/.cache/huggingface/hub`.
