## Training

### Stage 1: Pretraining

First, download the pretraining dataset for LLaVA-NeXT from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

To initiate pretraining, run the following script:

```
LLaVA-NeXT/scripts/train/mammoth_vl/pretrain_qwen_2_5.sh
```

This process will generate a `mm_projector.bin` file, which is required for the next stage.

### Stage 2: Fine-tuning(SI)

After obtaining the fine-tuning data, run the following script to begin fine-tuning:

```
LLaVA-NeXT/scripts/train/mammoth_vl/finetune_qwen_2_5_si.sh
```

### Stage 3: Fine-tuning(OV)

After obtaining the fine-tuning data, run the following script to begin fine-tuning:

```
LLaVA-NeXT/scripts/train/mammoth_vl/finetune_qwen_2_5_ov.sh
```