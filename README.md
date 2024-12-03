# MAmmoTH-VL: Scaling Synthetic Multimodal Instruction Data with Open Models

[Homepage](https://mammoth-vl.github.io/) | [Model](https://huggingface.co/MMSFT/MAmmoTH-VL-8B) | [Dataset](https://huggingface.co/datasets/MMSFT/MAmmoTH-VL-12M) | [Github](https://github.com/orgs/MAmmoTH-VL/MAmmoTH-VL)
| [Arxiv](https://arxiv.org/abs/2410.16153) | [PDF](https://arxiv.org/pdf/2410.16153) | [Demo](https://huggingface.co/spaces/MMSFT/MAmmoTH-VL-8B)

This repository provides the necessary resources and guidelines for training and evaluating.

## About MAmmoTH-VL
"Connector-training" methods (e.g., LLaVA) have significantly advanced open multimodal large language models (MLLMs) by integrating pretrained visual encoders with LLMs through a simple projection layer. These methods rely on high-quality supervised fine-tuning (SFT) data, yet generating large-scale datasets remains challenging in open-source settings due to constraints in cost, diversity, and accessibility. This paper introduces a cost-effective, scalable methodology to construct a 12-million-entry multimodal dataset using only open models. The approach involves (1) collecting and categorizing diverse instruction datasets into task-specific categories, (2) augmenting and rewriting data with open-weight MLLMs and LLMs, and (3)  quality filtering to ensure relevance and reduce hallucination. Our experimental results show that a model trained on this dataset, MAmmoTH-VL-8B, a fully open-source MLLM, achieves state-of-the-art performance on 10 datasets. This work establishes a scalable pathway for advancing open-source MLLMs with high-quality datasets.

## Repository Structure

The repository is organized into the following directories:

- **train**: Contains scripts and instructions for pretraining and finetuning the PANGEA model. We have made modifications from the open-source [Llava-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) repository.

- **evaluation**: Includes code and datasets to assess the model's performance across various tasks and languages. The code is modified from the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) repository for evaluation.

<!-- - **data**: Provides examples of the finetuning data used for PANGEA, facilitating understanding of the data format and structure. -->

<!-- - **predict**: Example Python code usage of Pangea-7B. -->

## Setting Up

To get started with MAmmoTH-VL:

1. **Clone the Repository**: Use Git to clone the repository to your local environment.

2. **Install Dependencies**: Ensure you have the required dependencies installed. For training, you need to do 

```bash
cd train/LLaVA-NeXT
pip install -e ".[train]"
```

For evaluation, you need to do

```bash
cd evaluation/lmms-eval
pip install -e .
```

3. **Download Datasets**: Acquire the necessary pretraining and fine-tuning datasets. For pretraining, download the LLaVA-Pretrain dataset from [HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). For finetuning, download the MAmmoTH-VL-12M dataset from [HuggingFace](https://huggingface.co/datasets/MMSFT/MAmmoTH-VL-12M).

<!-- ## Quick Start
After installing the required packages in `train/LLaVA-NeXT`, you could go to `predict` and run example Python code using MAmmoTH-VL-8B.

```bash
cd predict
python predict_all.py # You could evaluate both multimodal inputs and text-only inputs with this script
python predict_multimodal.py # You could evaluate multimodal inputs with this script but not text-only inputs
python predict_text_only.py # You could evaluate text-only inputs with this script but not multimodal inputs
``` -->

## Sample Data and Format

Here is an example of training data:

```json
{
   "id": str,
   "image": str/array,
   "video": str,
   "conversations": array,
}
```
<!-- ![ex](data/images/cultural/2433684022797.0.jpg)

The corresponding image file for this example is located at `data/images/cultural/2433684022797.0.jpg`. -->

### Data Structure:
- **id**: Unique identifier for the data sample.
- **image**: The path to the image file used in this instance.
- **video**: The path to the video file used in this instance.
- **conversations**: A series of conversations between the "human" and the model (in this case, referred to as "gpt").
   - **from**: Identifies the speaker (either "human" or "gpt").
   - **value**: The content of the message, which can include both text and image references.
<!-- - **language**: The language of the instruction and conversation (in this example, it is Korean). -->

## Training

### Stage 1: Pretraining

After setting up, initiate the pretraining phase:

1. **Run the Pretraining Script**:

```bash
cd train

bash LLaVA-NeXT/scripts/train/mammoth_vl/pretrain_qwen_2_5.sh
```
This result in the creation of a `mm_projector.bin` file essential for the finetuning stage.

Once pretraining is complete, proceed to finetune the model: **Ensure Fine-tuning Data is Available**

### Stage 2: Fine-tuning(SI)

After obtaining the fine-tuning data, run the following script to begin fine-tuning:

```
cd train

bash LLaVA-NeXT/scripts/train/mammoth_vl/finetune_qwen_2_5_si.sh
```

### Stage 3: Fine-tuning(OV)

After obtaining the fine-tuning data, run the following script to begin fine-tuning:

```
cd train

bash LLaVA-NeXT/scripts/train/mammoth_vl/finetune_qwen_2_5_ov.sh
```

## Evaluation

To evaluate the model's capabilities:

1. **Navigate to the Evaluation Directory**:

```bash
cd eval
```

2. **Run the Evaluation Script**:

To run the evaluation, use the following command:

```bash
export HF_HOME=xxx
export HF_TOKEN=xxx
export MLP_WORKER_0_PORT=xxx 
export OPENAI_API_KEY=xxx
source yourpath/miniconda3/bin/activate lmms-eval
FINAL_RUN_NAME=$1
Task_Name=$2

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmmu_val --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx
```

Here, `${FINAL_RUN_NAME}` refers to either a locally available model or a model on HuggingFace, identified by its repository ID. Note that we use `conv_template=qwen_2_5` for MAmmoTH-VL. You should remove this or change to other conv_template when appropriate.

`eval/lmms-eval/eval_mammoth_vl_example.sh` shows an example script to run evaluation.

## Citation
```
```