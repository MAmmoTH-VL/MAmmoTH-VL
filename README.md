# MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale

[Homepage](https://mammoth-vl.github.io/) | [Model](https://huggingface.co/MAmmoTH-VL/MAmmoTH-VL-8B) | [Dataset](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M) | [Code](https://github.com/MAmmoTH-VL/MAmmoTH-VL)
| [Arxiv](https://arxiv.org/abs/2412.05237) | [PDF](https://arxiv.org/pdf/2412.05237) | [Demo](https://huggingface.co/spaces/paralym/MAmmoTH-VL-8B)

This repository provides the necessary resources and guidelines for training and evaluating.

## About MAmmoTH-VL
Open-source multimodal large language models (MLLMs) have shown significant potential in a broad range of multimodal tasks. However, their reasoning capabilities remain constrained by existing instruction-tuning datasets, which were predominately repurposed from academic datasets such as VQA, AI2D, and ChartQA. These datasets target simplistic tasks, and only provide phrase-level answers without any intermediate rationales.
To address these challenges, we introduce a scalable and cost-effective method to construct a large-scale multimodal instruction-tuning dataset with rich intermediate rationales designed to elicit CoT reasoning. Using only open models, we create a dataset containing 12M instruction-response pairs to cover diverse, reasoning-intensive tasks with detailed and faithful rationales. Experiments demonstrate that training MLLMs on this dataset significantly improves reasoning capabilities, achieving state-of-the-art performance on benchmarks such as MathVerse (+8.1%), MMMU-Pro (+7%), and MuirBench (+13.3%). Additionally, the model demonstrates notable improvements of up to 4% on non-reasoning-based benchmarks. Ablation studies further highlight the importance of key components, such as rewriting and self-filtering, in the dataset construction process.

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
@article{guo2024mammothvlelicitingmultimodalreasoning,
      title={MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale}, 
      author={Jarvis Guo and Tuney Zheng and Yuelin Bai and Bo Li and Yubo Wang and King Zhu and Yizhi Li and Graham Neubig and Wenhu Chen and Xiang Yue},
      year={2024},
      eprint={2412.05237},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.05237}, 
}
```