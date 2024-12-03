## Evaluation

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

`lmms-eval/eval_mammoth_vl_example.sh` shows an example script to run evaluation.