export HF_HOME=xxx
export HF_TOKEN=xxx
export MLP_WORKER_0_PORT=xxx 
export OPENAI_API_KEY=xxx
source yourpath/miniconda3/bin/activate lmms-eval
FINAL_RUN_NAME=$1
Task_Name=$2

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmmu_val --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks realworldqa --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks seedbench --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks ai2d  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmstar --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks chartqa --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks mmvet  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes=1 -m lmms_eval --model llava_onevision --model_args pretrained=${FINAL_RUN_NAME},conv_template=qwen_2_5,model_name=llava_qwen --tasks llava_wilder_small  --batch_size 1 --log_samples --log_samples_suffix ${Task_Name} --output_path xxx &
wait
