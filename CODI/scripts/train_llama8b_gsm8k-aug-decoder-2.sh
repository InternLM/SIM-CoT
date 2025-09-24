SAVE_DIR=/mnt/shared-storage-user/weixilin/MLLM/coconut/codi/outputs

mkdir -p "$SAVE_DIR"

# cp scripts/train_28.20_ce_llama1b_dynamic-teacher_factor-exp_lat6.sh "$SAVE_DIR"

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_llama8b_latent_baseline-decoder-2-9_8 \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path /mnt/shared-storage-user/mllm/shared/mllm_ckpts/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
	--data_name icot \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
  	--gradient_accumulation_steps 16 \
	--bf16 \
	--num_train_epochs 6 \
	--learning_rate 1e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_total_limit 1 \
	  --save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
   --num_latent 6 \
   --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 4096 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 200 \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 200 \
	--use_decoder True

