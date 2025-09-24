SAVE_DIR=/scratch/prj/inf_multimodal_qa/scratch_tmp/efficient_cot/icae/codi2/gpt_codi_lora_nl

mkdir -p "$SAVE_DIR"

cp scripts/train_28.20_ce_noref_new_noaux_lat6.sh "$SAVE_DIR"

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name gsm8k_llama1b_latent_baseline \
	--logging_dir "$SAVE_DIR/logs"\
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name icot-full \
	--seed 11 \
	--model_max_length 512 \
	--per_device_train_batch_size 64 \
  	--gradient_accumulation_steps 2 \
	--bf16 \
	--num_train_epochs 40 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_safetensors False \
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to tensorboard \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--exp_data_num 2000 \
	--remove_eos True \
	--print_ref_model_stats True \
