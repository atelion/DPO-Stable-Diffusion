#!/bin/bash

accelerate launch train_diffusion_dpo_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0  \
  --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
  --output_dir="diffusion-sdxl-dpo" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=500 \
  --seed="0" \
