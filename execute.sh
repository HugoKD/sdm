export PYTORCH_CUDA_ALLOC_CONF=MAX_SPLIT_SIZE_MB=256

python3 image_train.py --data_dir ./data/ade20k --dataset_mode ade20k --lr 2e-7 --batch_size 1 --attention_resolutions 32,16,8 --diffusion_steps 2000 --image_size 512 --learn_sigma True --noise_schedule scaled_linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --use_checkpoint True --num_classes 5 --class_cond True --use_fp16 False --no_instance True --schedule_sampler ddpm
