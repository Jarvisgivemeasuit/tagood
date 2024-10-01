OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python ood_inference_distance.py \
  --batch-size 128 \
  --num-worker 64 \
  --num_dev 1 \
  --ind_root /path/to/ind_tokens \
  --ood_root /path/to/ood_tokens \
  --checkpoint-dir /path/to/checkpoint_folder \
  --save-dir /path/to/save_folder
