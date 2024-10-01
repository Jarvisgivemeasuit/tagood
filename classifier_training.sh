OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 TORCH_DISTRIBUTED_DEBUG=DETAIL python classifier_training_ema.py \
  --attn_root /home/et21-lijl/Documents/multimodalood/outputs/data_tokens_attn_multi_0.5 \
  --batch-size 256 \
  --num-worker 64 \
  --shuffle True \
  --num_dev 1 \
  --lr 0.01 \
  --lr-min 1e-5 \
  --epoch 100 \
  --output-dir outputs/ood_0930

