THRES=0.5
NUMWORKER=32
BATCHSIZE=80
IMAGENETPATH=/path/to/imagenet
OODPATH=/path/to/ood

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset ImageNet \
  --mode train \
  --threshold $THRES \
  --output-dir outputs/data_tokens_attn_multi_$THRES \
  --img_root $IMAGENETPATH

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset ImageNet \
  --mode val \
  --threshold $THRES \
  --output-dir outputs/val_tokens_attn_multi_$THRES \
  --img_root $IMAGENETPATH

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset iNaturalist \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset SUN \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset Places \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset Textures \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker $NUMWORKER \
  --shuffle False \
  --num_dev 1 \
  --dataset ImageNet-O \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH

  OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python gen_data_tokens.py \
  --tagging-type ram \
  --checkpoint /home/et21-lijl/Documents/recognize-anything/pretrained/ram_swin_large_14m.pth \
  --batch-size $BATCHSIZE \
  --num-worker 32 \
  --shuffle False \
  --num_dev 1 \
  --dataset OpenImage \
  --mode val \
  --output-dir outputs/ood_tokens_multi_$THRES \
  --threshold $THRES \
  --img_root $OODPATH