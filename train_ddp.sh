# 2 GPU
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    --num_gpu 2 \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof50 \
                                                    -lr 0.03 \
                                                    -lr_bk 0.01 \
                                                    --batch_size 8 \
                                                    --img_size 928 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x \
                                                    # --mosaic
