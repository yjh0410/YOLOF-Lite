python -m torch.distributed.run --nproc_per_node=8 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof50 \
                                                    -lr 0.12 \
                                                    -lr_bk 0.04 \
                                                    --batch_size 8 \
                                                    --img_size 800 \
                                                    --grad_clip_norm -1.0 \
                                                    --num_workers 8 \
                                                    --schedule 1x \
                                                    # --mosaic
