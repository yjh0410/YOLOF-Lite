python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yolof50 \
                                                    -lr 0.12 \
                                                    -lr_bk 0.04 \
                                                    --batch_size 32 \
                                                    --img_size 800 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 8 \
                                                    --num_gpu 2 \
                                                    --max_epoch 12 \
                                                    --lr_epoch 8 10
                                                    # --multi_scale \
                                                    # --mosaic
