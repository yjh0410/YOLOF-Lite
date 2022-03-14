python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v yoloq50 \
                                                    --batch_size 32 \
                                                    --img_size 640 \
                                                    --lr 0.01 \
                                                    --optimizer sgd \
                                                    --wp_iter 125 \
                                                    --num_workers 8 \
                                                    --num_gpu 2 \
                                                    --max_epoch 13 \
                                                    --lr_epoch 8 10
                                                    # --multi_scale \
                                                    # --mosaic
