python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof101 \
        -lr 0.03 \
        -lr_bk 0.01 \
        --batch_size 16 \
        --img_size 928 \
        --grad_clip_norm -1.0 \
        --num_workers 8 \
        --schedule 1x \
        # --mosaic
