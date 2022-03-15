# YOLOf config


yolof_config = {
    'yolof18': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': 32,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },

    'yolof50': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': 32,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667
    },

    'yolof50-DC5': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # model
        'backbone': 'resnet50-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },

    'yolof101': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # model
        'backbone': 'resnet101',
        'norm_type': 'FrozeBN',
        'stride': 32,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },

    'yolof101-DC5': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        # model
        'backbone': 'resnet101-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },

    'yolof53': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'BGR',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [1.0, 1.0, 1.0],
        # model
        'backbone': 'cspdarknet53',
        'norm_type': 'FrozeBN',
        'stride': 32,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [2, 4, 6, 8],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },

    'yolof53-DC5': {
        # input
        'multi_scale': [480, 544, 608, 672, 736, 800],
        'format': 'BGR',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [1.0, 1.0, 1.0],
        # model
        'backbone': 'cspdarknet53-d',
        'norm_type': 'FrozeBN',
        'stride': 16,
        # neck
        'neck': 'dilated_encoder',
        'dilation_list': [4, 8, 12, 16],
        'expand_ratio': 0.25,
        # head
        'head_dim': 512,
        'head': 'naive_head',
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # multi scale
        'random_size': [320, 384, 448, 512, 576, 640],
        # anchor box
        'anchor_size': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        # matcher
        'topk': 4,
        'iou_t': 0.15,
        'igt': 0.7,
        'ctr_clamp': 32,
        # optimizer
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667
    },
}