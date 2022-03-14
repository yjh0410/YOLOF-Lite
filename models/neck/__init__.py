from .dilated_encoder import DilatedEncoder
from .dynamic_attention import DynamicAttention


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'dilated_encoder':
        neck = DilatedEncoder(in_dim, 
                              out_dim, 
                              expand_ratio=cfg['expand_ratio'], 
                              dilation_list=cfg['dilation_list'])
    elif model == 'dynamic_attention':
        neck = DynamicAttention(in_dim,
                                out_dim,
                                expand_ratio=cfg['expand_ratio'],
                                nblocks=cfg['nblocks'])

    return neck
