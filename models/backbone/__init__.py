from .resnet import build_resnet
from .cspdarknet import build_cspdarknet


def build_backbone(model_name='resnet50-d', pretrained=False, norm_type='BN'):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet(model_name=model_name, 
                                       pretrained=pretrained,
                                       norm_type=norm_type)

    elif 'cspdarknet' in model_name:
        model, feat_dim = build_cspdarknet(model_name=model_name, 
                                           pretrained=pretrained,
                                           norm_type=norm_type)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
