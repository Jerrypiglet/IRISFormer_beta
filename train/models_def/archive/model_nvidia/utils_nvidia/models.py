'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

import torch

def load_pretrained_model(model, pretrained_path, model_dict_name='state_dict', optimizer=None, optimizer_dict_name='optimizer',
                          strict=True):
    r'''
    load the pre-trained model, if needed, also load the optimizer status
    '''

    print(f'---load model from {pretrained_path}---')
    pre_model_dict_info = torch.load(pretrained_path) 

    if isinstance(pre_model_dict_info, dict) and model_dict_name in pre_model_dict_info:
        pre_model_dict = pre_model_dict_info[model_dict_name]
    else:
        pre_model_dict = pre_model_dict_info

    model_dict = model.state_dict()
    pre_model_dict_feat = {k:v for k,v in pre_model_dict.items() if k in model_dict}

    # update the entries #
    model_dict.update( pre_model_dict_feat)
    # load the new state dict #
    model.load_state_dict( pre_model_dict_feat, strict=False)

    if optimizer is not None and optimizer in pre_model_dict_info:
        optimizer.load_state_dict(pre_model_dict_info[optimizer_dict_name])
        print('Also loaded the optimizer status')
    else:
        print('Ignore the optimizer during loading the checkpoint')

    return pre_model_dict_info

def save_args(args, filename='args.txt'):
    r'''
    Save the parsed arguments to file.
    This function is useful for recoding the experiment parameters.
    inputs:
    arg - the input arguments 
    filename (args.txt) - the txt file that saves the arguments 
    '''
    arg_str = []
    for arg in vars(args):
        arg_str.append( str(arg) + ': ' + str(getattr(args, arg)) ) 
    with open(filename, 'w') as f:
        for arg_str_ in arg_str:
            f.write('%s\n'%(arg_str_))
    return arg_str