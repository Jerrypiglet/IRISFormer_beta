import torch.distributed as dist

def postprocess_detectron(input_dict, output_dict, loss_dict, opt, time_meters, is_train, if_vis=False):
    detectron_output_list_of_dicts = output_dict['detectron_output_list_of_dicts']
    if is_train:
        detectron_loss_dict = {x.replace('loss_', 'loss_detectron-'):detectron_output_list_of_dicts[x] for x in detectron_output_list_of_dicts}
        detectron_loss_dict.update({'loss_detectron-ALL': sum(detectron_loss_dict.values())})
        
        loss_dict.update(detectron_loss_dict)
    else:
        output_dict.update({'output_detectron': detectron_output_list_of_dicts}) # [{instances: []}]
    return output_dict, loss_dict