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


def gather_lists(list0, num_gpus, process_group=None):
    list0_allgather = [None for _ in range(num_gpus)]
    if process_group is None:
        # print('======', list0[:10],len(list0))
        dist.all_gather_object(list0_allgather, list0)
        # print('======<')
    else:
        dist.all_gather_object(list0_allgather, list0, group=process_group)
    # print(len(list0_allgather), len(list0_allgather[0]), '<<<<<<<<<<-------', opt.rank)
    list0_allgather = [item for sublist in list0_allgather for item in sublist]
    return list0_allgather
