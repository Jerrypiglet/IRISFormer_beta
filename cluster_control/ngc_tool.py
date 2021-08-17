#!/usr/bin/env python

# Example:
## Create:
# python rui_tool.py create -f rui_torch_job_create.yaml -s 'python -m torch.distributed.launch --master_port 5324 --nproc_per_node=4 train_combine_v3_RCNNOnly_bbox.py --num_layers 3 --pointnet_camH --pointnet_camH_refine --pointnet_personH_refine --loss_last_layer --accu_model --task_name DATE_pod_BASELINEv4_detachcamParamsExceptCamHinFit_lossLastLayer_NEWDataV5_SmallerPersonBins_YcLargeBins5_DETACHinput_plateau750_cascadeL3-V0INPUT-SmallerPERSONBins1-190_lr1e-5_w360-10_human175STD15W05 --config-file maskrcnn/coco_config_small_synBN1108.yaml --weight_SUN360=10. SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 16 SOLVER.PERSON_WEIGHT 0.05 SOLVER.BASE_LR 1e-5 MODEL.HUMAN.MEAN 1.75 MODEL.HUMAN.STD 0.15 MODEL.RCNN_WEIGHT_BACKBONE 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug MODEL.RCNN_WEIGHT_CLS_HEAD 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug'
## Delete:
# python rui_tool.py delete 'z-torch-job-4-20200129' --all
## Sync:
# python rui_tool.py sync sum

import argparse
# from datetime import date
from datetime import datetime
# import yaml
import subprocess
import os, sys
import pprint
import re

valid_instances = {
    'dgx1v.16g.1.norm': 'GPUs:  1   GPU Mem: 16 GB   GPU Power:  160 W   CPUs:  8   System Mem:  50 GB', 
    'dgx1v.16g.2.norm': 'GPUs:  2   GPU Mem: 16 GB   GPU Power:  160 W   CPUs: 16   System Mem: 100 GB', 
    'dgx1v.16g.4.norm': 'GPUs:  4   GPU Mem: 16 GB   GPU Power:  160 W   CPUs: 32   System Mem: 200 GB', 
    'dgx1v.16g.8.norm': 'GPUs:  8   GPU Mem: 16 GB   GPU Power:  160 W   CPUs: 64   System Mem: 400 GB', 
    'dgx1v.32g.1.norm': 'GPUs:  1   GPU Mem: 32 GB   GPU Power:  165 W   CPUs:  8   System Mem:  50 GB', 
    'dgx1v.32g.2.norm': 'GPUs:  2   GPU Mem: 32 GB   GPU Power:  165 W   CPUs: 16   System Mem: 100 GB', 
    'dgx1v.32g.4.norm': 'GPUs:  4   GPU Mem: 32 GB   GPU Power:  165 W   CPUs: 32   System Mem: 200 GB', 
    'dgx1v.32g.8.norm': 'GPUs:  8   GPU Mem: 32 GB   GPU Power:  165 W   CPUs: 64   System Mem: 400 GB', 
}

dump_source = {'binary': 's3mm1:buffer-or/ORfull-seq-240x320-smaller-RE-quarter4', \
    'pickle': 's3mm1:buffer-or/ORfull-perFramePickles-240x320-quarter'}

dump_dest = {'binary': '/dev/shm/ORfull-seq-240x320-smaller-RE-quarter', \
    'pickle': '/dev/shm/ORfull-perFramePickles-240x320-quarter'}

def parse_args():
    parser = argparse.ArgumentParser(description='Kubectl Helper')
    subparsers = parser.add_subparsers(dest='command', help='commands')
    
    # tb_parser.add_argument('--logs_path', type=str, help='python path in pod', default='/root/miniconda3/bin/python')

    create_parser = subparsers.add_parser('create', help='Create a batch of jobs')
    create_parser.add_argument('-f', '--file', type=str, help='Path to template file', default='ngc_template.json')
    create_parser.add_argument('-s', '--string', type=str, help='Input command')
    create_parser.add_argument('-d', '--deploy', action='store_true', help='deploy the code')
    create_parser.add_argument('-c', '--copy', action='store_true', help='copy dataset to tmp storage')
    create_parser.add_argument('--copy_cmd', type=str, help='cmd to transfer data via network to local fast storage', default='rclone copy --progress --fast-list --checkers=128 --transfers=128 #DUMP_SRC #DUMP_DEST')
    create_parser.add_argument('--resume', type=str, help='resume_from: e.g. 20201129-232627', default='NoCkpt')
    create_parser.add_argument('--deploy_src', type=str, help='deploy to target path', default='~/Documents/Projects/semanticInverse/train/')
    create_parser.add_argument('--deploy_s3', type=str, help='deploy s3 container', default='s3mm1:train_ngc')
    create_parser.add_argument('--deploy_tar', type=str, help='deploy to target path', default='/newfoundland/semanticInverse/job_list/train')
    # create_parser.add_argument('--python_path', type=str, help='python path in pod', default='/newfoundland/envs/semanticInverse/bin/python')
    # create_parser.add_argument('--pip_path', type=str, help='python path in pod', default='/newfoundland/envs/semanticInverse/bin/pip')
    create_parser.add_argument('--python_path', type=str, help='python path in pod', default='/newfoundland/envs/py38/bin/python') # torch 1.8.0
    create_parser.add_argument('--pip_path', type=str, help='python path in pod', default='/newfoundland/envs/py38/bin/pip') # torch 1.8.0
    # create_parser.add_argument('--gpus', type=int, help='nubmer of GPUs', default=2)  
    # create_parser.add_argument('--cpur', type=int, help='request of CPUs', default=10)
    # create_parser.add_argument('--cpul', type=int, help='limit of CPUs', default=30)
    # create_parser.add_argument('--memr', type=int, help='request of memory in Gi', default=40)
    # create_parser.add_argument('--meml', type=int, help='limit of memory in Gi', default=50)
    create_parser.add_argument('--namespace', type=str, help='namespace of the job', default='ucsd-ravigroup')
    create_parser.add_argument('-i', '--instance', type=str, help='instance of the job', default='dgx1v.16g.1.norm')
    create_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    create_parser.add_argument('-r', '--num-replicas', type=int, help='Number of replicas')
    # create_parser.add_argument('-n', '--namespace', type=str, help='namespace')
    create_parser.add_argument('vals', help='Values to replace', nargs=argparse.REMAINDER)
    create_parser.add_argument('--debug', action='store_true', help='if debugging')


    create_parser = subparsers.add_parser('list', help='List jobs')
    create_parser.add_argument('--instance', type=str, help='instance of the job', default='dgx1v.16g.1.norm')

    args = parser.parse_args()

    assert args.instance in valid_instances, 'Invalid Instance: %s'%args.instance
    
    return args


# def iterate_dict(input_dict, var_replace_list=None):
#     if not isinstance(input_dict, dict):
#         if isinstance(input_dict, list):
#             return [iterate_dict(x, var_replace_list=var_replace_list) for x in input_dict]
#         else:
#             if var_replace_list is not None:
#                 for var in var_replace_list:
#                     # if input_dict == var:
#                     #     return var_replace_list[var]
#                     # print('------', str(input_dict), var)
#                     if var in str(input_dict):
#                         print(var, input_dict, '------>', input_dict.replace(str(var), str(var_replace_list[var])))
#                         return input_dict.replace(str(var), str(var_replace_list[var]))
#             return input_dict
    
#     new_dict = {}
#     for key in input_dict:
#         new_dict.update({key: iterate_dict(input_dict[key], var_replace_list=var_replace_list)})

#     return new_dict

# def replace_vars(args):
#     var_mapping = {'gpus': '#GPUS', 'cpur': '#CPUR', 'cpul': '#CPUL', 'memr': '#MEMR', 'meml': '#MEML', 'namespace': '#NAMESPACE'}
#     var_replace_list = {}
#     for var in args:
#         if var in var_mapping:
#             var_replace_list.update({var_mapping[var]: args[var]})
#     return var_replace_list

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1

# def load_json(yaml_path):
#     with open(yaml_path, 'r') as stream:
#         try:
#             loaded = yaml.load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     return loaded

# def dump_yaml(yaml_path, json_content):
#     with open(yaml_path, 'w') as stream:
#         try:
#             yaml.dump(json_content, stream, default_flow_style=False)
#         except yaml.YAMLError as exc:
#             print(exc)

import json

def load_json(json_path):
    with open(str(json_path)) as f:
        data = json.load(f)
    return data

def dump_json(json_path, json_content):
    # with open(json_path, 'w') as stream:
    #     try:
    #         json.dump(json_content, stream, default_flow_style=False)
    #     except json.jsonError as exc:
    #         print(exc)
    with open(str(json_path), 'w') as json_file:
        json.dump(json_content, json_file)

def run_command(command, namespace=None):
    if namespace is not None:
        command += ' --namespace='+namespace
    ret = subprocess.check_output(command, shell=True)
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret

def run_command_generic(command):
    #This command could have multiple commands separated by a new line \n
    # some_command = "export PATH=$PATH://server.sample.mo/app/bin \n customupload abc.txt"

    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()  

    #This makes the wait possible
    p_status = p.wait()

    #This will give you the output of the command being executed
    print("Command output: " + output.decode('utf-8'))

# def get_pods(pattern, namespace=None):
    # command = 'kubectl get pods -o custom-columns=:.metadata.name,:.status.succeeded'
    # if namespace is not None:
    #     command += ' --namespace='+namespace
    # ret = run_command(command)
    # pods = list(filter(None, ret.splitlines()))[1:]
    # pods = [pod.split() for pod in pods]
    # pods = list(filter(lambda x: re.match(pattern, x[0]), pods))
    # pods = [pod[0] for pod in pods]
    # if len(pods) == 1:
    #     pods = str(pods[0])
    # # pprint.pprint(pods)
    # print(pods)
    # return pods

def create_job_from_json(json_filename):
    # https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output
    result = subprocess.run('ngc batch run -f %s'%json_filename, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    stdout = result.stdout.decode('utf-8')
    print('>>>>>>>>>>>> ngc create %s result:'%json_filename)
    print(stdout)

def deploy_to_s3(args):
    deploy_command = 'cd /home/ruzhu/Documents/Projects/semanticInverse && rsync -ah --progress mm1:/home/ruizhu/Documents/Projects/semanticInverse/train . --filter="- *.pyc" --filter="- */__pycache__"'
    deploy_command += ' && rclone sync %s %s/%s'%(args.deploy_src, args.deploy_s3, args.datetime_str)
    # os.system(deploy_command)
    print('>>>>>>>>>>>> deploying with: %s'%deploy_command)
    run_command_generic(deploy_command)
    print('>>>>>>>>>>>> deployed/')

def create(args):
    if args.resume != 'NoCkpt':
        # datetime_str = args.resume
        # tmp_json_filaname = 'tasks/%s/tmp_%s.yaml'%(datetime_str, datetime_str)
        # print('============ Resuming from YAML file: %s'%tmp_json_filaname)
        # json_content = load_json(tmp_json_filaname)
        # os.system('kubectl delete job '+json_content['metadata']['name'])
        # print('============ Task removed: %s'%json_content['metadata']['name'])
        # json_content['metadata']['name'] += '-re'
        # command_str = json_content['spec']['template']['spec']['containers'][0]['args'][0]
        # s_split = command_str.split(' ')
        # start_index = s_split.index('rclone')
        # for i in range(5):
        #     s_split.pop(start_index)
        # insert_index = s_split.index('--if_cluster')
        # s_split.insert(insert_index+1, '--reset_latest_ckpt')
        # command_str = ' '.join(s_split)
        # json_content['spec']['template']['spec']['containers'][0]['args'][0] = command_str
        # tmp_json_filaname = tmp_json_filaname.replace('.yaml', '-RE.yaml')
        # dump_yaml(tmp_json_filaname, json_content)
        # print('============ YAML file dumped to %s'%tmp_json_filaname)
        # print(command_str)
        pass
    else:
        command_str = args.string
        args.datetime_str = get_datetime()
        command_str = command_str.replace('DATE', args.datetime_str)
        print('------------ Command string:')
        print(command_str)

        json_content = load_json(args.file)
        # var_replace_list = replace_vars(vars(args))
        # json_content = iterate_dict(json_content, var_replace_list=var_replace_list)
        print('------------ json_content:')
        print(json_content)

        command_str = command_str.replace('python', args.python_path)
        command_str = command_str.replace('--if_cluster', '--if_cluster --cluster ngc')
        if args.deploy:
            args.deploy_tar += '-%s'%args.datetime_str
            command_str = 'rclone copy %s/%s %s && cd %s && '%(args.deploy_s3, args.datetime_str, args.deploy_tar, args.deploy_tar) + command_str

        splits = command_str.split(' ')
        task_name = [splits[x+1] for x in range(len(splits)) if splits[x].startswith('--task_name')][0]
        task_name_ngc = task_name.lower()
        model_name_ngc = task_name_ngc.split('_')[2]
        # task_name = task_name.replace('[\'', '').replace('\']', '')
        json_content['command'] = json_content['command'].replace('sleep 480h', '')
        if 'DATASET.if_to_memory True' in command_str:
            if_binary = 'DATASET.if_binary True' in command_str
            if_pickle = 'DATASET.if_pickle True' in command_str
            assert if_binary or if_pickle
            assert not(if_binary and if_pickle)
            if if_binary:
                dump_source_str = dump_source['binary']
                dump_dest_str = dump_dest['binary']
            elif if_pickle:
                dump_source_str = dump_source['pickle']
                dump_dest_str = dump_dest['pickle']
            else:
                assert False
            copy_cmd = args.copy_cmd
            copy_cmd = copy_cmd.replace('#DUMP_SRC', dump_source_str).replace('#DUMP_DEST', dump_dest_str)
            json_content['command'] += copy_cmd
            json_content['command'] += ' && '
            print('[!!!!!!!] adding dumping to memory cmd: %s'%copy_cmd)
        # print(command_str)
        json_content['command'] += command_str
        # if args.copy:
        #     json_content['command'] += ' DATASET.binary_if_to_memory True'
        json_content['name'] += '.%s.%s'%(model_name_ngc, task_name_ngc)
        if len(json_content['name'])>=128:
            json_content['name'] = json_content['name'][:128]
        json_content['aceInstance'] = args.instance

        tmp_json_filaname = 'yamls/tmp_%s.json'%args.datetime_str
        dump_json(tmp_json_filaname, json_content)
        print('============ JSON file dumped to %s'%tmp_json_filaname)


    if args.deploy:
        deploy_to_s3(args)
    
    create_job_from_json(tmp_json_filaname)

    if args.resume == 'NoCkpt':
        task_dir = './tasks/%s'%args.datetime_str
        os.mkdir(task_dir)
        os.system('cp %s %s/'%(tmp_json_filaname, task_dir))
        text_file = open(task_dir + "/command.txt", "w")
        n = text_file.write(command_str)
        text_file.close()
        print('yaml and command file saved to %s'%task_dir)

    print(args.datetime_str)

    # os.remove(tmp_json_filaname)
    # print('========= REMOVED YAML file %s'%tmp_json_filaname)

    # pod_name = get_pods(json_content['metadata']['name'])
    
    # if pod_name and args.resume == 'NoCkpt':
    #     with open("all_commands.txt", "a+") as f:
    #         f.write("%s-%s\n" % (pod_name, datetime_str))
    #         f.write("%s\n" % command_str)

def delete(args, pattern=None, delete_all=False, answer=None):
    if pattern is None:
        pattern = args.pattern
    if args.namespace:
        namespace = args.namespace
    else:
        namespace = None
    print('Trying to delete pattern %s...'%pattern)
    ret = run_command('kubectl get jobs -o custom-columns=:.metadata.name,:.status.succeeded', namespace)
    jobs = list(filter(None, ret.splitlines()))[1:]
    jobs = [job.split() for job in jobs]
    if args.debug:
        print('Got jobs: ', jobs, pattern)
    if args.all or delete_all:
        jobs = list(filter(lambda x: re.match(pattern, x[0]), jobs))
    else:
        jobs = list(filter(lambda x: re.match(pattern, x[0]) and x[1] == '1', jobs))
    # pprint.pprint(jobs)
    # if debug:
    print('Filtered jobs:', jobs)
    if len(jobs) == 0:
        return
    while True:
        if answer is None:
            answer = input('Do you want to delete those jobs?[y/n]')
        if answer == 'y':
            job_names = [x[0] for x in jobs]
            ret = run_command('kubectl delete jobs ' + ' '.join(job_names), namespace)
            if isinstance(ret, bytes):
                ret = ret.decode()
            print(ret)
            break
        elif answer == 'n':
            break

def list(args):
    command = 'ngc batch list --format_type json'
    ret = run_command(command)
    job_list_dict = json.loads(ret)
    print(job_list_dict)


def main():
    args = parse_args()

    if args.command == 'create':
        create(args)
    elif args.command == 'list':
        list(args)

if __name__ == "__main__":
    main()