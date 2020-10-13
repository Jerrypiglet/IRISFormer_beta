#!/usr/bin/env python
import argparse
import subprocess
import pipes
import sys
import os
import re

def get_file(path):
    with open(path, "r") as f:
        return str(f.read())


def quote(v, ):
    if isinstance(v, list):
        return "[" + ", ".join(quote(x) for x in v) + "]"
    else:
        v = pipes.quote(v)
        if not v.startswith("'") or not v.startswith('"'):
            v = '"' + v + '"'
        return v


def variable_sub(s, table, no_quote):
    for k, v in table.iteritems():
        if k not  in no_quote:
            t = quote(v)
        else:
            t = v
        s = s.replace("${"+k+"}", t)

    return s


def parse_mem(mem):
    m = re.search("(\d+[KMGT]?)(I)?(B)?", mem.upper())
    if m is None:
        raise "{} is invalid size".format(mem)

    return m.group(1)

def get_mem_str(mem):
    if mem is None:
        return ""
    return "memory: {}i".format(parse_mem(mem))

def generate_yaml(type_name, cmd, name, mem_limit, mem_req, node, ngpus, ssh_key, cpu, special_node):
    variables={}
    variables["CMD"] = [cmd[0]]
    variables["ARGS"] = cmd[1:]
    variables["NAME"] = name

    variables["CPU_REQ"] = str(cpu)

    variables["NGPUS"] = str(ngpus)

    variables["MEM_LIMIT"] = get_mem_str(mem_limit)
    variables["MEM_REQ"] = get_mem_str(mem_req)
    variables["NODE_HOSTNAME"] = "kubernetes.io/hostname: " + quote(node) if node is not None else ""

    variables["SSH_KEY_NAME"] = "secretName: " + quote("ssh-key-secret-" + ssh_key) if ssh_key else ""

    variables['TOLERATIONS'] = 'tolerations: [ {"key": "region","operator": "Equal","value": "allow","effect": "NoSchedule"}]' if special_node else ''

    script_dir = os.path.dirname(os.path.realpath(__file__))
    return (variable_sub(get_file(os.path.join(script_dir, type_name + ".yaml")), variables, {"MEM_LIMIT", "MEM_REQ", "NODE_HOSTNAME", "SSH_KEY_NAME", "NGPUS", "TOLERATIONS", "CPU_REQ"}))


def run_cmd(cmd, stdin_str):
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr,stdin=subprocess.PIPE)

    out, err = proc.communicate(input=stdin_str)
    result = proc.returncode == 0

    if result == False:
        print("Error")

    return proc.returncode


def gen_kubectl_run_cmd(podname, type_name, repo=None, branch=None, cmds=None, mem_limit=None, mem_req=None, node=None, ngpus=1, key=None, cpu=None):
    if node == 'ravi':
        node = 'k8s-ravi-01.calit2.optiputer.net'
    cmds_len = len(cmds)

    if repo is not None:
        cmds = ["/hooks/startup.sh", repo, branch] + cmds
    else:
        cmds = ["/hooks/startup.sh"]

    kube_cmd = ["kubectl", "create", "-f", "-"]

    if False:
        kube_cmd.append("--dry-run")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    if cmds_len != 0 and os.path.join(script_dir, 'job_' + type_name + ".yaml"):
        print("Switching job")
        type_name = 'job_' + type_name

    if cmds_len != 0  and repo is None:
        repo = 'git@github.com:ak3636/sample_project.git'


    special_node = False
    if node == 'k8s-ravi-01.calit2.optiputer.net':
        special_node = True

    yaml_str = (generate_yaml(type_name, cmds, podname, mem_limit, mem_req, node, ngpus, key, cpu, special_node))
    print(yaml_str)
    return run_cmd(kube_cmd, yaml_str)


def get_arg(args):
    if args is not None:
        return args[0]
    return None


def main():
    parser = argparse.ArgumentParser(description='Deploying command')

    parser.add_argument('--name', metavar='pod_name', type=str, nargs=1,
                        help='Pod name', required=True)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--repo', metavar='my_git_repo', type=str, nargs=1,
                        help='Path to git repo')

    parser.add_argument('--branch', metavar='git_branch', type=str, nargs=1,
                        help='Git branch', default=["master"])

    group.add_argument('--empty',  action='store_true',
                        help='Creates pods without repo')

    parser.add_argument('--type',  type=str, nargs=1, choices=['deeplearning', 'no_gpu'],
                        help='Type of pod', default=['deeplearning'])

    parser.add_argument('--ngpus',  type=int,
                        help='Number of gpus', default=1 )

    parser.add_argument("command", nargs=argparse.REMAINDER)

    parser.add_argument('--meml', type=str, nargs=1, default=['12G'],
                        help='Memory limit')

    parser.add_argument('--memr', type=str, nargs=1, default=['12G'],
                        help='Memory request')

    parser.add_argument('--cpu', type=float, default=8,
                    help='cpu request')

    parser.add_argument('--node', type=str, nargs=1,
                        help='Node name')

    parser.add_argument('--key', type=str, nargs=1,
                        help='SSH key name for git', default=[None] )

    args = parser.parse_args()

    if args.branch is None and args.repo is None and len(args.command) == 0:
        args.empty = True

    ret_code = gen_kubectl_run_cmd(args.name[0], args.type[0], get_arg(args.repo), get_arg(args.branch), args.command, mem_limit=get_arg(args.meml), mem_req=get_arg(args.memr), node=get_arg(args.node), ngpus=args.ngpus, key=args.key[0], cpu=args.cpu)
    sys.exit(ret_code)

if __name__ == main():
    main()
