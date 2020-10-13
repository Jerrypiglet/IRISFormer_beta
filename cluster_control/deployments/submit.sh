#!/bin/bash


name='zhengqin-3'

ngpus=1
command="ls"
node='k8s-ravi-01.calit2.optiputer.net'
node='k8s-haosu-01.sdsc.optiputer.net'
# node='k8s-chase-ci-07.calit2.optiputer.net'
# node='clu-fiona2.ucmerced.edu'
# node='fiona8-2.calit2.uci.edu'
# node='k8s-chase-ci-01.noc.ucsb.edu'
# node='patternlab.calit2.optiputer.net' # 24G
# node='evldtn.evl.uic.edu' # 24G
# node='k8s-chase-ci-08.calit2.optiputer.net'


# /home/saibi/SSD/GitHub/cluster_control/deployments/deploy  \
    # --name $name \
    # --ngpus $ngpus \
    # --memr '24G' \
    # --meml '30G' \
    # --cpu 6 \
    # $command \

/home/zhl/cluster_control/deployments/deploy  \
    --name $name \
    --ngpus $ngpus \
    --memr '2G' \
    --meml '2G' \
    --node $node \
    --cpu 3 \
    $command \

