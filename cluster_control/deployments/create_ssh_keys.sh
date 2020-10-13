#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Please provide ssh key name"
    exit 1
fi

cd `dirname $0`

SECRETS_DIR="../../secrets-ssh-${1}"
mkdir -p "$SECRETS_DIR"
SSH_FILE="$SECRETS_DIR/id_rsa"

if [ -f "$SSH_FILE" ]; then
    echo "Ssh key already exist! Delete ${SECRETS_DIR} and 'kubectl delete secret ssh-key-secret-$1'"
    exit -1
fi

set -eo pipefail

ssh-keygen -t rsa -b 4096 -N "" -f "$SSH_FILE" > /dev/null
kubectl create secret generic "ssh-key-secret-$1" --from-file=id_rsa=${SSH_FILE} --from-file=id_rsa.pub=${SSH_FILE}.pub
echo Success. Public key is stored at: ${SSH_FILE}.pub
echo Now copy this public key to your github deployment keys
cat ${SSH_FILE}.pub
