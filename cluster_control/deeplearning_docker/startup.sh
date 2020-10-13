#!/bin/bash
export PATH="$PATH":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

echo Startup hook
cp -r /root/ssh_mount /root/.ssh
chmod -R 400  /root/.ssh
ls -al /root/.ssh/
# Enables cloning from github without prompt
echo "Host *
   StrictHostKeyChecking no
   UserKnownHostsFile=/dev/null" >> ~/.ssh/config

if [ "$#" -eq 0 ]; then
    echo "Running forever"
    sleep infinity
else
    echo "Cloning: $1:$2"
    git clone  --recursive --depth=1 -b "$2"  "$1" my_script
    cd my_script
    mkdir -p /VisualComputing/logs/
    LOG_FILENAME=/VisualComputing/logs/out_$(date +%s)
    CMD=$(printf '%q ' "${@:3}")
    echo "Running following command: " "$CMD" >> "$LOG_FILENAME"
    script -f -e  -a -c "$CMD" "$LOG_FILENAME"
    RET=$?
    echo "Exited with code: " $RET >> "$LOG_FILENAME"
    exit $RET

fi
