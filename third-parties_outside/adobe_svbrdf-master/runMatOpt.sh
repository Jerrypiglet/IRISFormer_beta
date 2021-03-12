#!/bin/sh

cd /siggraphasia20dataset/adobe_svbrdf-master
python script_optimGANdatasetAllcluster.py --start $1 --end $2
