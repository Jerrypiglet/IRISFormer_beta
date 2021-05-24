pip3 install -U torchvision==0.8.1  cython Pillow==6.2.2
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install -U opencv-python==4.2.0.34
git clone https://github.com/facebookresearch/detectron2 /eccv20dataset/elvis/outfileelvis/detectron2_repo
pip3 install -e /eccv20dataset/elvis/outfileelvis/detectron2_repo
pip3 install -U google-colab
pip3 install tensorboard==1.15
git config --global user.email "elvishelvis6@gmail.com"
git config --global user.name "Elvis Shi"
mkdir /eccv20dataset/elvis/outfileelvis
#python3 /eccv20dataset/elvis/labelTransform/processopen40.py
#python3 /eccv20dataset/elvis/ECCV2021/40opentrain.py
