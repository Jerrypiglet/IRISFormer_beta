# IRISFormer-non-official

This is a repo in the works; frequent changes can be expected.

## Environment
Tested with conda environment with Python 3.10 and Torch 2.0.0.

``` bash
pip install -r requirements.txt
```

Training logs will be dumped to `logs`, from which you can launch Tensorboard to monitor the stats ([demo 1](https://i.imgur.com/LL1LCza.jpg), [demo 2](https://i.imgur.com/R1STlmA.jpg))

## Train BRDF and geometry estimation

Below is an example of training BRDF and goemetry (albedo, roughness, normals, depth), on 4 GPUs, with validation over validation split, and visualization of selected samples to Tensorboard. 


``` bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 torchrun --master-port 1234 --nnodes=1 --nproc_per_node=4 train/train.py --task_name DATE-train_all --if_train True --if_val True --if_vis True --eval_every_iter 4000 --config-file train/configs/train_BRDF_all.yaml
```

See `train/configs/train_BRDF_all.yaml` for important params, and `train/utils/config/defaults.py` for all params. To override params, either modify the yaml config files, or append to the above command: e.g. to train only albedo estimation:

``` bash
CUDA_VISIBLE_DEVICES=0... .yaml DATA.data_read_list al MODEL_ALL.enable_list al
```

The full OpenRooms dataset is huge (~totaling 5TB including HDR images and lighting labels). The full dataset is available on local machines and clusters (check with Rui), or publicably downloable from [Zhengqin Li's project page](https://vilab-ucsd.github.io/ucsd-openrooms/).

Before you run any script, change the follwing paths in `train/utils/config/defaults.py`:

``` yaml
PATH.root_local # to your local project root
PATH.torch_home_local # for downloading temporary models from Torch model zoo
DATASET.dataset_path_local # to the full OpenRooms dataset path
```

### Quick start with mini dataset

For quick start on your local machine without downloading full dataset, you can try a mini subset of the dataset (downloadable at [DATASET.dataset_path_mini_local](https://drive.google.com/drive/folders/1-8RChRrXRO4F1HJv-UgaCucimihc9amy?usp=sharing) and [DATASET.png_path_local](https://drive.google.com/drive/folders/1otm31GBHdmTTsyjbzGRqOLU4eyBwJ63s?usp=sharing)). Change `DATASET.dataset_path_mini_local` and `DATASET.png_path_local` to the paths, where data should be organized as:

```
- {DATASET.dataset_path_mini_local} # unzip the first zip to e.g. /data/ruizhu/openrooms_mini; including labels of all modalities but not HDR RGB images
    - main_xml
        - scene0524_01
        - ...
    - main_xml1
    - mainDiffLight_xml
    - mainDiffMat_xml
    - mainDiffLight_xml1
    - mainDiffMat_xml1
- DATASET.png_path_local # unzip the second zip to /data/ruizhu/ORmini-pngs; include PNG RGB images
    - main_xml
        - scene0524_01
        - ...
    - main_xml1
    - mainDiffLight_xml
    - mainDiffMat_xml
    - mainDiffLight_xml1
    - mainDiffMat_xml1
```

``` bash
CUDA_VISIBLE_DEVICES=7 python train/train.py --task_name DATE-train_tmp_mini --if_train True --if_val True --if_vis True --eval_every_iter 4000 --config-file train/configs/train_BRDF_all.yaml DATA.if_load_png_not_hdr True DATASET.png_path_local /data/ruizhu/ORmini-pngs DATASET.mini True DATA.data_read_list al MODEL_ALL.enable_list al
```

## Train lighting estimation

[TODO]

## Pre-trained checkpoints and evaluation

Download checkpoints from [this folder](https://drive.google.com/drive/folders/18xWTUB_63GyzJ_SZrzV9cW2Dvy40rQPf?usp=share_link). Provided checkpoints include:

1. albedo-only model (6 layers) trained on full OR dataset: `20230517-224958--DATE-train_mm1_albedo`

    1.1 albedo-only model (6 layers) trained on full OR dataset, then finetuned on IIW: `20230522-001658--DATE-train_mm1_albedo_finetune_IIW_RESUME20230517-224958`

2. 4 modalities (albedo, roughness, depth, normals) (4 layers) trained on full OR dataset:

Place checkpoints at `Checkpoint/`. Launch one of the following commands; check the log outputs and Tensorboard for loss curves and visualizations at `logs/TASK_NAME` ([demo](https://i.imgur.com/K653psr.jpg)).

### Evaluate model 1. on IIW

[Demo results (without bilateral solver (BS))](https://i.imgur.com/Bddjq5q.jpg). WHDR=0.2645.

``` bash
iris_eval$ CUDA_VISIBLE_DEVICES=0 python train/train_iiw.py --task_name DATE-train_mm1_albedo_eval_IIW_RESUME20230517-224958 --if_train False --if_val True --if_vis True --eval_every_iter 4000 --config-file train/configs/train_albedo.yaml --resume 20230517-224958--DATE-train_mm1_albedo
```

### Evaluate model 1.1. on IIW

[Demo results (with bilateral solver (BS))](https://i.imgur.com/KFbF9cx.jpg). WHDR=0.1258.

``` bash
iris_eval$ CUDA_VISIBLE_DEVICES=0 python train/train_iiw.py --task_name DATE-train_mm1_albedo_finetuned_eval_BS_IIW_RESUME20230522-001658 --if_train False --if_val True --if_vis True --eval_every_iter 4000 --config-file train/configs/train_albedo.yaml --resume 20230522-001658--DATE-train_mm1_albedo_finetune_IIW_RESUME20230517-224958 MODEL_BRDF.if_bilateral True MODEL_BRDF.if_bilateral_albedo_only True
```

By default we use bilaterial solver (BS) to improve piece-wise smoothness and WHDR score, by appending `MODEL_BRDF.if_bilateral True MODEL_BRDF.if_bilateral_albedo_only True` to the above command. BS will lead to significantly slower inference.

## Training on Kubernates cluster of UCSD

[TODO]