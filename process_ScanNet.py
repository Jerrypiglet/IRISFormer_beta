import glob
import sys
import os.path as osp
import numpy as np
import pickle
import shutil 
import os
from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, '/home/ruizhu/Documents/Projects/Total3DUnderstanding/utils_OR/DatasetCreation/')
from sampleCameraPoseFromScanNet import computeCameraEx

val_list_file = '/home/ruizhu/Documents/Projects/semanticInverse/train/data/openrooms/list_OR_tmp/list/val.txt'
list_read = open(val_list_file).readlines()
scene_list = []
for line in list_read:
    line = line.strip()
    line_split = line.split(' ')
    meta_split = line_split[2].split('/')[0]
    scene_name = line_split[2].split('/')[1]
    scene_list.append('/'.join([meta_split, scene_name]))

scene_list = list(set(scene_list))

print(len(scene_list))

# render_dest = Path('/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val')
render_dest = Path('/home/ruizhu/Documents/Projects/semanticInverse/dataset/openrooms_sequence_val_notSkipFrames')
xml_dest = render_dest / 'scenes'
xml_ori = Path('/newfoundland2/ruizhu/siggraphasia20dataset/code/Routine/scenes')
ScanNet_RAW_root = Path('/newfoundland2/ruizhu/scannet')

cmd_file = '/home/ruizhu/Downloads/tmp_cmds.pickle'
cmd_list = []

# scene_idx_select = scene_list.index('mainDiffLight_xml1/scene0509_00')
# # scene_idx = 0
# for scene_idx in range(len(scene_list)):
#     if scene_idx != scene_idx_select:
#         continue
#     meta_split, scene_name = scene_list[scene_idx].split('/')

# ScanNet_root = Path('/newfoundland2/ruizhu/scannet/labels_2d_240x320/')
ScanNet_root = ScanNet_RAW_root / 'labels_2d_240x320_notSkipFrames_RE'

def process_scannet_scenes(scene_name):
    if scene_name != 'scene0377_00':
        return
    
    ScanNet_scene_path = ScanNet_root / scene_name
    if not ScanNet_scene_path.exists():
        ScanNet_scene_path.mkdir(exist_ok=True, parents=True)
        # dump ScanNet labels
        cmd = 'cd /home/ruizhu/Documents/Projects/ENet-ScanNet/prepare_data && python reader.py --filename /newfoundland2/ruizhu/scannet/scans/%s/%s.sens --output_path %s --export_poses --export_depth_images --export_intrinsics --export_color_images'%\
            (scene_name, scene_name, str(ScanNet_scene_path))
        print(cmd)
        os.system(cmd)
        
    # continue
    
def generate_dest_poses(meta_split_scene_name):
    meta_split, scene_name = meta_split_scene_name.split('/')

    ScanNet_pose_path = ScanNet_root  / scene_name / 'pose'
    render_dest_scene = render_dest / meta_split / scene_name
#     if render_dest_scene.exists():
#         continue
    render_dest_scene.mkdir(exist_ok=True, parents=True)

    # /home/ruizhu/Documents/Projects/Total3DUnderstanding/utils_OR/DatasetCreation/sampleCameraPoseFromScanNet.py

    # Load transformation file
    xml_ori_scene = xml_ori / meta_split.split('_')[1] / scene_name
    xml_dest_scene = xml_dest / meta_split.split('_')[1] / scene_name
#     assert xml_dest_scene.exists() == False
    if xml_dest_scene.exists() == False:
        shutil.copytree(str(xml_ori_scene), str(xml_dest_scene)) 
            
    transformFile = str(xml_dest_scene / 'transform.dat')
    with open(transformFile, 'rb') as fIn:
        transforms = pickle.load(fIn)

    # Generate cam.txt            
    poseDir = str(ScanNet_pose_path)
    poseNum = len(glob.glob(osp.join(poseDir, '*.txt') ) )
    isSelected = np.zeros(poseNum, dtype=np.int32 )
    camGap = 20

    for n in range(0, poseNum, camGap ):
        isSelected[n] = 1

    camPoses= []
    scannetFramePaths = []
    for n in range(0, 10000, camGap ):

        poseFile = osp.join(poseDir, '%d.txt' % n)
        if not osp.isfile(poseFile ):
            print('ScanNet pose file not found at %s'%poseFile)
            break

        camMat = np.zeros((4, 4), dtype=np.float32 )

        isValidCam = True
        with open(poseFile, 'r') as camIn:
            for n in range(0, 4):
                camLine = camIn.readline().strip()
                if camLine.find('inf') != -1 or camLine.find('Inf') != -1:
                    print(camLine, poseFile)
                    isValidCam = False
                    break

                camLine  = [float(x) for x in camLine.split(' ') ]
                for m in range(0, 4):
                    camMat[n, m] = camLine[m]

        if isValidCam == False:
            continue
            while not isValidCam:
                camMat = np.zeros((4,4), dtype=np.float32 )
                while True:
                    camId = np.random.randint(0, poseNum )
                    if isSelected[camId ] == 0:
                        break
                poseFile = osp.join(poseDir, '%d.txt' % camId )
                isValidCam = True
                with open(poseFile, 'r') as camIn:
                    for n in range(0, 4):
                        camLine = camIn.readline().strip()
                        if camLine.find('inf') != -1 or camLine.find('Inf') != -1:
                            isValidCam = False
                            break
                        camLine  = [float(x) for x in camLine.split(' ') ]

                        for m in range(0, 4):
                            camMat[n, m] = camLine[m]

            rot = camMat[0:3, 0:3]
            trans = camMat[0:3, 3]

            origin, lookat, up = computeCameraEx(rot, trans,
                    transforms[0][0][1], transforms[0][1][1], transforms[0][2][1] )
            isSelected[camId ] = 1

            origin = origin.reshape(1, 3 )
            lookat = lookat.reshape(1, 3 )
            up = up.reshape(1, 3 )
            camPose = np.concatenate([origin, lookat, up ], axis=0 )
            camPoses.append(camPose )
        else:
            rot = camMat[0:3, 0:3]
            trans = camMat[0:3, 3]

            origin, lookat, up = computeCameraEx(rot, trans,
                    transforms[0][0][1], transforms[0][1][1], transforms[0][2][1] )

            origin = origin.reshape(1, 3 )
            lookat = lookat.reshape(1, 3 )
            up = up.reshape(1, 3 )
            camPose = np.concatenate([origin, lookat, up ], axis=0 )
            camPoses.append(camPose )
            scannetFramePath = ScanNet_root  / scene_name / 'color' / ('%d.jpg'%n)
            scannetFramePaths.append(scannetFramePath)


    # Output the initial camera poses
    camNum = len(camPoses )
    xml_outDir = str(xml_dest_scene)
    with open(osp.join(xml_outDir, 'cam.txt'), 'w') as camOut:
        camOut.write('%d\n' % camNum )
        print('Final sampled camera poses: %d' % len(camPoses ) )
        print('===> writing cam.txt to %s'%osp.join(xml_outDir, 'cam.txt'))
        for camPose in camPoses:
            for n in range(0, 3):
                camOut.write('%.3f %.3f %.3f\n' % \
                        (camPose[n, 0], camPose[n, 1], camPose[n, 2] ) )

    with open(osp.join(xml_outDir, 'ScanNet_frame_paths.pkl'), 'wb') as file:
        pickle.dump(scannetFramePaths, file, protocol=pickle.HIGHEST_PROTOCOL)

scene_name_list = []
meta_split_scene_name_list = []
for line in list_read:
    line = line.strip()
    line_split = line.split(' ')
    meta_split = line_split[2].split('/')[0]
    scene_name = line_split[2].split('/')[1]
    scene_name_list.append(scene_name)
    meta_split_scene_name_list.append('/'.join([meta_split, scene_name]))

meta_split_scene_name_list = list(set(meta_split_scene_name_list))

processes = 16

print('===== Processing %d scenes for ScanNet...'%len(scene_name_list))
p = Pool(processes=processes)
avg_data = p.map(process_scannet_scenes, scene_name_list)
p.close()
p.join()

print('===== Processing %d pose files for dest scenes...'%len(meta_split_scene_name_list))
p = Pool(processes=processes)
avg_data = p.map(generate_dest_poses, meta_split_scene_name_list)
p.close()
p.join()