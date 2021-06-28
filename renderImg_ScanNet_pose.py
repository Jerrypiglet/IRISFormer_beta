import glob
import os
import os.path as osp
import argparse
import xml.etree.ElementTree as et


parser = argparse.ArgumentParser()
# Directories
parser.add_argument('--xmlRoot', default="/siggraphasia20dataset/code/Routine/scenes/xml1", help="outdir of xml file")
# Start and end point
parser.add_argument('--rs', default=320, type=int, help='the width of the image' )
parser.add_argument('--re', default=240, type=int, help='the height of the image' )
# xml file
parser.add_argument('--xmlFile', default='mainDiffLight', help='the xml file')
parser.add_argument('--sceneName', default='', help='if filter scenes')
# output file
parser.add_argument('--outRoot', default='/siggraphasia20dataset/code/Routine/DatasetCreation', help='output directory')
# Render Mode
parser.add_argument('--mode', default=0, type=int, help='the information being rendered')
# Control
parser.add_argument('--forceOutput', action='store_true', help='whether to overwrite previous results')
parser.add_argument('--medianFilter', action='store_true', help='whether to use median filter')
# Program
parser.add_argument('--program', default='/home/zhl/OptixRenderer/src/bin/optixRenderer', help='the location of render' )
opt = parser.parse_args()


scenes = glob.glob(osp.join(opt.xmlRoot, 'scene*') )
scenes = [x for x in scenes if osp.isdir(x) ]
scenes = sorted(scenes )

if opt.sceneName != '':
    scenes = [x for x in scenes if opt.sceneName in x]

# for n in range(opt.rs, min(opt.re, len(scenes ) ) ):
for n in range(len(scenes)):
    scene = scenes[n]
    sceneId = scene.split('/')[-1]

    print('%d/%d: %s' % (n, len(scenes), sceneId ) )

    outDir = osp.join(opt.outRoot, opt.xmlFile + '_' + opt.xmlRoot.split('/')[-1], sceneId )
    if not osp.isdir(outDir ):
        continue
        os.system('mkdir -p %s' % outDir )

    xmlFile = osp.join(scene, '%s.xml' % opt.xmlFile )
    camFile = osp.join(scene, 'cam.txt' )
    if not osp.isfile(xmlFile ) or not osp.isfile(camFile ):
        continue

    tree  = et.parse(xmlFile )
    root = tree.getroot()

    shapes = root.findall('shape')
    isFindAreaLight = False
    for shape in shapes:
        emitters = shape.findall('emitter')
        if len(emitters ) > 0:
            isFindAreaLight = True
            break


    # cmd = '%s -f %s -c %s -o %s -m %d --camStart 1 --camEnd 2' % (opt.program, xmlFile,
    #                                                              'cam.txt', osp.join(outDir, 'im.rgbe'), opt.mode )
    cmd = '%s -f %s -c %s -o %s -m %d --camEnd 100' % (opt.program, xmlFile, 'cam.txt', osp.join(outDir, 'im.rgbe'), opt.mode )

    if opt.forceOutput:
        cmd += ' --forceOutput'

    if opt.medianFilter:
        cmd += ' --medianFilter'

    if not isFindAreaLight:
        print('Warning: no area light found, may need more samples.' )
        cmd += ' --maxIteration 1'
    else:
        cmd += ' --maxIteration 1'

    print(cmd)
    os.system(cmd )
