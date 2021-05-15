import xml.etree.ElementTree as et
import numpy as np
from pathlib import Path
from utils.utils_total3D.utils_OR_mesh import loadMesh, computeBox, computeTransform, writeMesh
import copy
import random, string

def get_XML_root(main_xml_file):
    tree = et.parse(str(main_xml_file)) # L202 of sampleCameraPoseFromScanNet.py
    root  = tree.getroot()
    return root

def parse_XML_for_intrinsics(root):
    sensors = root.findall('sensor')
    assert len(sensors)==1
    sensor = sensors[0]

    film = sensor.findall('film')[0]
    integers = film.findall('integer')
    for integer in integers:
        if integer.get('name' ) == 'width':
            width = int(integer.get('value'))
        if integer.get('name' ) == 'height':
            height = int(integer.get('value'))
    fov_entry = sensor.findall('float')[0]
    assert fov_entry.get('name') == 'fov'
    fov = float(fov_entry.get('value'))
    f_px = width / 2. / np.tan(fov / 180. * np.pi / 2.)
    cam_K = np.array([[-f_px, 0., width/2.], [0., -f_px, height/2.], [0., 0., 1.]])
    return cam_K, {'fov': fov, 'f_px': f_px, 'width': width, 'height': height}


def parse_XML_for_shapes(root, root_uv_mapped, if_return_emitters=False):
    shapes = root.findall('shape')
    shapes_list = []
    if if_return_emitters:
        emitters_list = []

        # get envmap emitter(s)
        emitters_env = root.findall('emitter')
        assert len(emitters_env) == 1
        max_envmap_scale = -np.inf
        for emitter_env in emitters_env:
            assert emitter_env.get('type') == 'envmap'
            emitter_dict = {'if_emitter': True}
            emitter_dict['emitter_prop'] = {'emitter_type': 'envmap'}
            emitter_dict['emitter_prop']['if_obj'] = False
            assert len(emitter_env.findall('string')) == 1
            assert emitter_env.findall('string')[0].get('name') == 'filename'
            emitter_dict['emitter_prop']['emitter_filename'] = emitter_env.findall('string')[0].get('value')
            emitter_env.findall('float')[0].get('name') == 'scale'
            emitter_dict['emitter_prop']['emitter_scale'] = float(emitter_env.findall('float')[0].get('value'))
            max_envmap_scale = max(max_envmap_scale, emitter_dict['emitter_prop']['emitter_scale'])
            emitters_list.append(emitter_dict)
        if_bright_outside = max_envmap_scale > 1e-3

    for shape in shapes:
        shape_dict = {'id': shape.get('id'), 'id_random': ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))}

        shape_dict[shape.findall('string')[0].get('name')] = shape.findall('string')[0].get('value')
        
        assert len(shape.findall('transform')) == 1
        transforms = shape.findall('transform')[0]
        assert transforms.get('name') == 'toWorld'
        transforms_list = []
        for transform in transforms:
            transform_name = transform.tag
            assert transform_name in ['scale', 'rotate', 'translate']
            transform_dict = {transform_name: {key: float(transform.get(key)) for key in transform.keys()}}
            transforms_list.append(transform_dict)
        shape_dict['transforms_list'] = transforms_list
        shape_dict['if_correct_path'] = False

        # find emitter property: get objs emitter(s)
        emitters = shape.findall('emitter')
        shape_dict['emitter_prop'] = {}
        if 'window' in shape_dict['filename'] and if_bright_outside: # window that has light shining through
            shape_dict['if_emitter'] = True
            shape_dict['emitter_prop']['if_obj'] = True
            emitter_dict['emitter_prop']['combined_filename'] = str(root_uv_mapped / shape_dict['filename'].replace('../../../../../uv_mapped/', ''))
            emitters_list.append(copy.deepcopy(shape_dict))
        elif len(emitters) == 0: # not emitter
            shape_dict['if_emitter'] = False
        else: # lamps, ceilimg_lamps
            shape_dict['if_emitter'] = True
            shape_dict['emitter_prop']['if_obj'] = True
            assert len(emitters) == 1
            emitter = emitters[0]
            shape_dict['emitter_prop']['emitter_type'] = emitter.get('type')
            assert len(emitter.findall('rgb')) == 1
            shape_dict['emitter_prop']['emitter_rgb_float'] = [float(x) for x in emitter.findall('rgb')[0].get('value').split(' ')]

            emitters_list.append(copy.deepcopy(shape_dict))
        
        shapes_list.append(shape_dict)

    # post-process to merge lamp light and lamp base
    for idx, emitter_dict in enumerate(emitters_list):
        # print(shape_dict['id'], shape_dict['filename'])
        if not emitter_dict['emitter_prop']['if_obj']:
            continue
        emitter_filename_abs = root_uv_mapped / emitter_dict['filename'].replace('../../../../../uv_mapped/', '')
        if 'aligned_light.obj' not in emitter_dict['filename']:
            emitter_dict['emitter_prop']['combined_filename'] = str(emitter_filename_abs)
            continue
        for shape_dict in shapes_list:
            # print(shape_dict['filename'])
            if emitter_dict['filename'].replace('aligned_light.obj', 'aligned_shape.obj') == shape_dict['filename']:
                combined_filename_abs = Path(str(emitter_filename_abs).replace('aligned_light.obj', 'alignedNew.obj'))
                if not combined_filename_abs.exists():
                    other_part_filename_abs = Path(str(emitter_filename_abs).replace('aligned_light.obj', 'aligned_shape.obj'))
                    assert other_part_filename_abs.exists()
                    vertices_0, faces_0 = loadMesh(str(emitter_filename_abs))
                    vertices_1, faces_1 = loadMesh(str(other_part_filename_abs))
                    vertices_combine = np.vstack([vertices_0, vertices_1])
                    faces_combine = np.vstack([faces_0, faces_1+vertices_0.shape[0]])
                    writeMesh(str(combined_filename_abs), vertices_combine, faces_combine)
                    print('NEW mesh written to %s'%str(combined_filename_abs))
                emitter_dict['emitter_prop']['combined_filename'] = str(combined_filename_abs)
    
    if if_return_emitters:
        return shapes_list, emitters_list
    else:
        return shapes_list