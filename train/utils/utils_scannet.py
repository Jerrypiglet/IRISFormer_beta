import numpy as np
import PIL
import math
import utils.utils_nvidia.mdataloader.m_preprocess as m_preprocess
import torch

def read_ExtM_from_txt(fpath_txt):
    '''
    Read the external matrix from txt file. 
    The txt file for the exterminal matrix is got from 
    the sens file loader 

    return:
    ExtM - world to cam: p_c = ExtM @ p_w
           cam to world : p_w = ExM.inverse() @ p_c

    '''
    ExtM = np.eye(4)
    with open(fpath_txt, 'r') as f:
        content = f.readlines()
    content = [ x.strip() for x in content]
    
    for ir, row in enumerate(ExtM):
        row_content = content[ir].split()
        row = np.asarray([ float(x) for x in row_content ])
        ExtM[ir, :] = row
    ExtM = np.linalg.inv(ExtM)
    return ExtM

def convert_IntM_from_OR(cam_K, out_size=None):
    '''
    Read the intrinsic matrix from the txt file 
    The txt file for the exterminal matrix is got from 
    the sens file loader 
    input: out_size [width, height]
    cam_K: 3x3 cam_K: e.g. [[577.8708   0.     320.    ], [  0.     577.8708 240.    ], [  0.      0.       1.    ]]
    Output:
    cam_intrinsic - The cam_intrinsic structure, used in warping.homography:
        cam_intrinsic includes: {'hfov': hfov, 'vfov': vfov, 'unit_ray_array': unit_ray_array, 'intrinsic_M'} : 
        hfov, vfov
        fovs in horzontal and vertical directions (degrees)
        unit_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to the
        unit ray pointing from the camera center to the pixel
    '''
    # IntM = np.zeros((4,4))
    # with open(fpath_txt, 'r') as f:
    #     content = f.readlines()

    # contents = [ x.strip() for x in content]

    # assert contents[2].split('=')[0].strip() == 'm_colorWidth',\
    #         'un-recogonized _info.txt format '
    # width = int( contents[2].split('=')[1].strip())

    # assert contents[3].split('=')[0].strip() == 'm_colorHeight',\
    #         'un-recogonized _info.txt format '
    # height = int( contents[3].split('=')[1].strip())

    # assert contents[7].split('=')[0].strip() == 'm_calibrationColorIntrinsic',\
    #         'un-recogonized _info.txt format '

    # color_intrinsic_vec = contents[7].split('=')[1].strip().split()
    # color_intrinsic_vec = [float(x) for x in color_intrinsic_vec]
    # IntM = np.reshape(np.asarray(color_intrinsic_vec), (4,4))

    # print(out_size, IntM)
    # IntM = IntM[:3, :]

    IntM = cam_K # for size 240x320
    focal_length = np.mean([IntM[0,0], IntM[1,1]])
    h_fov = math.degrees(math.atan(IntM[0, 2] / IntM[0, 0]) * 2)
    v_fov = math.degrees(math.atan(IntM[1, 2] / IntM[1, 1]) * 2)

    width, height = int(IntM[0, 2]) * 2, int(IntM[1, 2]) * 2
    assert width==320 and height==240

    if out_size is not None: # the depth map is re-scaled #
        assert False
        camera_intrinsics = np.zeros((3,4))
        pixel_width, pixel_height = out_size[0], out_size[1]
        camera_intrinsics[2,2] = 1.
        camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(h_fov/2.0))
        camera_intrinsics[0,2] = pixel_width/2.0
        camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(v_fov/2.0))
        camera_intrinsics[1,2] = pixel_height/2.0
        IntM = camera_intrinsics
        focal_length = pixel_width / width * focal_length
        width, height = pixel_width, pixel_height

    # In scanenet dataset, the depth is perperdicular z, not ray distance #
    pixel_to_ray_array = normalised_pixel_to_ray_array(\
            width= width, height= height, hfov = h_fov, vfov = v_fov,
            normalize_z = True) 

    pixel_to_ray_array_2dM = np.reshape(np.transpose( pixel_to_ray_array, axes= [2,0,1] ), [3, -1])
    pixel_to_ray_array_2dM = torch.from_numpy(pixel_to_ray_array_2dM.astype(np.float32))

    cam_intrinsic = {\
            'hfov': h_fov,
            'vfov': v_fov,
            'unit_ray_array': pixel_to_ray_array,
            'unit_ray_array_2D': pixel_to_ray_array_2dM,
            'intrinsic_M_cuda': torch.from_numpy(IntM[:3,:3].astype(np.float32)),  
            'focal_length': focal_length,
            'intrinsic_M': IntM}  

    return cam_intrinsic 

def normalised_pixel_to_ray_array(width=320,height=240, hfov = 60, vfov = 45, normalize_z=True):
    '''
    Given the FOV (estimated from the intrinsic matrix for example), 
    get the unit ray vectors pointing from the camera center to the pixels

    Inputs: 
    width, height - The width and height of the image (in pixels)
    hfov, vfov - The field of views (in degree) in the horizontal and vertical directions
    normalize_z - 

    Outputs:
    pixel_to_ray_array - A tensor with size (height, width, 3). Each 'pixel' corresponds to 
                         the unit ray pointing from the camera center to the pixel
    '''
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            if normalize_z:
                # z=1 #
                pixel_to_ray_array[y,x] =np.array(\
                        pixel_to_ray( (x,y),
                        pixel_height=height,pixel_width=width,
                        hfov= hfov, vfov= vfov))
            else:
                # length=1 #
                pixel_to_ray_array[y,x] = normalize(np.array(\
                        pixel_to_ray( (x,y),
                        pixel_height=height,pixel_width=width,
                        hfov= hfov, vfov= vfov)))

    return pixel_to_ray_array

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    '''
    Inputs:
    pixel- pixel index (icol, irow)
    hfov, vfov - the field of views in the horizontal and vertical directions (in degrees)
    pixel_width, pixel_height - the # of pixels in the horizontal/vertical directions
    Outputs:
    (x, y, 1) - The coordinate location in 3D. The origin of the coordinate in 3D is the camera center.
    So given the depth d, the backprojected 3D point is simply: d * (x,y,1).
    Or if we have the ray distance d_ray, the backprojected 3D point is simply d_ray * (x,y,1) / norm_L2((x,y,z))
    '''
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ( (2.0 * ( (x+0.5)/pixel_width )  ) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ( (2.0 * ( (y+0.5)/pixel_height ) ) - 1.0)
    return (x_vect,y_vect,1.0) 

def normalize(v):
    return v/np.linalg.norm(v)

def CamIntrinsic_to_cuda(CamIntrinsic, device, members = ['intrinsic_M_cuda', 'unit_ray_array_2D']):
    '''put the necessary members in the CamIntrinsic structure to the specified device'''
    for mem in members:
        CamIntrinsic[mem] = CamIntrinsic[mem].to(device)

def read_img(path, img_size = None, no_process= False, only_resize = False):
    '''
    Read image and process 
    '''
    proc_img = m_preprocess.get_transform()
    if no_process:
        img = PIL.Image.open(path)
        width, height = img.size
    else:
        if img_size is not None:
            img = PIL.Image.open(path).convert('RGB')
            img = img.resize( img_size, PIL.Image.BICUBIC )
        else:
            img = PIL.Image.open(path).convert('RGB')
        width, height = img.size
        if not only_resize:
            img = proc_img(img)

    raw_sz = (width, height)
    return img,  raw_sz
