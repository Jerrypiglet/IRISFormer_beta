import numpy as np
import matplotlib.pyplot as plt
import colorsys
from PIL import Image

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def vis_index_map(index_map):
    """
    input: [H, W], np.uint8, with indexs from [0, 1, 2, 3, ...] where 0 is no object
    return: [H, W], np.float32, RGB ~ [0., 1.]
    """
    num_colors = np.amax(index_map)
    colors = _get_colors(num_colors)
    index_map_vis = np.zeros((index_map.shape[0], index_map.shape[1], 3))
    for color_idx, color in enumerate(colors):
        mask = index_map == color_idx
        index_map_vis[mask] = color
    return index_map_vis

def reindex_output_map(index_map, invalid_index):
    index_map_reindex = np.zeros_like(index_map)
    index_map_reindex[index_map==invalid_index] = 0

    for new_index, index in enumerate(list(np.unique(index_map[index_map!=invalid_index]))):
        index_map_reindex[index_map==index] = new_index + 1

    return index_map_reindex

def vis_disp_colormap(disp_array, file=None, normalize=True):
    # disp_array = cv2.applyColorMap(disp_array, cv2.COLORMAP_JET)
    # disp_array = cv2.applyColorMap(disp_array, get_mpl_colormap('jet'))
    cm = plt.get_cmap('jet')
    # disp_array = disp_array[:, :, :3]
    if normalize:
        disp_array = disp_array/(1e-6+np.amax(disp_array))
    else:
        disp_array = np.clip(disp_array, 0., 1.)
    disp_array = (cm(disp_array)[:, :, :3] * 255).astype(np.uint8)
    
    # print('+++++', np.amax(disp_array), np.amin(disp_array))
    if file is not None:
        from PIL import Image, ImageFont, ImageDraw
        disp_Image = Image.fromarray(disp_array)
        disp_Image.save(file)
    else:
        return disp_array

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color