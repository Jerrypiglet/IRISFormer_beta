import numpy as np
import colorsys

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


