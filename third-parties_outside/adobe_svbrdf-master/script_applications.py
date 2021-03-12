import os
import glob

import numpy as np
import torch

import sys
sys.path.insert(1, 'src/')
from util import *
import applications
import render

def save_images(all_outputs, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    img_idx = 0
    for output in all_outputs:
        all_images_np = output['image']  # shape [num_interps, H, W, C]

        renders = []
        for i in range(len(all_images_np)):
            image = all_images_np[i]  # shape [B, C, H, W]
            render = applications.render_map(image)
            renders.append(render)
        images_np = np.hstack(renders)
        images_pil = gyArray2PIL(images_np)
        images_pil.save(os.path.join(save_dir, str(img_idx).zfill(3) + ".png"))
        img_idx += 1


def interpolate_random_textures(num_maps, num_interps, search_type, interp_type, latent_space_type, save_dir):
    applications.interpolate_random_textures(num_maps=num_maps,
                                             num_interps=num_interps,
                                             search_type=search_type,
                                             save_dir=save_dir,
                                             interp_type=interp_type,
                                             latent_space_type=latent_space_type)


def interpolate_projected_textures(data_glob, search_type, save_dir, num_interps, num_maps=None):

    latent_paths = glob.glob(os.path.join(data_glob, "optim_latent.pt"))
    noises_paths = glob.glob(os.path.join(data_glob, "optim_noise.pt"))
    image_paths = [os.path.dirname(x)+"/tex.png" for x in noises_paths]

    print("Num latents: {} | Num noises: {}".format(len(latent_paths), len(noises_paths)))

    keep_list = ["carpet", "foam", "wood", "tile"]

    def filter_list(ls1, ls2, ls3):
        new_ls1, new_ls2, new_ls3 = [], [], []
        for x in keep_list:
            for l1, l2, l3 in zip(ls1, ls2, ls3):
                if str(l1).lower().find(x) >= 0 and \
                   str(l2).lower().find(x) >= 0 and \
                   str(l3).lower().find(x) >= 0:
                    new_ls1.append(l1)
                    new_ls2.append(l2)
                    new_ls3.append(l3)
        return new_ls1, new_ls2, new_ls3

    latent_paths, noises_paths, image_paths = filter_list(latent_paths, noises_paths, image_paths)

    print("Num latents: {} | Num noises: {} | Num images: {}".format(len(latent_paths), len(noises_paths), len(image_paths)))

    applications.interpolate_projected_textures(latent_paths=latent_paths,
                                                noises_paths=noises_paths,
                                                image_paths =image_paths,
                                                search_type=search_type,
                                                save_dir=save_dir,
                                                num_interps=num_interps,
                                                num_maps=num_maps)

'''

interpolate_random_textures(num_maps=500,
                            num_interps=5,
                            search_type="random",
                            interp_type="lerp",
                            save_dir="results/fake2fake3",
                            latent_space_type="w")





interpolate_projected_textures(data_glob="data/ours7_picked/*",
                               search_type="random",
                               num_interps=5,
                               save_dir="results/projected2fake",
                               num_maps=5)


interpolate_projected_textures(data_glob="data/ours7_picked/fake*",
                               search_type="brute",
                               num_interps=5,
                               save_dir="results/projected2projected/fake2")
'''
interpolate_projected_textures(data_glob="data/ours7_picked/real*",
                               search_type="brute",
                               num_interps=5,
                               save_dir="results/projected2projected/real4")
