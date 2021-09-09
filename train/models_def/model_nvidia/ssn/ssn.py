import torch
from .pair_wise_distance import PairwiseDistFunction

# for debugging
from matplotlib.pyplot import * 
# from mutils.misc import mimshow, msavefig 
import torch.nn.functional as F

def calc_init_centroid(images, num_spixels_width, num_spixels_height):
    """
    calculate initial superpixels

    Args:
        images: torch.Tensor
            A Tensor of shape (B, C, H, W)
        spixels_width: int
            initial superpixel width
        spixels_height: int
            initial superpixel height

    Return:
        centroids: torch.Tensor
            A Tensor of shape (B, C, H_spix * W_spix)
        init_label_map: torch.Tensor
            A Tensor of shape (B, H * W)
        num_spixels_width: int
            A number of superpixels in each column
        num_spixels_height: int
            A number of superpixels int each raw
    """
    batchsize, channels, height, width = images.shape
    device = images.device

    centroids = torch.nn.functional.adaptive_avg_pool2d(images, (num_spixels_height, num_spixels_width))

    with torch.no_grad():
        num_spixels = num_spixels_width * num_spixels_height
        labels = torch.arange(num_spixels, device=device).reshape(1, 1, *centroids.shape[-2:]).type_as(centroids)
        init_label_map = torch.nn.functional.interpolate(labels, size=(height, width), mode="nearest")
        # print(init_label_map)
        init_label_map = init_label_map.repeat(batchsize, 1, 1, 1)

    init_label_map = init_label_map.reshape(batchsize, -1)
    centroids = centroids.reshape(batchsize, channels, -1)

    return centroids, init_label_map


@torch.no_grad()
def get_abs_indices(init_label_map, num_spixels_width):
    b, n_pixel = init_label_map.shape
    device = init_label_map.device
    r = torch.arange(-1, 2.0, device=device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)

    abs_pix_indices = torch.arange(n_pixel, device=device)[None, None].repeat(b, 9, 1).reshape(-1).long()
    abs_spix_indices = (init_label_map[:, None] + relative_spix_indices[None, :, None]).reshape(-1).long()
    abs_batch_indices = torch.arange(b, device=device)[:, None, None].repeat(1, 9, n_pixel).reshape(-1).long()

    return torch.stack([abs_batch_indices, abs_spix_indices, abs_pix_indices], 0)


@torch.no_grad()
def get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width):
    relative_label = affinity_matrix.max(1)[1]
    r = torch.arange(-1, 2.0, device=affinity_matrix.device)
    relative_spix_indices = torch.cat([r - num_spixels_width, r, r + num_spixels_width], 0)
    label = init_label_map + relative_spix_indices[relative_label]
    return label.long()


def ssn_iter(pixel_features,  n_iter,
             num_spixels_width, num_spixels_height, 
             mask_pixel=None, 
             index_add=False):
    """
    computing assignment iterations
    detailed process is in Algorithm 1, line 2 - 6

    Args:
        pixel_features: torch.Tensor
            A Tensor of shape (B, C, H, W)
        n_iter: int
            A number of iterations
        num_spixels_width, num_pixels_height: int
            number of pixels in the width and height directions

    """
    height, width = pixel_features.shape[-2:]
    batch_size = pixel_features.shape[0]

    # num_spixels_width  = int(math.sqrt(num_spixels * width / height))
    # num_spixels_height = int(math.sqrt(num_spixels * height / width))
    num_spixels = num_spixels_width * num_spixels_height

    spixel_features, init_label_map = calc_init_centroid(pixel_features, num_spixels_width, num_spixels_height)
    # print(init_label_map.shape, print(init_label_map))

    if not index_add:
        abs_indices = get_abs_indices(init_label_map, num_spixels_width) 
        mask = (abs_indices[1] >= 0) * (abs_indices[1] < num_spixels)
    else:
        abs_indices = get_abs_indices(init_label_map[0].unsqueeze(0), num_spixels_width) 

    pixel_features = pixel_features.reshape(*pixel_features.shape[:2], -1)
    permuted_pixel_features = pixel_features.permute(0, 2, 1).contiguous()

    if mask_pixel is None:
        mask_pixel = torch.ones((batch_size, height, width), dtype=torch.float32, device=spixel_features.device)
    mask_pixel_flattened = mask_pixel.reshape(batch_size, 1, -1)


    # import time
    # st = time.time()
    for _ in range(n_iter):
        # E-step: calculate /gammas
    # def forward(self, pixel_features, spixel_features, init_spixel_indices, num_spixels_width, num_spixels_height):
        
        #debug backward()
        # from torch.autograd import gradcheck
        # init_label_map_resize = F.interpolate(init_label_map.reshape(1, 1, 256, 384), size=(num_spixels_height*2, num_spixels_width*2))
        # init_label_map_resize = init_label_map_resize.reshape(1,-1)
        # test = gradcheck(PairwiseDistFunction.apply,
        #                  (pixel_features[:, :3,  :init_label_map_resize.shape[-1]].double(),
        #                   spixel_features[:, :3, :].double(),
        #                   init_label_map_resize.double(), 
        #                   num_spixels_width, num_spixels_height))
        # import ipdb; ipdb.set_trace()
        ###

        dist_matrix = PairwiseDistFunction.apply(
            pixel_features, 
            spixel_features, 
            init_label_map, 
            num_spixels_width, num_spixels_height) 

        # print(dist_matrix.shape, mask_pixel_flattened.shape, mask.shape)

        dist_matrix_masked_dense = dist_matrix * mask_pixel_flattened + torch.ones_like(dist_matrix) * 1e8 * (1. - mask_pixel_flattened)
        # print((dist_matrix * mask_pixel_flattened).shape, torch.max(dist_matrix * mask_pixel_flattened), torch.min(dist_matrix * mask_pixel_flattened), torch.median(dist_matrix * mask_pixel_flattened), '---')

        affinity_matrix = (-dist_matrix_masked_dense).softmax(1)
        reshaped_affinity_matrix = affinity_matrix.reshape(-1) 

        # print(dist_matrix.shape, mask_pixel_flattened.shape) # torch.Size([1, 9, 81920]) torch.Size([1, 1, 81920])
        # affinity_matrix_normalized_by_pixels = (-dist_matrix).softmax(2) * mask_pixel_flattened # torch.Size([1, 48, 76800])

        # M-step: update the centroids

        if not index_add:
            #sparse to dense
            sparse_abs_affinity = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_affinity_matrix[mask]) #Bx9xN
            abs_affinity = sparse_abs_affinity.to_dense().contiguous() #BxJxN
            spixel_features = torch.bmm(abs_affinity, permuted_pixel_features) / (abs_affinity.sum(2, keepdim=True) + 1e-16) #BxJxC
            spixel_features = spixel_features.permute(0, 2, 1).contiguous()#BxCxJ

        else:
            #use the index_add function to collect the spixel features, so we don't have to use the sparse_to_dense function
            spixel_features_w= (affinity_matrix).unsqueeze(-1) * permuted_pixel_features.unsqueeze(1) # Bx9xNxC
            # print(spixel_features_w.shape, affinity_matrix.shape, permuted_pixel_features.shape) # torch.Size([1, 9, 81920, 4]) torch.Size([1, 9, 81920]) torch.Size([1, 81920, 4])
            spixel_features_w = spixel_features_w.reshape(spixel_features_w.shape[0], -1, spixel_features_w.shape[-1]).contiguous()
            index_abs2rel =  abs_indices[1, :].reshape(9, height, width).contiguous() #9xHxW
            
            out_bound_mask_ = torch.logical_or(index_abs2rel>=num_spixels, index_abs2rel<=0,)
            index_abs2rel[out_bound_mask_]= \
                    index_abs2rel[4].unsqueeze(0).expand(
                            9, 
                            index_abs2rel.shape[1], 
                            index_abs2rel.shape[2])[out_bound_mask_]

            spixel_features= torch.zeros( 
                spixel_features_w.shape[0], 
                num_spixels, 
                spixel_features_w.shape[-1], 
                device=spixel_features_w.device).float() #BxJxC

            spixel_features.index_add_(dim=1, index=index_abs2rel.clone().reshape(-1), source=spixel_features_w)
            spixel_features = spixel_features.permute(0,2,1).contiguous() # BxCxJ

            affinity_matrix_sum = torch.zeros( spixel_features_w.shape[0], num_spixels, device=spixel_features_w.device).float() #BxJ
            affinity_matrix_sum.index_add_(dim=1, index=index_abs2rel.clone().reshape(-1), source=affinity_matrix.reshape(spixel_features_w.shape[0], -1)) #BxJ
            spixel_features= spixel_features / affinity_matrix_sum.unsqueeze(1) #BxCxJ
            #TODO: get gamma matrix


    # print(f'ssn_2d iteration took : {(time.time()-st)*1000:03f} ms for {n_iter:03d} iters') 

    # hard_labels = get_hard_abs_labels(affinity_matrix, init_label_map, num_spixels_width)
    # import ipdb; ipdb.set_trace()

    spixel_pixel_mul = None

    if not index_add:
        #abs_affinity: B x J x N
        # assert False
        # print(dist_matrix.shape, torch.max(dist_matrix), torch.min(dist_matrix), torch.median(dist_matrix))
        reshaped_dist_matrix_masked_dense = dist_matrix.reshape(-1)
        # print(reshaped_dist_matrix_masked_dense.shape)
        # print(reshaped_dist_matrix_masked_dense.shape, torch.max(reshaped_dist_matrix_masked_dense), torch.min(reshaped_dist_matrix_masked_dense), torch.median(reshaped_dist_matrix_masked_dense))
        sparse_dist_matrix = torch.sparse_coo_tensor(abs_indices[:, mask], reshaped_dist_matrix_masked_dense[mask]) #Bx9xN
        sparse_dist_matrix_mask = torch.sparse_coo_tensor(abs_indices[:, mask], torch.ones_like(reshaped_dist_matrix_masked_dense[mask])) #Bx9xN
        # print(abs_indices[:, mask].shape, reshaped_dist_matrix_masked_dense[mask].shape, sparse_dist_matrix.shape, '-====')
        dist_matrix_sparse = sparse_dist_matrix.to_dense().contiguous() #BxJxN
        dist_matrix_sparse_mask = sparse_dist_matrix_mask.to_dense().contiguous() #BxJxN
        dist_matrix_sparse = dist_matrix_sparse * dist_matrix_sparse_mask + torch.ones_like(dist_matrix_sparse_mask) * 1e8 * (1. - dist_matrix_sparse_mask)
        # print(dist_matrix_sparse.shape, torch.max(dist_matrix_sparse), torch.min(dist_matrix_sparse), torch.median(dist_matrix_sparse), mask_pixel_flattened.shape)
        dist_matrix_sparse = dist_matrix_sparse * mask_pixel_flattened + torch.ones_like(dist_matrix_sparse) * 1e8 * (1. - mask_pixel_flattened)
        # print(dist_matrix_sparse.shape, torch.max(dist_matrix_sparse), torch.min(dist_matrix_sparse), torch.median(dist_matrix_sparse))

        abs_affinity_normalized_by_pixels = (-dist_matrix_sparse).softmax(2) * mask_pixel_flattened # torch.Size([1, 48, 76800])\

        return abs_affinity, abs_affinity_normalized_by_pixels, dist_matrix_masked_dense, spixel_features, spixel_pixel_mul, None
    else:
        #abs_affinity:    B x J x N
        #affinity_matrix: B x 9 x N
        #index_abs2rel:   9 x H x W
        abs_affinity = torch.zeros(spixel_features_w.shape[0], num_spixels, height*width, device=spixel_features_w.device) # BxJxN
        abs_affinity = msparse2dense(abs_affinity, affinity_matrix, index_abs2rel)

        dist_matrix_sparse = torch.zeros(spixel_features_w.shape[0], num_spixels, height*width, device=spixel_features_w.device) + 1e8 # BxJxN
        dist_matrix_sparse = msparse2dense(dist_matrix_sparse, dist_matrix, index_abs2rel)
        abs_affinity_normalized_by_pixels = (-dist_matrix_sparse).softmax(2) * mask_pixel_flattened # torch.Size([1, 48, 76800])\

        return abs_affinity, abs_affinity_normalized_by_pixels, dist_matrix_masked_dense, spixel_features, spixel_pixel_mul, index_abs2rel

def msparse2dense(
    abs_affinity,    #BxJxN
    affinity_matrix, #Bx9xN
    index_abs2rel,   #9xHxW
    ):

    B, J, N = abs_affinity.shape[0], abs_affinity.shape[1], abs_affinity.shape[2]
    abs_affinity = abs_affinity.permute(0,2,1).reshape(-1) # vec(abs_affinity), BNJ
    offset= torch.arange(0,B*N, device=affinity_matrix.device).reshape(B*N, 1)*J
    offseted_abs2rel = index_abs2rel.permute(1,2,0).reshape(-1,9).repeat([B,1]) +  offset #(BN x 9)
    abs_affinity[offseted_abs2rel.reshape(-1)]= affinity_matrix.permute(0,2,1).reshape(-1) #BNJ vec
    abs_affinity= abs_affinity.reshape(B,N,J).permute(0,2,1) #BxJxN
    return abs_affinity
