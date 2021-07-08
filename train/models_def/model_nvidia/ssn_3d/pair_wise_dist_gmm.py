import torch 

import socket

import pdist3dgmm
# if socket.gethostname() == 'NV':
#     import pdist3dgmm
# else:
#     from torch.utils.cpp_extension import load_inline
#     from .pair_wise_dist_3dgmm import source
#     print("compile cuda source of 'pair_wise_dist_3dgmm' function...")
#     pdist3dgmm = load_inline("pair_wise_dist_3dgmm", cpp_sources="", cuda_sources=source)
#     print("done")

class PairwiseDistFunction(torch.autograd.Function):
# torch::Tensor forward_cuda(
#     const torch::Tensor pixel_mu,
#     const torch::Tensor spixel_mu,
#     const torch::Tensor spixel_invcov,
#     const torch::Tensor cmix,
#     const torch::Tensor spixel_indices,
#     torch::Tensor dist_matrix,
#     int num_spixels_w, int num_spixels_h
    @staticmethod
    def forward(self, pts, spixel_mu,  spixel_invcov, cmix,
                init_spixel_indices, 
                num_spixels_width, 
                num_spixels_height):
        '''
        pts - Bx3xN
        spixel_mu - Bx3xJ
        spixel_invcov - Bx9xJ
        cmix - Bx1xJ
        init_spixel_indices - 1xN
        '''

        self.num_spixels_width  = num_spixels_width
        self.num_spixels_height = num_spixels_height
        output = pts.new(pts.shape[0], 9, pts.shape[-1]).zero_() 
        self.save_for_backward(pts, spixel_mu, spixel_invcov,
                               cmix, init_spixel_indices)  # TODO: is this better than self.xxx ?

        return pdist3dgmm.forward(
            pts, spixel_mu, spixel_invcov, cmix, 
            init_spixel_indices, output, self.num_spixels_width, self.num_spixels_height)

        # return pdist3dgmm.forward(
        #     pts.contiguous(), 
        #     spixel_mu.contiguous(),
        #     spixel_invcov.contiguous(),
        #     cmix.contiguous(),
        #     init_spixel_indices.contiguous(), 
        #     output, self.num_spixels_width, self.num_spixels_height)

    @staticmethod
    def backward(self, dist_matrix_grad):
        pts, spixel_mu, spixel_invcov, cmix, init_spixel_indices = self.saved_tensors

        pts_grad = torch.zeros_like(pts)
        spixel_mu_grad = torch.zeros_like(spixel_mu)
        spixel_invcov_grad = torch.zeros_like(spixel_invcov)
        cmix_grad = torch.zeros_like(cmix) 

        # pts_grad, spixel_mu_grad, spixel_invcov_grad, cmix_grad = pdist3dgmm.backward(
        #     dist_matrix_grad.contiguous(), 
        #     pts.contiguous(),
        #     spixel_mu.contiguous(), 
        #     spixel_invcov.contiguous(),
        #     cmix.contiguous(),
        #     init_spixel_indices.contiguous(),
        #     pts_grad, 
        #     spixel_mu_grad,
        #     spixel_invcov_grad,
        #     cmix_grad,
        #     self.num_spixels_width, self.num_spixels_height)

        pts_grad, spixel_mu_grad, spixel_invcov_grad, cmix_grad = pdist3dgmm.backward(
            dist_matrix_grad, pts, spixel_mu, spixel_invcov, cmix, init_spixel_indices,
            pts_grad, spixel_mu_grad, spixel_invcov_grad, cmix_grad,
            self.num_spixels_width, self.num_spixels_height)

        return pts_grad, spixel_mu_grad, spixel_invcov_grad, cmix_grad, None, None, None
