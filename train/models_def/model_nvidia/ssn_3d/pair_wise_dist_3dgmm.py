source='''
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 256

#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ pixel_mu,
    const scalar_t* __restrict__ spixel_mu,
    const scalar_t* __restrict__ spixel_invcov,
    const scalar_t* __restrict__ cmix,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ dist_matrix,
    int batchsize,  int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    //input sizes:
    // channels = 3
    // pixel_mu - Bx3xN_pix
    // spixel_mu - Bx3xN_spix
    // spixel_invcov - Bx9xN_spix
    // cmix - Bx1xN_spix. The log of the normalization constant multiplied with the mix parameter
    // spixel_indices - Bx1xN_pix
    //output size:
    // dist_matrix - Bx9xN_pix. log(\pi_j p(z_i | \theta_j)) for point i and superpixel j (j in {0, ..., 8})

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;

    int init_spix_index = spixel_indices[b * num_pixels + p];

    int x_index = init_spix_index % num_spixels_w; // col index
    int spixel_offset_x = (spixel_offset % 3 - 1);

    int y_index = init_spix_index / num_spixels_w; // row indx
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = -1e16;
    }
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) {
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = -1e16;
    }
    else {
        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y; 
        scalar_t sum_maha_dist = 0.;
        scalar_t dx = 0., dy = 0., dz = 0.;

        dx = pixel_mu[b * 3*num_pixels + 0 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 0 * num_spixels + query_spixel_index];
        dy = pixel_mu[b * 3*num_pixels + 1 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 1 * num_spixels + query_spixel_index];
        dz = pixel_mu[b * 3*num_pixels + 2 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 2 * num_spixels + query_spixel_index];

        sum_maha_dist +=  dx*dx * spixel_invcov[b*9*num_spixels+0*num_spixels+query_spixel_index] 
                        + dx*dy * spixel_invcov[b*9*num_spixels+1*num_spixels+query_spixel_index] 
                        + dx*dz * spixel_invcov[b*9*num_spixels+2*num_spixels+query_spixel_index] 
                        + dy*dx * spixel_invcov[b*9*num_spixels+3*num_spixels+query_spixel_index] 
                        + dy*dy * spixel_invcov[b*9*num_spixels+4*num_spixels+query_spixel_index] 
                        + dy*dz * spixel_invcov[b*9*num_spixels+5*num_spixels+query_spixel_index]
                        + dz*dx * spixel_invcov[b*9*num_spixels+6*num_spixels+query_spixel_index] 
                        + dz*dy * spixel_invcov[b*9*num_spixels+7*num_spixels+query_spixel_index] 
                        + dz*dz * spixel_invcov[b*9*num_spixels+8*num_spixels+query_spixel_index];

        sum_maha_dist *= -0.5; 
        sum_maha_dist += cmix[b*1*num_spixels + query_spixel_index];
        dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = sum_maha_dist;
    }
}

template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ dist_matrix_grad,
    const scalar_t* __restrict__ pixel_mu,
    const scalar_t* __restrict__ spixel_mu,
    const scalar_t* __restrict__ spixel_invcov,
    const scalar_t* __restrict__ cmix,
    const scalar_t* __restrict__ spixel_indices,
    scalar_t* __restrict__ pixel_mu_grad,
    scalar_t* __restrict__ spixel_mu_grad,
    scalar_t* __restrict__ spixel_invcov_grad,
    scalar_t* __restrict__ cmix_grad,
    int batchsize, int channels, int num_pixels, int num_spixels,
    int num_spixels_w, int num_spixels_h
){
    //input sizes:
    // pixel_mu - Bx3xN_pix
    // spixel_mu - Bx3xN_spix
    // spixel_invcov - Bx9xN_spix
    // cmix - Bx1xN_spix. The log of the normalization constant multiplied with the mix parameter
    // spixel_indices - Bx1xN_pix
    //output size:
    // dist_matrix - Bx9xN_pix. log(\pi_j p(z_i | \theta_j)) for point i and superpixel j (j in {0, ..., 8})


    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // if (index >= batchsize * num_pixels * 9) return;

    // int b = index % batchsize;
    // int spixel_offset = (index / batchsize) % 9;
    // int p = (index / (batchsize * 9)) % num_pixels;

    // int init_spix_index = spixel_indices[b * num_pixels + p];

    // int x_index = init_spix_index % num_spixels_w; // col index
    // int spixel_offset_x = (spixel_offset % 3 - 1);

    // int y_index = init_spix_index / num_spixels_w; // row indx
    // int spixel_offset_y = (spixel_offset / 3 - 1);

    // if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) {
    //     dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = -1e16;
    // }
    // else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) {
    //     dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = -1e16;
    // }


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batchsize * num_pixels * 9) return;

    int b = index % batchsize;
    int spixel_offset = (index / batchsize) % 9;
    int p = (index / (batchsize * 9)) % num_pixels;

    int init_spix_index = spixel_indices[b * num_pixels + p];
    int x_index = init_spix_index % num_spixels_w;
    int spixel_offset_x = (spixel_offset % 3 - 1);

    int y_index = init_spix_index / num_spixels_w;
    int spixel_offset_y = (spixel_offset / 3 - 1);

    if (x_index + spixel_offset_x < 0 || x_index + spixel_offset_x >= num_spixels_w) return;
    else if (y_index + spixel_offset_y < 0 || y_index + spixel_offset_y >= num_spixels_h) return;
    else {

        // dx = pixel_mu[b * 3*num_pixels + 0 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 0 * num_spixels + query_spixel_index];
        // dy = pixel_mu[b * 3*num_pixels + 1 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 1 * num_spixels + query_spixel_index];
        // dz = pixel_mu[b * 3*num_pixels + 2 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 2 * num_spixels + query_spixel_index];

        // sum_maha_dist +=  dx*dx * spixel_invcov[b*9*num_spixels+0*num_spixels+query_spixel_index] 
        //                 + dx*dy * spixel_invcov[b*9*num_spixels+1*num_spixels+query_spixel_index] 
        //                 + dx*dz * spixel_invcov[b*9*num_spixels+2*num_spixels+query_spixel_index] 
        //                 + dy*dx * spixel_invcov[b*9*num_spixels+3*num_spixels+query_spixel_index] 
        //                 + dy*dy * spixel_invcov[b*9*num_spixels+4*num_spixels+query_spixel_index] 
        //                 + dy*dz * spixel_invcov[b*9*num_spixels+5*num_spixels+query_spixel_index]
        //                 + dz*dx * spixel_invcov[b*9*num_spixels+6*num_spixels+query_spixel_index] 
        //                 + dz*dy * spixel_invcov[b*9*num_spixels+7*num_spixels+query_spixel_index] 
        //                 + dz*dz * spixel_invcov[b*9*num_spixels+8*num_spixels+query_spixel_index]; 
        // sum_maha_dist *= -0.5; 
        // sum_maha_dist += cmix[b*1*num_spixels + query_spixel_index];
        // dist_matrix[b * (9 * num_pixels) + spixel_offset * num_pixels + p] = sum_maha_dist;

        int query_spixel_index = init_spix_index + spixel_offset_x + num_spixels_w * spixel_offset_y;
        scalar_t dist_matrix_grad_val = dist_matrix_grad[b * (9 * num_pixels) + spixel_offset * num_pixels + p];
        scalar_t dx = 0., dy = 0., dz = 0., wdx=0., wdy=0., wdz=0., wtdx=0., wtdy=0., wtdz=0.;
        dx = pixel_mu[b * 3*num_pixels + 0 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 0 * num_spixels + query_spixel_index];
        dy = pixel_mu[b * 3*num_pixels + 1 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 1 * num_spixels + query_spixel_index];
        dz = pixel_mu[b * 3*num_pixels + 2 * num_pixels + p] - spixel_mu[b * 3*num_spixels + 2 * num_spixels + query_spixel_index]; 

        wdx =   dx * spixel_invcov[b*9*num_spixels+0*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+1*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+2*num_spixels+query_spixel_index];

        wdy =   dx * spixel_invcov[b*9*num_spixels+3*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+4*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+5*num_spixels+query_spixel_index];

        wdz =   dx * spixel_invcov[b*9*num_spixels+6*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+7*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+8*num_spixels+query_spixel_index];

        wtdx =   dx * spixel_invcov[b*9*num_spixels+0*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+3*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+6*num_spixels+query_spixel_index];

        wtdy =   dx * spixel_invcov[b*9*num_spixels+1*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+4*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+7*num_spixels+query_spixel_index];

        wtdz =   dx * spixel_invcov[b*9*num_spixels+2*num_spixels+query_spixel_index]  
              + dy * spixel_invcov[b*9*num_spixels+5*num_spixels+query_spixel_index] 
              + dz * spixel_invcov[b*9*num_spixels+8*num_spixels+query_spixel_index];

        // pixel_mu_grad
        atomicAdd(&pixel_mu_grad[b * 3*num_pixels + 0 * num_pixels + p], -.5 * (wdx + wtdx) *dist_matrix_grad_val);
        atomicAdd(&pixel_mu_grad[b * 3*num_pixels + 1 * num_pixels + p], -.5 * (wdy + wtdy) *dist_matrix_grad_val);
        atomicAdd(&pixel_mu_grad[b * 3*num_pixels + 2 * num_pixels + p], -.5 * (wdz + wtdz) *dist_matrix_grad_val);

        // spixel_mu_grad
        atomicAdd(&spixel_mu_grad[b * 3*num_spixels + 0 * num_spixels + query_spixel_index], wdx*dist_matrix_grad_val);
        atomicAdd(&spixel_mu_grad[b * 3*num_spixels + 1 * num_spixels + query_spixel_index], wdy*dist_matrix_grad_val);
        atomicAdd(&spixel_mu_grad[b * 3*num_spixels + 2 * num_spixels + query_spixel_index], wdz*dist_matrix_grad_val);

        // spixel_invcov_grad
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+0*num_spixels+query_spixel_index], -.5 * dx*dx*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+1*num_spixels+query_spixel_index], -.5 * dx*dy*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+2*num_spixels+query_spixel_index], -.5 * dx*dz*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+3*num_spixels+query_spixel_index], -.5 * dy*dx*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+4*num_spixels+query_spixel_index], -.5 * dy*dy*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+5*num_spixels+query_spixel_index], -.5 * dy*dz*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+6*num_spixels+query_spixel_index], -.5 * dz*dx*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+7*num_spixels+query_spixel_index], -.5 * dz*dy*dist_matrix_grad_val);
        atomicAdd(&spixel_invcov_grad[b*9*num_spixels+8*num_spixels+query_spixel_index], -.5 * dz*dz*dist_matrix_grad_val);

        // cmix_grad
        atomicAdd(&cmix_grad[b*1*num_spixels + query_spixel_index], dist_matrix_grad_val);
    }
}


// BxCxN
torch::Tensor forward_cuda(
    const torch::Tensor pixel_mu,
    const torch::Tensor spixel_mu,
    const torch::Tensor spixel_invcov,
    const torch::Tensor cmix,
    const torch::Tensor spixel_indices,
    torch::Tensor dist_matrix,
    int num_spixels_w, int num_spixels_h
){
    int batchsize   = pixel_mu.size(0);
    // int channels    = pixel_mu.size(1);
    int num_pixels  = pixel_mu.size(2);
    int num_spixels = spixel_mu.size(2);

    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(dist_matrix.type(), "forward_kernel", ([&] {
        forward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            pixel_mu.data<scalar_t>(),
            spixel_mu.data<scalar_t>(),
            spixel_invcov.data<scalar_t>(),
            cmix.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            dist_matrix.data<scalar_t>(),
            batchsize, num_pixels, num_spixels, 
            num_spixels_w, num_spixels_h
        );
    }));

    return dist_matrix; // Bx9xN
}



std::vector<torch::Tensor> backward_cuda(
    const torch::Tensor dist_matrix_grad,
    const torch::Tensor pixel_mu,
    const torch::Tensor spixel_mu,
    const torch::Tensor spixel_invcov,
    const torch::Tensor cmix,
    const torch::Tensor spixel_indices,
    torch::Tensor pixel_mu_grad,
    torch::Tensor spixel_mu_grad,
    torch::Tensor spixel_invcov_grad,
    torch::Tensor cmix_grad,
    int num_spixels_w, int num_spixels_h
){
    int batchsize = pixel_mu.size(0);
    int channels = pixel_mu.size(1);
    int num_pixels = pixel_mu.size(2);
    int num_spixels = spixel_mu.size(2);


    dim3 block((batchsize * 9 * num_pixels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES(pixel_mu_grad.type(), "backward_kernel", ([&] {
        backward_kernel<scalar_t><<< block, CUDA_NUM_THREADS >>>(
            dist_matrix_grad.data<scalar_t>(),
            pixel_mu.data<scalar_t>(),
            spixel_mu.data<scalar_t>(),
            spixel_invcov.data<scalar_t>(),
            cmix.data<scalar_t>(),
            spixel_indices.data<scalar_t>(),
            pixel_mu_grad.data<scalar_t>(),
            spixel_mu_grad.data<scalar_t>(),
            spixel_invcov_grad.data<scalar_t>(),
            cmix_grad.data<scalar_t>(),
            batchsize, channels, num_pixels,
            num_spixels, num_spixels_w, num_spixels_h
        );
    }));

    return {pixel_mu_grad, spixel_mu_grad, spixel_invcov_grad, cmix_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_cuda, "pair_wise_distance forward");
  m.def("backward", &backward_cuda, "pair_wise_distance backward");
}
'''
