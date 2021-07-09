'''
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu@nvidia.com>
'''

# batch loader for videos, each batch would be a dataloader (dl_scanNet, dl_vkitti or dl_kitti)

import numpy as np
import random

import torch
import torch.utils.data as data
import model.align as align 

import utils.misc as m_misc 
import copy

def collate_fn(batch):
    ''' collate functino for the batch data returned from Batch_Loader_iter__next__() 

    OUTPUTS:
    dict with keys:
    imgs_ref -      nbatch x 3 x H x W
    imgs_src -      nbatch x N_src*3 x H x W
    dmaps_ref-      nbatch x 1 x H x W
    dmaps_src -     nbatch x N_src x H x W
    src_cam_poses - nbatch x N_src x 4 x 4
    imgs_ref_path - [str0, ..., str_nbatch]
    '''
    for b in batch:
        if b is None: # end of batch
            raise StopIteration

    last_frame_in_traj =False
    for b in batch:
        if b['last_frame_in_traj']:
            last_frame_in_traj= True
            break

    first_frame_in_traj =False
    for b in batch:
        if b['first_frame_in_traj']:
            first_frame_in_traj= True
            break

    imgs_ref_path = [b['ref_dats']['img_path'] for b in batch]
    imgs_ref = torch.cat( [b['ref_dats']['img'] for b in batch] ,dim=0 )
    dmaps_ref = torch.cat( [b['ref_dats']['dmap_imgsize'].unsqueeze(0) for b in batch] ,dim=0 )

    dmaps_ref_raw = torch.cat( [b['ref_dats']['dmap_rawsize'].unsqueeze(0) for b in batch] ,dim=0 )
    imgs_ref_raw = torch.cat( [b['ref_dats']['img_raw'] for b in batch] ,dim=0 )
    extMs = torch.cat( [ torch.from_numpy(b['ref_dats']['extM']).unsqueeze(0) for b in batch] ,dim=0 )

    if len(b['src_cam_poses'])>0:
        src_cam_poses = torch.cat([b['src_cam_poses'] for b in batch], dim=0 )
        imgs_src, dmaps_src = [], []
        for b in batch:
            imgs_src.append(torch.cat([b['src_dats'][i]['img'] for i in range( src_cam_poses.shape[1] ) ] ,dim=1))
            dmaps_src.append(torch.cat([b['src_dats'][i]['dmap_imgsize'].unsqueeze(0) for i in range( src_cam_poses.shape[1] ) ] ,dim=1 ) )
        imgs_src = torch.cat(imgs_src,  dim=0)
        dmaps_src = torch.cat(dmaps_src, dim=0)

    else:#t_win_r==0
        imgs_src = []
        dmaps_src = []
        src_cam_poses = []

    output = {
            'imgs_ref': imgs_ref, 'imgs_src': imgs_src, 
            'imgs_ref_raw': imgs_ref_raw,
            'dmaps_ref': dmaps_ref, 'dmaps_src': dmaps_src, 
            'dmaps_ref_raw': dmaps_ref_raw, 
            'src_cam_poses': src_cam_poses, 
            'imgs_ref_path': imgs_ref_path, 
            'last_frame_traj': last_frame_in_traj, 
            'first_frame_traj': first_frame_in_traj, 
            'extM': extMs}

    return output
    


class Batch_Loader_iter(data.IterableDataset):

    def __init__(
            self, batch_size, fun_get_paths, 
            dataset_traj, nTraj, 
            dataset_name,
            t_win_r = 2, 
            traj_Indx = None,
            preload_poses= False,
            dso_res_path= None):
        '''
        fun_get_paths - function, takes input the traj index and output: 
        fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path =  fun_get_paths(traj_idx) 

        inputs:
        dataset_traj - a dataset object for one trajectory, we will have multiple dataset in one batch 

        preload_poses - if preload all camera poses for a given trajectory. This is useful if we are working with DSO poses

        members: 
            nbatch - the # of trajs per batch loop 

            state variables: 
                self.trajs_st_frame
                self.traj_len 
                self.dataset_batch
                self.frame_idx_cur - the current frame index in the trajectories 
                self.dat_arrays - list of dat_array, each dat_array is a list of data items for one local time window, 
        ''' 
        # assert batch_size > 1
        
        self.first_frame_in_batch = True
        self.batch_size = batch_size
        self.fun_get_paths = fun_get_paths 
        self.dataset_name = dataset_name
        self.ntraj = nTraj 
        self.t_win_r = int(t_win_r )
        self.ib = 0 #pointer to current batch being output
        self.dataset_traj = dataset_traj
        if traj_Indx is None:
            traj_Indx = np.arange(0, self.ntraj) 

        self.traj_Indx = traj_Indx 
        batch_traj_st = np.arange(0, len(traj_Indx))[::self.batch_size]
        self.nbatch = len(batch_traj_st)
        self.batch_traj_st = batch_traj_st 

        # initialize the traj schedule # 
        self.batch_traj_ed = batch_traj_st + batch_size-1 
        self.batch_traj_ed[self.batch_traj_ed >= len(traj_Indx)-1] = len(traj_Indx)-1

        self.batch_idx = 0 # the batch index
        self.batch_traj_idx_cur = np.linspace(\
                self.batch_traj_st[self.batch_idx], self.batch_traj_ed[self.batch_idx], self.batch_size).astype(int) # the trajectory indexes in the current batch

        # Initialize the batch #
        dataset_batch = [] 
        for ibatch in range(batch_size):
            traj_indx_per = self.traj_Indx[self.batch_traj_idx_cur[ibatch]]
            try:
                fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = fun_get_paths(traj_indx_per) 
            except:
                print('loader fail')
                import ipdb; ipdb.set_trace() 

            dataset_batch.append( copy.copy(dataset_traj) ) 

            if dataset_name == 'scanNet_vid':
                dataset_batch[-1].set_traj_idx(traj_indx_per) # load video
            else:
                dataset_batch[-1].set_paths(img_seq_paths, dmap_seq_paths, poses) 
                if dataset_name == 'scanNet':
                    # For each trajector in the dataset, we will update the intrinsic matrix #
                    dataset_batch[-1].get_cam_intrinsics(intrin_path)

        self.dataset_batch = dataset_batch

        # get the start frame and traj lengths for all trajectories #
        self.trajs_st_frame, self.traj_len = self._get_traj_lengths() 

        # get imgs_ref #
        #dat_arrays is a list of dat_array, each dat_array is a list of data items for one local time window#
        dat_arrays = [] # data
        for ibatch in range( batch_size ):
            dat_array_ = [self.dataset_batch[ibatch][idx] for idx in range(
                          self.trajs_st_frame[ibatch] - t_win_r, self.trajs_st_frame[ibatch] + t_win_r + 1)]
            dat_arrays.append(dat_array_) 
        self.frame_idx_cur = 0 
        self.dat_arrays = dat_arrays

        self.preload_poses = preload_poses
        if self.preload_poses: # preload all pre-computed camera poses
            assert dso_res_path is not None
            pass

    def reset(self):
        self.__init__(self.batch_size, self.fun_get_paths, self.dataset_traj, self.ntraj, self.dataset_name, self.t_win_r, self.traj_Indx)

    def _get_traj_lengths(self):
        raw_traj_batch_len = np.array( [len(dataset_) for dataset_ in self.dataset_batch] ) 
        traj_len = raw_traj_batch_len.min() - 2 * self.t_win_r
        trajs_st_frame = np.zeros(self.batch_size).astype( np.int) 
        t_win_r = self.t_win_r 
        for ibatch in range(self.batch_size):
            if raw_traj_batch_len[ibatch] == traj_len + 2* self.t_win_r:
                trajs_st_frame[ibatch] = self.t_win_r
            elif raw_traj_batch_len[ibatch] - traj_len - t_win_r > t_win_r:
                trajs_st_frame[ibatch] = int(random.randrange(self.t_win_r, raw_traj_batch_len[ibatch] - traj_len - t_win_r) )
            else:
                trajs_st_frame[ibatch] = self.t_win_r 
        return trajs_st_frame, traj_len 


    def __len__(self):
        # return self.nbatch
        return self.traj_len

    def __iter__(self):
        return self

    def __next__(self):
        # get val #
        local_info = self.local_info(self.ib)

        local_info['last_frame_in_traj'] = False
        if self.frame_idx_cur >= self.traj_len-1:
            local_info['last_frame_in_traj'] = True 

        local_info['first_frame_in_traj'] = False
        if self.first_frame_in_batch:
            local_info['first_frame_in_traj'] = True 
            self.first_frame_in_batch= False

        # proceed #
        self.ib += 1
        self.ib = int(self.ib) % int(self.batch_size)
        if self.ib==0:
            self.frame_idx_cur += 1 
            if self.frame_idx_cur >= self.traj_len:
                print('Batch_Loader_iter(): already the last frame of current traj! Proceed traj')
                print(f'last img path: {local_info["ref_dats"]["img_path"]}')

                self.batch_idx +=1 
                if self.batch_idx >= self.nbatch:
                    # print('Batch_Loader_iter(): end of one epoch!')
                    return None # end of iterration
                else:
                    self.proceed_batch()
            else:
                self.proceed_frame()

        return local_info

    def proceed_batch(self):
        '''
        Move forward one batch of trajecotries
        ''' 

        # self.batch_idx += 1
        batch_size = self.batch_size

        if self.batch_idx >= self.nbatch: # reaching the last batch 
            return False

        else: 
            self.batch_traj_idx_cur = np.linspace(
                    self.batch_traj_st[self.batch_idx], 
                    self.batch_traj_ed[self.batch_idx], 
                    self.batch_size).astype(int) # the trajectory indexes in the current batch

            # re-set the traj. in the current batch #

            for ibatch in range(batch_size):
                traj_indx_per = self.traj_Indx[ self.batch_traj_idx_cur[ibatch] ]
                fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = self.fun_get_paths(traj_indx_per) 

                if self.dataset_name == 'scanNet_vid':
                    self.dataset_batch[ibatch].set_traj_idx(traj_indx_per) # load video
                else:
                    self.dataset_batch[ibatch].set_paths(img_seq_paths, dmap_seq_paths, poses) 
                    if self.dataset_name == 'scanNet':
                        # For each trajector in the dataset, we will update the intrinsic matrix #
                        self.dataset_batch[ibatch].get_cam_intrinsics(intrin_path)


            # get the start frame and traj lengths for all trajectories #
            trajs_st_frame, traj_len = self._get_traj_lengths() 
            self.trajs_st_frame = trajs_st_frame
            self.traj_len = traj_len
            self.frame_idx_cur = 0

            # get dat_arrays #
            dat_arrays = []
            for ibatch in range( batch_size ):
                dat_array_ = [ self.dataset_batch[ibatch][idx] for idx in range(
                                trajs_st_frame[ibatch] - self.t_win_r, trajs_st_frame[ibatch] + self.t_win_r + 1) ]  

                dat_arrays.append(dat_array_)

            self.dat_arrays = dat_arrays 

            self.first_frame_in_batch = True

            return True

    def proceed_frame(self):
        '''
        Move forward one frame for all trajectories
        ''' 
        for ibatch in range( self.batch_size ):
            self.dat_arrays[ibatch].pop(0) 
            self.dat_arrays[ibatch].append( 
                    self.dataset_batch[ibatch][self.frame_idx_cur + self.trajs_st_frame[ibatch] + self.t_win_r])

    def local_info(self, ibatch):
        '''
        return local info, including { cam_intrins, ref_dats, src_dats, src_cam_poses, is_valid } 
        each is a list of variables 

        src_cam_poses[i] - 1 x n_src x 4 x 4
        '''


        dat_array_ = self.dat_arrays[ibatch]
        ref_dat_, src_dat_ = m_misc.split_frame_list(dat_array_, self.t_win_r) 
        src_cam_extMs = m_misc.get_entries_list_dict(src_dat_, 'extM') 
        # print(src_cam_extMs, src_cam_extMs[0].shape, len(src_cam_extMs)) # list of 4x4 arrays
        src_cam_pose_ = [ align.get_rel_extrinsicM(ref_dat_['extM'], src_cam_extM_) for src_cam_extM_ in src_cam_extMs ] 
        src_cam_pose_ = [ torch.from_numpy(pose.astype(np.float32)).unsqueeze(0) for pose in src_cam_pose_ ] 
        # print(src_cam_pose_, src_cam_pose_[0].shape, len(src_cam_pose_)) # list of 4x4 arrays

        # src_cam_poses size: N V 4 4
        if len(src_cam_pose_)>0:
            src_cam_pose_ = torch.cat(src_cam_pose_, dim=0).unsqueeze(0) 
        else:
            src_cam_pose_ = []

        ref_dats = ref_dat_
        src_dats = src_dat_
        cam_intrins = self.dataset_batch[ibatch].cam_intrinsics
        # print(cam_intrins['intrinsic_M'])
        src_cam_poses = src_cam_pose_

        return { 'cam_intrins': cam_intrins, 'ref_dats': ref_dats, 'src_dats': src_dats, 'src_cam_poses': src_cam_poses } 