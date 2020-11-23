import argparse
#from data_kitti import *
from dataloader.data_loader import *
import os
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
#parser.add_argument('--split', type=str, default='eigen', help='data split')
parser.add_argument('--predicted_depthSup_path', type=str, default='/eccv20dataset/yyeh/Synthetic2Realistic/results/openroom_nyu_supervised/test_latest/imagesLabeled', help='path to estimated depth')
parser.add_argument('--predicted_depthDA_path', type=str, default='/eccv20dataset/yyeh/Synthetic2Realistic/results/openroom_nyu_wsupervised/test_latest/imagesLabeled', help='path to estimated depth')
parser.add_argument('--gt_path', type = str, default='/eccv20dataset/NYU/depths/',
                    help = 'path to labeled NYU dataset')
#parser.add_argument('--file_path', type = str, default='../datasplit/', help = 'path to datasplit files')
#parser.add_argument('--save_path', type = str, default='/home/asus/lyndon/program/data/Image2Depth_31_KITTI/', help='path to save the train and test dataset')
parser.add_argument('--min_depth', type=float, default=1, help='minimun depth for evaluation')
parser.add_argument('--max_depth', type=float, default=8, help='maximun depth for evaluation, indoor 8.0, outdoor 50')
parser.add_argument('--normize_depth', type=float, default=10, help='depth normalization value, indoor 8.0, outdoor 80 (training scale)')
#parser.add_argument('--eigen_crop',action='store_true', help='if set, crops according to Eigen NIPS14')
#parser.add_argument('--garg_crop', action='store_true', help='if set, crops according to Garg  ECCV16')
parser.add_argument('--outputDir', type=str, default='combinedEval')
args = parser.parse_args()

def compute_errors(ground_truth, predication):

    # accuracy
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_depth(file_path, max_depth=10):
    depths = []
    dataset, _ = make_dataset(file_path)
    for data in dataset:
        depth = Image.open(data)
        depth = np.array(depth)#[:,:,0]
        depth = depth.astype(np.float32) / 255 * max_depth
        depths.append(depth)
    return depths

def loadNYUDepth(gt_path, pred_path, max_depth=10):
    depths = []
    dataset, _ = make_dataset(pred_path)    
    for imName in dataset:
        id = osp.basename(imName).split('_')[0]
        depth = osp.join(gt_path, '%s.tiff' % id)
        im = cv2.imread(depth, -1) # .tiff float32, meter
        im = np.asarray(im, dtype=np.float32)
        im = np.clip(im, 0, 10) 
        depths.append(im)
    return depths

def loadNYUImg(gt_path, pred_path, max_depth=10):
    depths = []
    dataset, _ = make_dataset(pred_path)    
    for imName in dataset:
        id = osp.basename(imName).split('_')[0]
        depth = osp.join(gt_path.replace('depths', 'images'), '%s.png' % id)
        im = cv2.imread(depth, -1) # 
        im = np.asarray(im, dtype=np.uint8)
        im = im[:, :, ::-1]
        depths.append(im)
    return depths

if __name__ == "__main__":

    predicted_depths_sup = load_depth(args.predicted_depthSup_path)
    predicted_depths_da = load_depth(args.predicted_depthDA_path)
    ground_truths = loadNYUDepth(args.gt_path, args.predicted_depthDA_path)
    inputs = loadNYUImg(args.gt_path, args.predicted_depthDA_path)
    num_samples = len(ground_truths)

    

    for mode in ['sup', 'da']:
        print('Evaluation mode: %s' % mode)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples,np.float32)
        rmse = np.zeros(num_samples,np.float32)
        rmse_log = np.zeros(num_samples,np.float32)
        a1 = np.zeros(num_samples,np.float32)
        a2 = np.zeros(num_samples,np.float32)
        a3 = np.zeros(num_samples,np.float32)
        for i in range(len(ground_truths)):
        # for i in range(1):
            ground_depth = ground_truths[i]

            if mode == 'sup':
                predicted_depth = predicted_depths_sup[i]
            elif mode == 'da':
                predicted_depth = predicted_depths_da[i]

            # print(ground_depth.max(),ground_depth.min())
            # print(predicted_depth.max(),predicted_depth.min())

            # depth_predicted = (predicted_depth / 7) * 255
            # depth_predicted = Image.fromarray(depth_predicted.astype(np.uint8))
            # depth_predicted.save(os.path.join('/home/asus/lyndon/program/Image2Depth/results/predicted_depth/', str(i)+'.png'))

            # depth = (depth / 80) * 255
            # depth = Image.fromarray(depth.astype(np.uint8))
            # depth.save(os.path.join('/data/result/syn_real_result/KITTI/ground_truth/{:05d}.png'.format(t_id)))

            predicted_depth[predicted_depth < args.min_depth] = args.min_depth
            predicted_depth[predicted_depth > args.max_depth] = args.max_depth

            ground_depth = ground_depth[12:468, 16:624]

            height, width = ground_depth.shape
            predicted_depth = cv2.resize(predicted_depth,(width,height),interpolation=cv2.INTER_LINEAR)

            mask = np.logical_and(ground_depth > args.min_depth, ground_depth < args.max_depth)

            abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(ground_depth[mask],predicted_depth[mask])

            print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i]))

        print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
        print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
            .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))

    print('Saving combined figures ...')
    if not osp.exists(args.outputDir):
        os.system('mkdir -p %s' % args.outputDir)
    for i in range(len(ground_truths)):
        # for i in range(1):
        ground_depth = ground_truths[i]
        predicted_depth_sup = predicted_depths_sup[i]
        predicted_depth_da = predicted_depths_da[i]
        input = inputs[i]

        ground_depth = ground_depth[12:468, 16:624]
        predicted_depth_sup = predicted_depth_sup[6:186, 8:248]
        predicted_depth_sup = predicted_depth_sup[6:186, 8:248]
        input = input[12:468, 16:624]
        height, width = ground_depth.shape
        predicted_depth_sup = cv2.resize(predicted_depth_sup,(width,height),interpolation=cv2.INTER_LINEAR)
        predicted_depth_da = cv2.resize(predicted_depth_da,(width,height),interpolation=cv2.INTER_LINEAR)
        input = cv2.resize(input,(width,height),interpolation=cv2.INTER_LINEAR)

        depths = np.concatenate([predicted_depth_sup, predicted_depth_da, ground_depth], axis= 1)
        dMin = np.min(depths)
        dMax = np.max(depths)
        depthsImg= np.uint8(cm.jet( (depths-dMin)/(dMax-dMin) )*255)[:,:,:3]
        output = np.concatenate([input, depthsImg], axis=1)
        Image.fromarray(output).save(osp.join(args.outputDir, '%03d.png' % i) )
        print('Combined image saved at %s' % osp.join(args.outputDir, '%03d.png' % i) )
