--------------------------------------------------------

usage (master branch):
1. download https://drive.google.com/file/d/11iR41htU5X8U4fAJ-oRCGf0-KbqdGEbq/view?usp=sharing and unzip it to root folder root/data/
2. open script_optim.sh
3. run >>bash script_optim.sh
 
Folders:

	apriltag/
	higan/
	data/
		pretrian/
		in/
		out/
	src/

-----------------------------------------------------------
usage:
1. download https://drive.google.com/file/d/1lifa0VAz0eLvoXhvfpTK4Fe5JGuyUSKe/view?usp=sharing and unzip "data.zip"
2. open script_optim.sh and modify root_dir=data/ 
3. run >>bash script_optim.sh
 
data:

	svbrdf_data.zip
		checkpoints/
			checkpoint_autoencoder.pt
			checkpoint_stylegan_256.pt
			checkpoint_stylegan_256.pt
			latent_avg_w.pt
			latent_avg_w+.pt
			vgg_conv.pth
		in/ours/
		out/ours/

----------------------------------------------------------
usage: (old)

>> python3 optim.py 

[Required]:

	--in_dir            path to input texture maps, should have at least one material
						eg. ../svbrdf_data/in/ 

	--out_dir           path to results 
						eg. ../svbrdf_data/out/test/ 

	
	--checkpoint_dir    path to autoencoder or stylegan weights
				eg. ../svbrdf_data/checkpoints/checkpoint_stylegan.pt 

	--use_styleGAN      replace it with --use_autoEncoder if use autoencoder
	
	--gan_latent	    optimize on z|w|w+ space
	
	--gan_init          init of optimization, `random` or path to average latent space
				eg. ../svbrdf_data/checkpoints/latent_avg_w+.pt

	--vgg_weight_dir    path to "vgg_conv.pth"
				eg. ../svbrdf_data/checkpoints/vgg_conv.pth 

	--loss_type         loss on rendering: 
				1.   pixel loss
				2.   feature loss
				12.  pixel loss + feature loss
			    loss on texture maps:
				10.  pixel loss
				20.  feature loss
				120. pixel loss + feature loss

	--loss_weight	    depands on which loss_type is using
				eg. --loss_type 13, --loss_weight 10 1 
	--scalar	    downsample image before computing loss

	--vgg_layer_weight  0.002 0.002 0.008 0.016

	--epochs 	    no default value. use N+1
				eg. 1001 or 5001			 

[Optional]:

	--mat_fn	    specify one material, eg. leather_bull
				remove this line will process all the materials in --in_dir
	
	--init_from_latent  styleGAN initialization, eg. ../svbrdf_data/checkpoints/latentW_avg.pth 
				remove this line for random initialization 
	
	--im_res 	    input image resolution, default 256 
	
	--lr 		    learning rate, default 0.01 
	
	--num_render        number of renderings used for optimization, default 5	
