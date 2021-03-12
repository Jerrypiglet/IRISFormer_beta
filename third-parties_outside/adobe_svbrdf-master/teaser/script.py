import os
import numpy as np
from PIL import Image
eps = 0.01
# os.system('cd /home/guoyu/Documents/1_layeredbsdf/layeredbsdf/')
# os.system('source setpath.sh')
# os.system('cd /home/guoyu/Documents/3_svbrdf/svbrdf/teaser/')

def gyCreateFolder(dir):
    if not os.path.exists(dir):
        print("\ncreate directory: ", dir)
        os.makedirs(dir)

def gyPIL2Array(im):
    return np.array(im).astype(np.float32)/255

def gyArray2PIL(im):
    return Image.fromarray((im*255).astype(np.uint8))

def gyApplyGamma(im, gamma):
    if gamma < 1: im = im.clip(min=eps)
    return im**gamma

def gyApplyGammaPIL(im, gamma):
    return gyArray2PIL(gyApplyGamma(gyPIL2Array(im),gamma))

if True:
	mat_list = [ \
	# 'real_bathroomtile1', \
	# 'real_bathroomtile2', \
	# 'real_book1', \
	# 'real_cards-blue', \
	# 'real_cards-red', \
	# 'real_giftbag1', \
	'real_giftbag2', \
	'real_leather-blue', \
	'real_plastic-foam', \
	'real_plastic-red-carton', \
	'real_stone-bigtile', \
	'real_stone-smalltile', \
	'real_wall-color', \
	'real_wood-t', \
	'real_wood-tile', \
	'real_wood-walnut' \
	]
	for mat in mat_list:
		in_dir = '../data/teaser/in/'
		out_dir = '../data/teaser/out/'
		tmp_dir = '../data/teaser/tmp/'
		gyCreateFolder(out_dir + mat + '/')
		res = 256

		# copy maps
		tex = Image.open(in_dir + mat + '.png')
		tex.crop((0,0,res,res)).save(tmp_dir+mat+'_albedo.png')
		tex.crop((res,0,res*2,res)).save(tmp_dir+mat+'_normal.png')
		rough = tex.crop((res*2,0,res*3,res))
		rough = gyPIL2Array(rough)
		rough = ((rough ** 2.2) ** 2) ** (1/2.2)
		gyArray2PIL(rough).save(tmp_dir+mat+'_rough.png')
		tex.crop((res*3,0,res*4,res)).save(tmp_dir+mat+'_specular.png')

		scaleDiff = 1
		scaleSpec = 1
		if mat == 'real_cards-red' or mat == 'real_cards-blue':
			scaleDiff = 1.5
			scaleSpec = 1.5
		if mat == 'real_giftbag2' or mat == 'real_wood-jatoba':
			scaleDiff = 0.5
			scaleSpec = 0.5
		if mat == 'real_book1':
			scaleDiff = 0.2
			scaleSpec = 0.2
		if mat == 'real_bathroomtile1':
			scaleDiff = 0.4
			scaleSpec = 0.4

		for i, angle in enumerate(range(-30,21,2)):
			# render
			cmd = 'mitsuba -p 6 material_anim.xml' \
				+ ' -o %s%s/%03d.png' % (out_dir, mat, i) \
				+ ' -Dalbedo=%s%s_albedo.png' % (tmp_dir, mat) \
				+ ' -Dnormal=%s%s_normal.png' % (tmp_dir, mat) \
				+ ' -Drough=%s%s_rough.png' % (tmp_dir, mat) \
				+ ' -Dspecular=%s%s_specular.png' % (tmp_dir, mat) \
				+ ' -DscaleDiffuse=%f' % scaleDiff \
				+ ' -DscaleSpecular=%f' % scaleSpec \
				+ ' -Dy_angle=%f' % angle
			print(cmd)
			os.system(cmd)
		for j, ii in enumerate(range(i-1,0,-1)):
			os.system('cp %s%s/%03d.png %s%s/%03d.png' % (out_dir, mat, ii, out_dir, mat, i+j+1))
		# exit()
		os.system('convert -delay 4 -loop 99 %s%s/*.png %s%s.gif' % (out_dir, mat, out_dir, mat))
		# break

if False:
	# mat_list = [ \
	# 'real_cards-red', \
	# 'real_cards-blue' \
	# 'real_giftbag1', \
	# 'real_leather-blue', \
	# 'real_plastic-foam', \
	# 'real_plastic-red-carton', \
	# 'real_stone-bigtile', \
	# 'real_stone-smalltile', \
	# 'real_wall-color', \
	# 'real_wood-t', \
	# 'real_wood-tile', \
	# 'real_wood-walnut', \
	# ]

	mat_list = [ \
	'real_bathroomtile1'\
	# 'real_book1'\
	]
	in_dir = '/home/guoyu/Documents/3_svbrdf/svbrdf_paper/materialgan/images/teaser3/maps/'
	out_dir = '/home/guoyu/Documents/3_svbrdf/svbrdf_paper/materialgan/images/teaser3/'
	res = 256
	for mat in mat_list:
		# copy maps
		tex = Image.open(in_dir + mat + '.png')
		tex.crop((0,0,res,res)).save('maps/'+mat+'_albedo.png')
		tex.crop((res,0,res*2,res)).save('maps/'+mat+'_normal.png')
		rough = tex.crop((res*2,0,res*3,res))
		rough = gyPIL2Array(rough)
		rough = ((rough ** 2.2) ** 2) ** (1/2.2)
		gyArray2PIL(rough).save('maps/'+mat+'_rough.png')
		tex.crop((res*3,0,res*4,res)).save('maps/'+mat+'_specular.png')

		scaleDiff = 1
		scaleSpec = 1
		if mat == 'real_cards-red' or mat == 'real_cards-blue':
			scaleDiff = 1.5
			scaleSpec = 1.5
		if mat == 'real_giftbag2' or mat == 'real_wood-jatoba':
			scaleDiff = 0.5
			scaleSpec = 0.5
		if mat == 'real_book1':
			scaleDiff = 0.2
			scaleSpec = 0.2
		if mat == 'real_bathroomtile1':
			scaleDiff = 0.4
			scaleSpec = 0.4

		# render
		cmd = 'mitsuba -p 6 material.xml -o %s/%s.png -Dalbedo=maps/%s_albedo.png -Dnormal=maps/%s_normal.png -Drough=maps/%s_rough.png -Dspecular=maps/%s_specular.png -DscaleDiffuse=%f -DscaleSpecular=%f' % (out_dir, mat, mat, mat, mat, mat, scaleDiff, scaleSpec)
		print(cmd)
		os.system(cmd)
		# exit()

if False:
	mat_list = [ \
	'interp1' \
	]
	res = 256
	for mat in mat_list:
		# copy maps
		tex = Image.open('interp/' + mat + '.png')
		tex.crop((0,0,res*5,res)).save('maps/'+mat+'_albedo.png')
		tex.crop((0,res,res*5,res*2)).save('maps/'+mat+'_normal.png')
		rough = tex.crop((0,res*2,res*5,res*3))
		rough = gyPIL2Array(rough)
		rough = rough ** 2
		gyArray2PIL(rough).save('maps/'+mat+'_rough.png')
		tex.crop((0,res*3,res*5,res*4)).save('maps/'+mat+'_specular.png')

		# render
		cmd = 'mitsuba -p 6 material_interp.xml -o out/%s.png -Dalbedo=maps/%s_albedo.png -Dnormal=maps/%s_normal.png -Drough=maps/%s_rough.png -Dspecular=maps/%s_specular.png' % (mat, mat, mat, mat, mat)
		print(cmd)
		os.system(cmd)
		# exit()
