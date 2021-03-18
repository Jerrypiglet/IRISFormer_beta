import os
from PIL import Image

# mat_list = ['real_book1', 'real_cards-red', 'real_giftbag1', 'real_wall-plaster-white', 'real_plastic-red-carton', 'real_wood-t', 'real_bathroomtile2', 'real_book2', 'real_other-bamboo-veawe', 'real_wood-tile']
mat_list = ['real_leather-blue']
out_dir = '../svbrdf_paper/materialgan/images/results/main/'

def gyCreateFolder(dir):
    if not os.path.exists(dir):
        print("\ncreate directory: ", dir)
        os.makedirs(dir)

def copyfolder(A,B):
	for i in range(9):
		s = 'data/out/%s/%s/%02d.png' % (A, mat, i)
		d = '%s%s/%s/%02d.png' % (out_dir, mat, B, i)
		cmd = 'cp %s %s' % (s, d)
		os.system(cmd)

	s = 'data/out/%s/%s/tex.png' % (A, mat)
	d = '%s%s/%s/tex.png' % (out_dir, mat, B)
	cmd = 'cp %s %s' % (s, d)
	os.system(cmd)

for mat in mat_list:
	root_dir = out_dir+mat+'/'
	gyCreateFolder(root_dir)

	# ref
	gyCreateFolder(root_dir + 'ref/')
	for i in range(9):
		s = 'data/in/%s/%02d.png' % (mat, i)
		d = '%s%s/ref/%02d.png' % (out_dir, mat, i)
		Image.open(s).resize((256,256)).save(d)
	if mat[:4] == 'fake':
		s = 'data/in/%s/tex.png' % (mat)
		d = '%s%s/ref/tex.png' % (out_dir, mat)
		cmd = 'cp %s %s' % (s, d)
		os.system(cmd)

	# oursA
	# gyCreateFolder(root_dir + 'oursA/')
	# copyfolder('ours7_picked','oursA')

	# oursAR
	gyCreateFolder(root_dir + 'oursAR/')
	copyfolder('ours7_picked_refine','oursAR')

	# oursB
	# gyCreateFolder(root_dir + 'oursB/')
	# copyfolder('ours7_egsr7','oursB')

	# oursBR
	# gyCreateFolder(root_dir + 'oursBR/')
	# copyfolder('ours7_egsr7_refine','oursBR')

	# msraA
	# gyCreateFolder(root_dir + 'msraA/')
	# copyfolder('msra7_picked','msraA')

	# msraB
	# gyCreateFolder(root_dir + 'msraB/')
	# copyfolder('msra7_egsr7','msraB')

	# msraBR
	gyCreateFolder(root_dir + 'msraBR/')
	copyfolder('msra7_egsr7_refine','msraBR')
