import sys
sys.path.insert(1, 'src/')
from util import *
import shutil

html_dir = 'data/html/'

ref_dir    = 'data/in/'
egsr00_dir  = 'data/out/egsr7/'
egsr01_dir  = 'data/out/egsr7_refine/'
msra00_dir = 'data/out/msra7_avg/'
msra01_dir = 'data/out/msra7_avg_refine/'
msra10_dir = 'data/out/msra7_const/'
msra11_dir = 'data/out/msra7_const_refine/'
msra20_dir = 'data/out/msra7_egsr7/'
msra21_dir = 'data/out/msra7_egsr7_refine/'
msra30_dir = 'data/out/msra7_picked/'
msra31_dir = 'data/out/msra7_picked_refine/'
ours00_dir  = 'data/out/ours7_avg/'
ours01_dir  = 'data/out/ours7_avg_refine/'
ours10_dir  = 'data/out/ours7_const/'
ours11_dir  = 'data/out/ours7_const_refine/'
ours20_dir  = 'data/out/ours7_egsr7/'
ours21_dir  = 'data/out/ours7_egsr7_refine/'
ours30_dir  = 'data/out/ours7_picked/'
ours31_dir  = 'data/out/ours7_picked_refine/'

_ref_dir    = 'img/ref/'
_egsr00_dir  = 'img/egsr7/'
_egsr01_dir  = 'img/egsr7_refine/'
_msra00_dir = 'img/msra7_avg/'
_msra01_dir = 'img/msra7_avg_refine/'
_msra10_dir = 'img/msra7_const/'
_msra11_dir = 'img/msra7_const_refine/'
_msra20_dir = 'img/msra7_egsr7/'
_msra21_dir = 'img/msra7_egsr7_refine/'
_msra30_dir = 'img/msra7_picked/'
_msra31_dir = 'img/msra7_picked_refine/'
_ours00_dir  = 'img/ours7_avg/'
_ours01_dir  = 'img/ours7_avg_refine/'
_ours10_dir  = 'img/ours7_const/'
_ours11_dir  = 'img/ours7_const_refine/'
_ours20_dir  = 'img/ours7_egsr7/'
_ours21_dir  = 'img/ours7_egsr7_refine/'
_ours30_dir  = 'img/ours7_picked/'
_ours31_dir  = 'img/ours7_picked_refine/'

_ref_png_dir    = 'img_png/ref/'
_egsr00_png_dir  = 'img_png/egsr7/'
_egsr01_png_dir  = 'img_png/egsr7_refine/'
_msra00_png_dir = 'img_png/msra7_avg/'
_msra01_png_dir = 'img_png/msra7_avg_refine/'
_msra10_png_dir = 'img_png/msra7_const/'
_msra11_png_dir = 'img_png/msra7_const_refine/'
_msra20_png_dir = 'img_png/msra7_egsr7/'
_msra21_png_dir = 'img_png/msra7_egsr7_refine/'
_msra30_png_dir = 'img_png/msra7_picked/'
_msra31_png_dir = 'img_png/msra7_picked_refine/'
_ours00_png_dir  = 'img_png/ours7_avg/'
_ours01_png_dir  = 'img_png/ours7_avg_refine/'
_ours10_png_dir  = 'img_png/ours7_const/'
_ours11_png_dir  = 'img_png/ours7_const_refine/'
_ours20_png_dir  = 'img_png/ours7_egsr7/'
_ours21_png_dir  = 'img_png/ours7_egsr7_refine/'
_ours30_png_dir  = 'img_png/ours7_picked/'
_ours31_png_dir  = 'img_png/ours7_picked_refine/'

def func(png, jpg):
    file.write('    <td><a href="%s" target="_blank"><img alt="" src="%s"></a></td>\n' % (png, jpg))


def writeline(name, folder):
    file.write('  <tr>\n')
    file.write('    <td>%s</td>\n' % name)
    png = os.path.join(folder, 'tex.jpg')
    jpg = os.path.join(folder, 'jpg/tex.jpg')
    func(png, jpg)
    png = os.path.join(folder, 'rendered_val.jpg')
    jpg = os.path.join(folder, 'jpg/rendered_val.jpg')
    func(png, jpg)
    png = os.path.join(folder, 'rendered_nov.jpg')
    jpg = os.path.join(folder, 'jpg/rendered_nov.jpg')
    func(png, jpg)
    file.write('  </tr>\n')

def copyfile(s,d):
    gyCreateFolder(d+'jpg/')
    if os.path.exists(s+'tex.png'):
        tex = Image.open(s+'tex.png')
        tex.save(d+'tex.jpg', quality=95, optimize=True)
        tex.resize((128*4,128)).save(d+'jpg/tex.jpg', quality=95, optimize=True)
    if os.path.exists(s+'rendered.png'):
        rendered = Image.open(s+'rendered.png')
        rendered1 = rendered.crop((0,0,256*7,256))
        rendered1.save(d+'rendered_val.jpg', quality=95, optimize=True)
        rendered2 = rendered.crop((256*7,0,256*9,256))
        rendered2.save(d+'rendered_nov.jpg', quality=95, optimize=True)
        rendered1.resize((128*7,128)).save(d+'jpg/rendered_val.jpg', quality=95, optimize=True)
        rendered2.resize((128*2,128)).save(d+'jpg/rendered_nov.jpg', quality=95, optimize=True)

def copyfile_png(s,d):
    gyCreateFolder(d)
    if os.path.exists(s+'tex.png'):
        tex = Image.open(s+'tex.png')
        tex.save(d+'tex.png')
    if os.path.exists(s+'rendered.png'):
        rendered = Image.open(s+'rendered.png')
        rendered1 = rendered.crop((0,0,256*7,256))
        rendered1.save(d+'rendered_val.png')
        rendered2 = rendered.crop((256*7,0,256*9,256))
        rendered2.save(d+'rendered_nov.png')



mat_list = gyListNames(ref_dir+'*')
for idx, mat in enumerate(mat_list):
    if 1 == 1:
        print(mat)

        # copyfile_png(ref_dir+mat+'/',    html_dir+_ref_png_dir+mat+'/')
        # copyfile_png(egsr00_dir+mat+'/',  html_dir+_egsr00_png_dir+mat+'/')
        copyfile_png(egsr01_dir+mat+'/',  html_dir+_egsr01_png_dir+mat+'/')
        # copyfile_png(msra00_dir+mat+'/', html_dir+_msra00_png_dir+mat+'/')
        # copyfile_png(msra01_dir+mat+'/', html_dir+_msra01_png_dir+mat+'/')
        # copyfile_png(msra10_dir+mat+'/', html_dir+_msra10_png_dir+mat+'/')
        # copyfile_png(msra11_dir+mat+'/', html_dir+_msra11_png_dir+mat+'/')
        # copyfile_png(msra20_dir+mat+'/', html_dir+_msra20_png_dir+mat+'/')
        # copyfile_png(msra21_dir+mat+'/', html_dir+_msra21_png_dir+mat+'/')
        # copyfile_png(msra30_dir+mat+'/', html_dir+_msra30_png_dir+mat+'/')
        # copyfile_png(msra31_dir+mat+'/', html_dir+_msra31_png_dir+mat+'/')
        # copyfile_png(ours00_dir+mat+'/',  html_dir+_ours00_png_dir+mat+'/')
        # copyfile_png(ours10_dir+mat+'/',  html_dir+_ours10_png_dir+mat+'/')
        # copyfile_png(ours20_dir+mat+'/',  html_dir+_ours20_png_dir+mat+'/')
        # copyfile_png(ours21_dir+mat+'/',  html_dir+_ours21_png_dir+mat+'/')
        # copyfile_png(ours30_dir+mat+'/',  html_dir+_ours30_png_dir+mat+'/')
        # copyfile_png(ours31_dir+mat+'/',  html_dir+_ours31_png_dir+mat+'/')


        # copyfile(ref_dir+mat+'/',    html_dir+_ref_dir+mat+'/')
        # copyfile(egsr00_dir+mat+'/',  html_dir+_egsr00_dir+mat+'/')
        copyfile(egsr01_dir+mat+'/',  html_dir+_egsr01_dir+mat+'/')
        # copyfile(msra00_dir+mat+'/', html_dir+_msra00_dir+mat+'/')
        # copyfile(msra01_dir+mat+'/', html_dir+_msra01_dir+mat+'/')
        # copyfile(msra10_dir+mat+'/', html_dir+_msra10_dir+mat+'/')
        # copyfile(msra11_dir+mat+'/', html_dir+_msra11_dir+mat+'/')
        # copyfile(msra20_dir+mat+'/', html_dir+_msra20_dir+mat+'/')
        # copyfile(msra21_dir+mat+'/', html_dir+_msra21_dir+mat+'/')
        # copyfile(msra30_dir+mat+'/', html_dir+_msra30_dir+mat+'/')
        # copyfile(msra31_dir+mat+'/', html_dir+_msra31_dir+mat+'/')
        # copyfile(ours00_dir+mat+'/',  html_dir+_ours00_dir+mat+'/')
        # copyfile(ours10_dir+mat+'/',  html_dir+_ours10_dir+mat+'/')
        # copyfile(ours20_dir+mat+'/',  html_dir+_ours20_dir+mat+'/')
        # copyfile(ours21_dir+mat+'/',  html_dir+_ours21_dir+mat+'/')
        # copyfile(ours30_dir+mat+'/',  html_dir+_ours30_dir+mat+'/')
        # copyfile(ours31_dir+mat+'/',  html_dir+_ours31_dir+mat+'/')


with open(html_dir + 'fake.html', 'w') as file:

    file.write('<!DOCTYPE html>\n')
    file.write('<html>\n')
    file.write('<body>\n')
    file.write('<h1 align="center">SVBRDF</h1>\n')
    file.write('<table align="center">\n')

    file.write('  <tr>\n')
    file.write('    <td colspan="10">All images are in gamma 2.2 space, except "Normal"</td>\n')
    file.write('  </tr>\n')

    mat_list = gyListNames(ref_dir+'fake_*')
    for idx, mat in enumerate(mat_list):
        if 1 == 1:
            print(mat)
            file.write('  <tr>\n')
            file.write('    <td><center><b>%s</b></center></td>\n' % mat)
            file.write('  </tr>\n')

            writeline('<center>ref</center>',             _ref_dir+mat+'/')
            writeline('<center>egsr</center>',         _egsr00_dir+mat+'/')
            writeline('<center>egsr-r</center>',       _egsr01_dir+mat+'/')
            writeline('<center>msra-avg</center>',     _msra00_dir+mat+'/')
            writeline('<center>msra-const</center>',   _msra10_dir+mat+'/')
            writeline('<center>msra-pick</center>',    _msra30_dir+mat+'/')
            writeline('<center>msra-pick-r</center>',  _msra31_dir+mat+'/')
            writeline('<center>msra-egsr</center>',    _msra20_dir+mat+'/')
            writeline('<center>msra-egsr-r</center>',  _msra21_dir+mat+'/')
            writeline('<center>ours-avg</center>',     _ours00_dir+mat+'/')
            writeline('<center>ours-const</center>',   _ours10_dir+mat+'/')
            writeline('<center>ours-pick</center>',    _ours30_dir+mat+'/')
            writeline('<center>ours-pick-r</center>',  _ours31_dir+mat+'/')
            writeline('<center>ours-egsr</center>',    _ours20_dir+mat+'/')
            writeline('<center>ours-egsr-r</center>',  _ours21_dir+mat+'/')
        # break

    file.write('</table>\n')
    file.write('</body>\n')
    file.write('</html>\n')

file.close()


with open(html_dir + 'real.html', 'w') as file:

    file.write('<!DOCTYPE html>\n')
    file.write('<html>\n')
    file.write('<body>\n')
    file.write('<h1 align="center">SVBRDF</h1>\n')
    file.write('<table align="center">\n')

    file.write('  <tr>\n')
    file.write('    <td colspan="10">All images are in gamma 2.2 space, except "Normal"</td>\n')
    file.write('  </tr>\n')

    mat_list = gyListNames(ref_dir+'real_*')
    for idx, mat in enumerate(mat_list):
        if 1 == 1:
            print(mat)
            file.write('  <tr>\n')
            file.write('    <td><center><b>%s</b></center></td>\n' % mat)
            file.write('  </tr>\n')

            writeline('<center>ref</center>',             _ref_dir+mat+'/')
            writeline('<center>egsr</center>',          _egsr00_dir+mat+'/')
            writeline('<center>msra-avg</center>',     _msra00_dir+mat+'/')
            writeline('<center>msra-const</center>',   _msra10_dir+mat+'/')
            writeline('<center>msra-pick</center>',    _msra30_dir+mat+'/')
            writeline('<center>msra-pick-r</center>',  _msra31_dir+mat+'/')
            writeline('<center>msra-egsr</center>',    _msra20_dir+mat+'/')
            writeline('<center>msra-egsr-r</center>',  _msra21_dir+mat+'/')
            writeline('<center>ours-avg</center>',     _ours00_dir+mat+'/')
            writeline('<center>ours-const</center>',   _ours10_dir+mat+'/')
            writeline('<center>ours-pick</center>',    _ours30_dir+mat+'/')
            writeline('<center>ours-pick-r</center>',  _ours31_dir+mat+'/')
            writeline('<center>ours-egsr</center>',    _ours20_dir+mat+'/')
            writeline('<center>ours-egsr-r</center>',  _ours21_dir+mat+'/')
        # break


    file.write('</table>\n')
    file.write('</body>\n')
    file.write('</html>\n')

file.close()