#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain1s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain1s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain1s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --rs 1000 --re 1600

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight1s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight1s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight1s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --rs 1000 --re 1600

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat1s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat1s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat1s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --rs 1000 --re 1600

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain2s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain2s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmain2s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --rs 1000 --re 1600

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight2s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight2s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflight2s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --rs 1000 --re 1600

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat2s0to500 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --rs 0 --re 500
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat2s500to1000 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --rs 500 --re 1000
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmat2s1000to1600 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --rs 1000 --re 1600

python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainalbedo --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --mode 1 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainnormal --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --mode 2 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainrough --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --mode 3 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainmask --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --mode 4 --rs 0 --re 1600
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmaindepth --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile main --mode 5 --rs 0 --re 1600

python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatalbedo --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --mode 1 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatnormal --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --mode 2 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatrough --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffMat --mode 3 --rs 0 --re 1600 --forceOutput

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightmask --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --mode 4 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightalbedo --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --mode 1 --rs 0 --re 1600 --forceOutput
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightrough --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml --xmlFile mainDiffLight --mode 3 --rs 0 --re 1600 --forceOutput

python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainalbedo1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --mode 1 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainnormal1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --mode 2 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainrough1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --mode 3 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmainmask1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --mode 4 --rs 0 --re 1600
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhlmaindepth1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile main --mode 5 --rs 0 --re 1600

python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatalbedo1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --mode 1 --rs 0 --re 1600 --forceOutput
#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatnormal1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --mode 2 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldiffmatrough1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffMat --mode 3 --rs 0 --re 1600 --forceOutput

#python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightmask1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --mode 4 --rs 0 --re 1600
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightalbedo1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --mode 1 --rs 0 --re 1600 --forceOutput
python2 deploy.py --name k8s-haosu-19.sdsc.optiputer.net --name zhldifflightrough1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderImg.py --xmlRoot xml1 --xmlFile mainDiffLight --mode 3 --rs 0 --re 1600 --forceOutput
