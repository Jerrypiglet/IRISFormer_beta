#python2 deploy.py --name zmain1 --cpu 1 --memr 1G --meml 1G --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml --xmlFile main --rs 0 --re 1600
python2 deploy.py --name zdifflight1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml --xmlFile mainDiffLight --rs 0 --re 1600
python2 deploy.py --name zdiffmat1 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml --xmlFile mainDiffMat --rs 0 --re 1600
python2 deploy.py --name zmain2 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml1 --xmlFile main --rs 0 --re 1600
python2 deploy.py --name zdifflight2 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml1 --xmlFile mainDiffLight --rs 0 --re 1600
python2 deploy.py --name zdiffmat2 --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python renderLight.py --xmlRoot xml1 --xmlFile mainDiffMat --rs 0 --re 1600
