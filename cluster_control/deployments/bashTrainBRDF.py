python2 deploy.py --ngpus 2 --name zBRDF --repo git@github.com:lzqsd/empty.git --key siggraphasia20  python trainLight.py --file trainBRDF.py --flags "--cuda --deviceIds 0 1"
