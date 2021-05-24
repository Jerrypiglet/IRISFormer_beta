import h5py
import numpy as np

def loadH5(imName ): 
    try:
        hf = h5py.File(imName, 'r')
        im = np.array(hf.get('data' ) )
        return im 
    except:
        return None
