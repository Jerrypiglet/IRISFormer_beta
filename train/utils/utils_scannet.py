import numpy as np

def read_ExtM_from_txt(fpath_txt):
    '''
    Read the external matrix from txt file. 
    The txt file for the exterminal matrix is got from 
    the sens file loader 

    return:
    ExtM - world to cam: p_c = ExtM @ p_w
           cam to world : p_w = ExM.inverse() @ p_c

    '''
    ExtM = np.eye(4)
    with open(fpath_txt, 'r') as f:
        content = f.readlines()
    content = [ x.strip() for x in content]
    
    for ir, row in enumerate(ExtM):
        row_content = content[ir].split()
        row = np.asarray([ float(x) for x in row_content ])
        ExtM[ir, :] = row
    ExtM = np.linalg.inv(ExtM)
    return ExtM
