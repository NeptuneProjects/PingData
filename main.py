"""
Part of PING Mapper software

Co-Developed by Cameron S. Bodine and Dr. Daniel Buscombe

This software builds upon PyHum software,
originally developed by Dr. Daniel Buscombe

https://github.com/dbuscombe-usgs/PyHum

"""

from common_funcs import *
from pj_readFiles import read_master_func
from pj_rectify import rectify_master_func

import time
start_time = time.time()

#============================================
if __name__ == "__main__":

    H = []; S = []; P = []

    keep_going = True

    while keep_going is True:

        # Path to data/output
        humFile = '.\\exampleData\\test.DAT'
        sonPath = '.\\exampleData\\test'
        projDir = '.\\procData\\PINGMapperTest'

        H.append(humFile)

        sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))
        S.append(sonFiles)
        print(sonFiles)

        P.append(projDir)

        keep_going = False

    #################
    # User Parameters
    t = 10 #Temperature in Celsius
    nchunk = 500 #Number of pings per chunk
    wcp = True #Export tiles with water column present
    src = True #Export Tiles with water column removed/slant range corrected
    detectDepth = False #True==Automatically detect depth; False==Use Humminbird depth
    smthDep = True #Smooth depth before water column removal

    rect_wcp = True #Export rectified tiles with water column present
    rect_src = True #Export rectified tiles with water column removed/slant range corrected

    #==================================================
    t = float(t)/10
    print('\n===========================================')
    print('===========================================')
    print('***** READING *****')
    for k in range(len(H)):
        print("working on "+P[k])
        read_master_func(S[k], H[k], P[k], t, nchunk, wcp, src, detectDepth, smthDep)

    #==================================================
    if rect_wcp or rect_src:
        print('\n===========================================')
        print('===========================================')
        print('***** RECTIFYING *****')
        for k in range(len(H)):
            print("working on "+P[k])
            rectify_master_func(S[k], H[k], P[k], nchunk, detectDepth, smthDep, rect_wcp, rect_src)

    keep_going = False
print("Total Processing Time: ",round((time.time() - start_time),ndigits=2))
