# Part of PING-Mapper software
#
# Co-Developed by Cameron S. Bodine and Dr. Daniel Buscombe
#
# Inspired by PyHum: https://github.com/dbuscombe-usgs/PyHum
#
# MIT License
#
# Copyright (c) 2022-23 Cameron S. Bodine
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
sys.path.insert(0, 'src')

from funcs_common import *
from main_readFiles import read_master_func
from main_rectify import rectify_master_func
from main_mapSubstrate import map_master_func

import time
import datetime

# Get processing script's dir so we can save it to file
scriptDir = os.getcwd()
# generate filename with timestring
copied_script_name = os.path.basename(__file__)+'_'+time.strftime("%Y-%m-%d_%H%M")+'.py'
script = os.path.join(scriptDir, os.path.basename(__file__))


# inDir = r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20220302_USM1'
# outDir = r'Z:\PINGMapper_Outputs\RKM559_Sill'

# inDir = r'G:\Shared drives\MAISRC_zebra mussel project\Phase III\SideScanMapping\HumminBird Data\StC'
# outDir = r'Z:\MAISRC_zebra mussel project\Phase III\SideScanMapping\Processed_Data_csb\AutoSubstrate'

# inDir = r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data'
# outDir = r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data_Processed\Mosaics\AutoDepth'

# inDir = r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data'
# outDir = r'E:\SynologyDrive\Modeling\00_forLabeling\SpdCor_EGN_AllGSRecordings'

inDir = '/mnt/md0/SynologyDrive/GulfSturgeonProject/SSS_Data'
outDir = '/mnt/md0/SynologyDrive/GulfSturgeonProject/SSS_Data_Processed/EGN'

# inDir = r'/mnt/md0/SynologyDrive/GulfSturgeonProject/SSS_Data'
# outDir = r'/mnt/md0/SynologyDrive/Modeling/00_forLabeling/SpdCor_EGN_AllGSRecordings'

inDir = os.path.normpath(inDir)
outDir = os.path.normpath(outDir)

# # For McAulay
# outDir = r'E:\SynologyDrive\RFI\20220324_McAulayJaunsen_USM'
# outDir = os.path.normpath(outDir)

#################
# User Parameters

# *** IMPORTANT ****: Overwriting project and outputs
# Export Mode: project_mode
## 0==NEW PROJECT: Create a new project. [DEFAULT]
##      If project already exists, program will exit without any project changes.
##
## 1==UPDATE PROJECT: Export additional datasets to existing project.
##      Use this mode to update an existing project.
##      If selected datasets were previously exported, they will be overwritten.
##      To ensure datasets aren't overwritten, deselect them below.
##      If project does not exist, program will exit without any project changes.
##
## 2==MAYHEM MODE: Create new project, regardless of previous project state.
##      If project exists, it will be DELETED and reprocessed.
##      If project does not exist, a new project will be created.
project_mode = 1

# General Parameters
tempC = 10 #Temperature in Celsius
nchunk = 500 #Number of pings per chunk
exportUnknown = False #Option to export Unknown ping metadata
fixNoDat = True # Locate and flag missing pings; add NoData to exported imagery.
threadCnt = -4 #Number of compute threads to use; 0==All threads; <0==(Total threads + threadCnt); >0==Threads to use up to total threads


# Position Corrections
## Provide an x and y offset to account for position offset between
## control head (or external GPS) and transducer.
## Origin (0,0) is the location of control head (or external GPS)
## X-axis runs from bow (fore, or front) to stern (aft, or rear) with positive offset towards the bow, negative towards stern
## Y-axis runs from portside (left) to starboard (right), with negative values towards the portside, positive towards starboard
## Z-offsets can be provided with `adjDep` below.
x_offset = 0.0 # [meters]
y_offset = 0.0 # [meters]


# Sonar Intensity Corrections
egn = True
egn_stretch = 1 # 0==Min-Max; 1==% Clip; 2==Standard deviation
egn_stretch_factor = 0.5 # If % Clip, the percent of histogram tails to clip (1.0 == 1%);
                         ## If std, the number of standard deviations to retain


# Sonogram Exports
tileFile = '.jpg' # Img format for plots and sonogram exports
wcp = False #Export tiles with water column present: 0==False; 1==True, side scan channels only; 2==True, all available channels.
wcr = False #Export Tiles with water column removed (and slant range corrected): 0==False; 1==True, side scan channels only; 2==True, all available channels.


# Speed corrected sonogram Exports
lbl_set = 0 # Export images for labeling: 0==False; 1==True, keep water column & shadows; 2==True, remove water column & shadows (based on maxCrop)
spdCor = 1 # Speed correction: 0==No Speed Correction; 1==Stretch by GPS distance; !=1 or !=0 == Stretch factor.
maxCrop = False # True==Ping-wise crop; False==Crop tile to max range.


# Depth Detection and Shadow Removal Parameters
remShadow = 2 # 0==Leave Shadows; 1==Remove all shadows; 2==Remove only bank shadows
detectDep = 1 # 0==Use Humminbird depth; 1==Auto detect depth w/ Zheng et al. 2021;
## 2==Auto detect depth w/ Thresholding

smthDep = False #Smooth depth before water column removal
adjDep = 10 #Aditional depth adjustment (in pixels) for water column removaL
pltBedPick = False #Plot bedpick on sonogram


# Rectification Sonar Map Exports
rect_wcp = False #Export rectified tiles with water column present
rect_wcr = True #Export rectified tiles with water column removed/slant range corrected


# Substrate Mapping
pred_sub = 0 # Automatically predict substrates and save to npz: 0==False; 1==True, SegFormer Model
# pred_stride = 250 # Stride size, in pings, for moving window prediction: 0==No moving window
pltSubClass = False # Export plots of substrate classification and predictions
map_sub = False # Export substrate maps (as rasters): 0==False; 1==True. Requires substrate predictions saved to npz.
export_poly = False # Convert substrate maps to shapefile: map_sub must be > 0 or raster maps previously exported
map_predict = 0 #Export rectified tiles of the model predictions: 0==False; 1==Probabilities; 2==Logits. Requires substrate predictions saved to npz.
map_class_method = 'max' # 'max' only current option. Take argmax of substrate predictions to get final classification.


# Mosaic Exports
pix_res = 0.25 # Pixel resolution [meters]: 0 = Default (~0.02 m). ONLY APPLIES TO MOSAICS
mosaic_nchunk = 0 # Number of chunks per mosaic: 0=All chunks. Specifying a value >0 generates multiple mosaics if number of chunks exceeds mosaic_nchunk.
mosaic = 0 #Export sonar mosaic; 0==Don't Mosaic; 1==Do Mosaic - GTiff; 2==Do Mosaic - VRT
map_mosaic = 1 #Export substrate mosaic; 0==Don't Mosaic; 1==Do Mosaic - GTiff; 2==Do Mosaic - VRT



#################
#################

#============================================


# Find all DAT and SON files in all subdirectories of inDir
os.path.normpath(inDir)
os.path.normpath(outDir)

inFiles=[]
for root, dirs, files in os.walk(inDir):
    for file in files:
        if file.endswith('.DAT'):
            inFiles.append(os.path.join(root, file))




# # For McAulay
# inFiles = [r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210511_FWSB1\289_244_Rec00001.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210511_FWSC1\289_244_Rec00004.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210303_FWSA1\489_451_Rec00003.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210303_FWSA1\451_438_Rec00004.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210303_FWSB1\453_451_Rec00005.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210303_FWSB1\451_438_Rec00006.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210304_FWSA1\393_363_Rec00007.DAT',
#            r'E:\SynologyDrive\GulfSturgeonProject\SSS_Data\Pearl\Pearl\PRL_20210304_FWSB1\395_365_Rec00009.DAT']






inFiles = sorted(inFiles, reverse=False)

for i, f in enumerate(inFiles):
    print(i, ":", f)

errorRecording = []
for i, datFile in enumerate(inFiles):
    # try:
    start_time = time.time()

    # inPath = os.path.dirname(datFile)
    # humFile = datFile
    # recName = os.path.basename(humFile).split('.')[0]
    # sonPath = os.path.join(inDir, recName)
    # sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))
    #
    # projDir = os.path.join(outDir, recName)

    humFile = datFile
    sonPath = datFile.split('.')[0]
    sonFiles = sorted(glob(sonPath+os.sep+'*.SON'))

    # recName = os.path.basename(humFile).split('.')[0]
    # dateBoat = os.path.dirname(humFile).split(os.sep)[-1]
    # river = os.path.dirname(humFile).split(os.sep)[-2]
    # projName = river+'_'+dateBoat+'_'+recName

    # # StC
    # recName = os.path.basename(humFile).split('.')[0]
    # location = os.path.dirname(humFile).split(os.sep)[-1]
    # projName = location+'_'+recName

    # # WBL
    # recName = os.path.basename(humFile).split('.')[0]
    # date = os.path.dirname(humFile).split(os.sep)[-1]
    # date = date.replace('-', '')
    # location = os.path.dirname(humFile).split(os.sep)[-2]
    # projName = location+'_'+date+'_'+recName

    # GS Exports
    recName = os.path.basename(humFile.split('.')[0])
    try:
        upRKM = recName.split('_')[0]
        dnRKM = recName.split('_')[1]
        recNum = recName.split('_')[2]
    except:
        upRKM = 'XXX'
        dnRKM = 'XXX'
        recNum = recName

    riverDate = os.path.dirname(humFile).split(os.sep)[-1]
    river = riverDate.split('_')[0]
    date = riverDate.split('_')[1]
    unit = riverDate.split('_')[2]

    projName = river+'_'+upRKM+'_'+dnRKM+'_'+date+'_'+unit+'_'+recNum
    print(projName)

    projDir = os.path.join(outDir, projName)

    # Store params in a dictionary
    params = {
        'project_mode':project_mode,
        'script':[script, copied_script_name],
        'humFile':humFile,
        'sonFiles':sonFiles,
        'projDir':projDir,
        'tempC':tempC,
        'nchunk':nchunk,
        'exportUnknown':exportUnknown,
        'fixNoDat':fixNoDat,
        'threadCnt':threadCnt,
        'x_offset':x_offset,
        'y_offset':y_offset,
        'egn':egn,
        'egn_stretch':egn_stretch,
        'egn_stretch_factor':egn_stretch_factor,
        'tileFile':tileFile,
        'wcp':wcp,
        'wcr':wcr,
        'lbl_set':lbl_set,
        'spdCor':spdCor,
        'maxCrop':maxCrop,
        'USE_GPU':False,
        'remShadow':remShadow,
        'detectDep':detectDep,
        'smthDep':smthDep,
        'adjDep':adjDep,
        'pltBedPick':pltBedPick,
        'rect_wcp':rect_wcp,
        'rect_wcr':rect_wcr,
        'pred_sub':pred_sub,
        'map_sub':map_sub,
        'export_poly':export_poly,
        'map_predict':map_predict,
        'pltSubClass':pltSubClass,
        'map_class_method':map_class_method,
        'pix_res':pix_res,
        'mosaic_nchunk':mosaic_nchunk,
        'mosaic':mosaic,
        'map_mosaic':map_mosaic
        }

    # try:
    print('sonPath',sonPath)
    print('\n\n\n+++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++')
    print('***** Working On *****')
    print('Index:', i)
    print('Output Director:', projDir)
    print('Input File:', humFile)
    print('Start Time: ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))

    print('\n===========================================')
    print('===========================================')
    print('***** READING *****')
    read_master_func(**params)

    if rect_wcp or rect_wcr:
        print('\n===========================================')
        print('===========================================')
        print('***** RECTIFYING *****')
        rectify_master_func(**params)

    #==================================================
    if pred_sub or map_sub or export_poly or map_predict or pltSubClass or map_mosaic:
        print('\n===========================================')
        print('===========================================')
        print('***** MAPPING SUBSTRATE *****')
        print("working on "+projDir)
        map_master_func(**params)

    # except:
    #     print('Could not process:', datFile)
    #     errorRecording.append(projDir)

    gc.collect()
    print("\n\nTotal Processing Time: ",datetime.timedelta(seconds = round(time.time() - start_time, ndigits=0)), '\n\n\n')
    # sys.exit()

if len(errorRecording) > 0:
    print('\n\nUnable to process the following:')
    for d in errorRecording:
        print('\n',d)
