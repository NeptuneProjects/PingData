from funcs_common import *
from funcs_bedpick import *
# from rasterio.merge import merge
# from rasterio.enums import Resampling
import gdal

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

import itertools

class portstarObj(object):
    '''

    '''
    #=======================================================================
    def __init__(self, objs):
        '''
        '''
        for obj in objs:
            if obj.beamName == 'ss_port':
                self.port = obj
            elif obj.beamName == 'ss_star':
                self.star = obj
            else:
                print("Object is unknown...")
        return

    ############################################################################
    # Get Port/Star Sonar Intensity and Merge                                  #
    ############################################################################

    #=======================================================================
    def _getPortStarScanChunk(self, i):
        # Load sonar intensity into memory
        self.port._getScanChunkSingle(i)
        self.star._getScanChunkSingle(i)

        # Rotate and merge arrays into one
        portSon = np.rot90(self.port.sonDat, k=1, axes=(1,0))
        portSon = np.flipud(portSon)
        starSon = np.rot90(self.star.sonDat, k=1, axes=(0,1))

        # Ensure portSon/starSon same length.
        if (portSon.shape[0] != starSon.shape[0]):
            pL = portSon.shape[0]
            sL = starSon.shape[0]
            # Add rows to shortest array from longest array
            if (pL > sL):
                slice = np.fliplr(portSon[(sL-pL):,:])
                starSon = np.append(starSon, slice, axis=0)

            else:
                slice = np.fliplr(starSon[(pL-sL):, :])
                portSon = np.append(portSon, slice, axis=0)


        # Concatenate arrays column-wise
        mergeSon = np.concatenate((portSon, starSon), axis = 1)
        del self.port.sonDat, self.star.sonDat

        self.mergeSon = mergeSon
        return self

    ############################################################################
    # Mosaic                                                                   #
    ############################################################################

    #=======================================================================
    def _createMosaic(self, mosaic, overview):
        maxChunk = 100
        self.imgsToMosaic = []

        if self.port.rect_wcp:
            portPath = os.path.join(self.port.outDir, 'rect_wcp')
            port = sorted(glob(os.path.join(portPath, '*.tif')))

            starPath = os.path.join(self.star.outDir, 'rect_wcp')
            star = sorted(glob(os.path.join(starPath, '*.tif')))

            if len(port) > maxChunk:
                port = [port[i:i+maxChunk] for i in range(0, len(port), maxChunk)]
                star = [star[i:i+maxChunk] for i in range(0, len(star), maxChunk)]
                wcpToMosaic = [list(itertools.chain(*i)) for i in zip(port, star)]
            else:
                wcpToMosaic = [port + star]

        if self.port.rect_wcr:
            portPath = os.path.join(self.port.outDir, 'rect_wcr')
            port = sorted(glob(os.path.join(portPath, '*.tif')))

            starPath = os.path.join(self.star.outDir, 'rect_wcr')
            star = sorted(glob(os.path.join(starPath, '*.tif')))

            if len(port) > maxChunk:
                port = [port[i:i+maxChunk] for i in range(0, len(port), maxChunk)]
                star = [star[i:i+maxChunk] for i in range(0, len(star), maxChunk)]
                srcToMosaic = [list(itertools.chain(*i)) for i in zip(port, star)]
            else:
                srcToMosaic = [port + star]

        if mosaic == 1:
            if self.port.rect_wcp:
                self._mosaicGtiff(overview, wcpToMosaic)
            if self.port.rect_wcr:
                self._mosaicGtiff(overview, srcToMosaic)
        elif mosaic == 2:
            if self.port.rect_wcp:
                self._mosaicVRT(overview, wcpToMosaic)
            if self.port.rect_wcr:
                self._mosaicVRT(overview, srcToMosaic)


    #=======================================================================
    def _mosaicGtiff(self, overview, imgsToMosaic):
        i = 0
        for imgs in imgsToMosaic:

            filePrefix = os.path.split(self.port.projDir)[-1]
            fileSuffix = os.path.split(os.path.dirname(imgs[0]))[-1] + '_mosaic_'+str(i)+'.vrt'
            outVRT = os.path.join(self.port.projDir, filePrefix+'_'+fileSuffix)
            outTIF = outVRT.replace('.vrt', '.tif')

            vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
            gdal.BuildVRT(outVRT, imgs, options=vrt_options)

            ds = gdal.Open(outVRT)

            kwargs = {'format': 'GTiff',
                      'creationOptions': ['NUM_THREADS=ALL_CPUS', 'COMPRESS=LZW']
                      }
            gdal.Translate(outTIF, ds, **kwargs)
            if overview:
                dest = gdal.Open(outTIF, 1)
                gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
                dest.BuildOverviews('nearest', [2 ** j for j in range(1,10)])

            os.remove(outVRT)
            i+=1

        return self

    #=======================================================================
    def _mosaicVRT(self, overview, imgsToMosaic):
        i = 0
        for imgs in imgsToMosaic:

            filePrefix = os.path.split(self.port.projDir)[-1]
            fileSuffix = os.path.split(os.path.dirname(imgs[0]))[-1] + '_mosaic_'+str(i)+'.vrt'
            outFile = os.path.join(self.port.projDir, filePrefix+'_'+fileSuffix)

            vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
            gdal.BuildVRT(outFile, imgs, options=vrt_options)

            if overview:
                dest = gdal.Open(outFile)
                gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
                dest.BuildOverviews('nearest', [2 ** j for j in range(1,10)])
            i+=1

    ############################################################################
    # Bedpicking                                                               #
    ############################################################################

    #=======================================================================
    def _findBed(self, segArr):
        '''
        Find bed location in pixels
        '''
        # Find center of array
        H, W = segArr.shape[0], segArr.shape[1] # height (row), width (column)
        C = int(W/2) # center of array

        # Find bed location
        portBed = []
        starBed = []
        for k in range(H):
            pB = np.where(segArr[k, 0:C]==1)[0]
            pB = np.split(pB, np.where(np.diff(pB) != 1)[0]+1)[0][-1]

            portBed.append(C-pB)

            sB = np.where(segArr[k, C:]==1)[0]
            sB = np.split(sB, np.where(np.diff(sB) != 1)[0]+1)[-1][0]
            starBed.append(sB)

        return portBed, starBed

    #=======================================================================
    def _initModel(self):
        '''

        '''
        SEED=42
        np.random.seed(SEED)
        AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

        tf.random.set_seed(SEED)

        print("Version: ", tf.__version__)
        print("Eager mode: ", tf.executing_eagerly())
        print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        USE_GPU = True

        if USE_GPU == True:
           ##use the first available GPU
           os.environ['CUDA_VISIBLE_DEVICES'] = '0' #'1'
        else:
           ## to use the CPU (not recommended):
           os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        #suppress tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Open model configuration file
        with open(self.configfile) as f:
            config = json.load(f)

        globals().update(config)

        # for k in config.keys():
        #     exec(k+'=config["'+k+'"]')

        model =  custom_resunet((TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                        FILTERS,
                        nclasses=[NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        kernel_size=(KERNEL,KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,#0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,#0.0,
                        dropout_type=DROPOUT_TYPE,#"standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,#False,
                        )

        model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [mean_iou, dice_coef])
        model.load_weights(self.weights)

        self.bedpickModel = model

        return self

    #=======================================================================
    def _doPredict(self, model, arr):
        # Read array into a cropped and resized tensor
        ## Compression step from Zheng et al. 2021
        if N_DATA_BANDS<=3:
            image, w, h, bigimage = seg_file2tensor_3band(arr, TARGET_SIZE)

        # Standardize
        image = standardize(image.numpy()).squeeze()

        # Do segmentation on compressed sonogram
        est_label = model.predict(tf.expand_dims(image, 0), batch_size=1).squeeze()

        # Up-sample / rescale to original dimensions
        ## Store prediction for water and bed classes after resize
        E0 = []; E1 = [];

        E0.append(resize(est_label[:,:,0],(w,h), preserve_range=True, clip=True))
        E1.append(resize(est_label[:,:,1],(w,h), preserve_range=True, clip=True))
        del est_label

        e0 = np.average(np.dstack(E0), axis=-1)
        e1 = np.average(np.dstack(E1), axis=-1)
        del E0, E1

        est_label = (e1+(1-e0))/2
        del e0, e1

        thres = threshold_otsu(est_label)

        est_label = (est_label>thres).astype('uint8')

        return est_label

    #=======================================================================
    def _detectDepth(self, method, i):
        '''
        '''

        if method == 1:
            self._depthZheng(i)
        elif method == 2:
            self.port._loadSonMeta()
            self.star._loadSonMeta()
            self._depthThreshold(i)

    #=======================================================================
    def _depthZheng(self, i):
        doFilt = True
        model = self.bedpickModel

        self._getPortStarScanChunk(i)
        mergeSon = self.mergeSon
        # Make 3-band array
        b1 = mergeSon.copy()
        b2 = np.fliplr(b1.copy())
        b3 = np.mean([b1,b2], axis=0)
        del mergeSon

        son3bnd = np.dstack((b1, b2, b3)).astype(np.uint8)
        del b1, b2, b3

        ##########################################
        # Do initial prediction on entire sonogram
        init_label = self._doPredict(model, son3bnd)

        ############################
        # Find pixel location of bed
        portDepPix, starDepPix = self._findBed(init_label)

        ##################################
        # Crop image using initial bedpick
        ## Zheng et al. 2021 method
        ## Once cropped, sonogram is closer in size to model TARGET_SIZE, so
        ## any errors introduced after rescaling prediction will be minimized.

        N = son3bnd.shape[1] # Number of samples per ping (width of non-resized sonogram)
        W = TARGET_SIZE[1] # Width of model output prediction
        Wp = np.add(portDepPix, starDepPix) # Width of water column estimate
        del portDepPix, starDepPix

        # Determine amount of WC to crop (Wwc)
        if (W < (np.max(Wp)+np.min(Wp)) ):
            Wwc = int( (np.max(Wp)+np.min(Wp)-W) / 2 )
        else:
            Wwc = 0

        # Determine amount of bed to crop (Ws)
        if (W > (np.max(Wp)-np.min(Wp)) ):
            Ws = int( (N/2) - (np.max(Wp)+np.min(Wp)+W)/2 )
        else:
            Ws = 0

        # Crop the original sonogram
        C = int(N/2) # Center of sonogram

        ## Port crop
        lC = Ws # Left side crop
        rC = int(C - (Wwc/2)) # Right side crop
        portCrop = son3bnd[:, lC:rC,:]

        ## Starboard crop
        lC = int(C + (Wwc/2)) # Left side crop
        rC = int(N-Ws) # Right side crop
        starCrop = son3bnd[:, lC:rC, :]

        ## Concatenate port & star crop
        sonCrop = np.concatenate((portCrop, starCrop), axis = 1)
        sonCrop = sonCrop.astype(np.uint8)
        del portCrop, starCrop

        #######################
        # Segment cropped image
        crop_label = self._doPredict(model, sonCrop)

        ###################################################
        # Potential filters for removing small artifacts???
        if doFilt:
            crop_label = remove_small_holes(crop_label.astype(bool), 2*N)#2*self.port.nchunk)
            crop_label = remove_small_objects(crop_label, 2*N)#2*self.port.nchunk)
            # # There should be three total regions: port bed, wc, star bed
            # ## These should also be the largest 3 regions
            # regions, num = label(crop_label, return_num=True)
            # if num > 3:
            #     for region in regionprops(regions):
            #         print(region.area)

        #########
        # Recover
        portDepPixCrop, starDepPixCrop = self._findBed(crop_label) # get pixel location of bed

        # add Wwc/2 to get final estimate at original sonogram dimensions
        portDepPixCrop = np.flip( np.asarray(portDepPixCrop) + int(Wwc/2) )
        starDepPixCrop = np.flip( np.asarray(starDepPixCrop) + int(Wwc/2) )

        ########################
        # Store depth detections
        self.portDepDetect[i] = portDepPixCrop
        self.starDepDetect[i] = starDepPixCrop
        del portDepPixCrop, starDepPixCrop


        #*#*#*#*#*#*#*#
        # Plot
        # color map
        class_label_colormap = ['#3366CC','#DC3912']

        color_label = label_to_colors(init_label, son3bnd[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
        imsave(os.path.join(self.port.projDir, str(i)+"initLabel_"+str(i)+".png"), (color_label).astype(np.uint8), check_contrast=False)
        imsave(os.path.join(self.port.projDir, str(i)+"son3bnd_"+str(i)+".png"), (son3bnd).astype(np.uint8), check_contrast=False)
        imsave(os.path.join(self.port.projDir, str(i)+"cropImg_"+str(i)+".png"), (sonCrop).astype(np.uint8), check_contrast=False)

        color_label = label_to_colors(crop_label, sonCrop[:,:,0]==0, alpha=128, colormap=class_label_colormap, color_class_offset=0, do_alpha=False)
        imsave(os.path.join(self.port.projDir, str(i)+"cropLabel_"+str(i)+".png"), (color_label).astype(np.uint8), check_contrast=False)

        del son3bnd, init_label, crop_label, sonCrop
        return self

    #=======================================================================
    def _depthThreshold(self, chunk):
        # Parameters
        window = 10 # For peak removal in bed pick: moving window size
        max_dev = 5 # For peak removal in bed pick: maximum standard deviation
        pix_buf = 50 # Buffer size around min/max Humminbird depth

        portstar = [self.port, self.star]
        for son in portstar:
            # Load sonar intensity, standardize & rescale
            son._getScanChunkSingle(chunk)
            img = son.sonDat
            img = standardize(img, 0, 1, True)[:,:,-1].squeeze()
            W, H = img.shape[1], img.shape[0]

            # Get chunks sonar metadata and instrument depth
            isChunk = son.sonMetaDF['chunk_id']==1
            sonMeta = son.sonMetaDF[isChunk].reset_index()
            acousticBed = round(sonMeta['inst_dep_m'] / sonMeta['pix_m'], 0).astype(int)

            ##################################
            # Step 1 : Acoustic Bedpick Filter
            # Use acoustic bed pick to crop image
            bedMin = max(min(acousticBed)-50, 0)
            bedMax = max(acousticBed)+pix_buf

            cropMask = np.ones((H, W)).astype(int)
            cropMask[:bedMin,:] = 0
            cropMask[bedMax:,:] = 0

            # Mask the image with bed_mask
            imgMasked = img*cropMask

            ###########################
            # Step 2 - Threshold Filter
            # Binary threshold masked image
            imgMasked = gaussian(imgMasked, 3, preserve_range=True) # Do a gaussian blur
            imgMasked[imgMasked==0]=np.nan # Set zero's to nan

            imgBinaryMask = np.zeros((H, W)).astype(bool) # Create array to store thresholded sonar img
            # Iterate over each sonar record
            for i in range(W):
                thresh = max(np.nanmedian(imgMasked[:,i]), np.nanmean(imgMasked[:,i])) # Determine threshold value
                # stdev = np.nanstd(imgMasked[:,i])
                imgBinaryMask[:,i] = imgMasked[:,i] > thresh # Keep only intensities greater than threshold

            # Clean up image binary mask
            imgBinaryMask = remove_small_objects(imgBinaryMask, 2*H)
            imgBinaryMask = remove_small_holes(imgBinaryMask, 2*H)
            imgBinaryMask = np.squeeze(imgBinaryMask[:H,:W])

            ########################################
            # Step 3 - Non-Contiguous region removal
            # Make sure we didn't accidently zero out the last row, which should be bed.
            # If we did, we will fill it back in
            # Try filtering image_binary_mask through labeling regions
            labelImage, num = label(imgBinaryMask, return_num=True)
            allRegions = []

            # Find the lowest/deepest region (this is the bed pixels)
            max_row = 0
            finalRegion = 0
            for region in regionprops(labelImage):
                allRegions.append(region.label)
                minr, minc, maxr, maxc = region.bbox
                # if (maxr > max_row) and (maxc > max_col):
                if (maxr > max_row):
                    max_row = maxr
                    finalRegion = region.label

            # If finalRegion is 0, there is only one region
            if finalRegion == 0:
                finalRegion = 1

            # Zero out undesired regions
            for regionLabel in allRegions:
                if regionLabel != finalRegion:
                    labelImage[labelImage==regionLabel] = 0

            imgBinaryMask = labelImage # Update thresholded image
            imgBinaryMask[imgBinaryMask>0] = 1 # Now set all val's greater than 0 to 1 to create the mask

            # Now fill in above last row filled to make sure no gaps in bed pixels
            lastRow = bedMax
            imgBinaryMask[lastRow] = True
            for i in range(W):
                if imgBinaryMask[lastRow-1,i] == 0:
                    gaps = np.where(imgBinaryMask[:lastRow,i]==0)[0]
                    # Split each gap cluster into it's own array, subset the last one,
                    ## and take top value
                    topOfGap = np.split(gaps, np.where(np.diff(gaps) != 1)[0]+1)[-1][0]
                    imgBinaryMask[topOfGap:lastRow,i] = 1

            # Clean up image binary mask
            imgBinaryMask = imgBinaryMask.astype(bool)
            imgBinaryMask = remove_small_objects(imgBinaryMask, 2*H)
            imgBinaryMask = remove_small_holes(imgBinaryMask, 2*H)
            imgBinaryMask = np.squeeze(imgBinaryMask[:H,:W])

            #############################
            # Step 4 - Water Below Filter
            # Iterate each ping and determine if there is water under the bed.
            # If there is, zero out everything except for the lowest region.
            # Iterate each sonar record
            for i in range(W):
                labelPing, num = label(imgBinaryMask[:,i], return_num=True)
                if num > 1:
                    labelPing[labelPing!=num] = 0
                    labelPing[labelPing>0] = 1
                imgBinaryMask[:,i] = labelPing

            ###################################################
            # Step 5 - Final Bedpick: Locate Bed & Remove Peaks
            # Now relocate bed from image_binary_mask
            bed = []
            for k in range(W):
                try:
                    b = np.where(imgBinaryMask[:,k]==1)[0][0]
                except:
                    b=0
                bed.append(b)
            bed = np.array(bed).astype(np.float32)

            # Interpolate over nan's
            nans, x = np.isnan(bed), lambda z: z.nonzero()[0]
            bed[nans] = np.interp(x(nans), x(~nans), bed[~nans])
            bed = bed.astype(int)

            if son.beamName == 'ss_port':
                self.portDepDetect[chunk] = bed
            elif son.beamName == 'ss_star':
                self.starDepDetect[chunk] = bed

        return self

    #=======================================================================
    def _saveDepth(self, chunks, detectDep, smthDep, adjDep):
        # Load sonar metadata file
        self.port._loadSonMeta()
        portDF = self.port.sonMetaDF
        self.star._loadSonMeta()
        starDF = self.star.sonMetaDF

        if detectDep == 0:
            portInstDepth = portDF['inst_dep_m']
            starInstDepth = starDF['inst_dep_m']

            if smthDep:
                # print("\nSmoothing depth values...")
                portInstDepth = savgol_filter(portInstDepth, 51, 3)
                starInstDepth = savgol_filter(starInstDepth, 51, 3)

            if adjDep != 0:
                # print("\tIncreasing/Decreasing depth values by {} meters...".format(adjBy))
                adjBy = portDF['pix_m'][0]*adjDep
                portInstDepth += adjBy
                starInstDepth += adjBy

            portDF['dep_m'] = portInstDepth
            starDF['dep_m'] = starInstDepth

            portDF['dep_m_Method'] = 'Instrument Depth'
            starDF['dep_m_Method'] = 'Instrument Depth'

            portDF['dep_m_smth'] = smthDep
            starDF['dep_m_smth'] = smthDep

            portDF['dep_m_adjBy'] = str(adjDep) + ' pixels'
            starDF['dep_m_adjBy'] = str(adjDep) + ' pixels'

        elif detectDep > 0:
            # Prepare depth detection dictionaries
            portFinal = []
            starFinal = []
            for i in sorted(chunks):
                portDep = self.portDepDetect[i]
                starDep = self.starDepDetect[i]

                portFinal.extend(portDep)
                starFinal.extend(starDep)

            # Check shapes to ensure they are same length.  If not, slice off extra.
            if len(portFinal) > portDF.shape[0]:
                lenDif = portDF.shape[0] - len(portFinal)
                portFinal = portFinal[:lenDif]

            if len(starFinal) > starDF.shape[0]:
                lenDif = starDF.shape[0] - len(starFinal)
                starFinal = starFinal[:lenDif]

            if smthDep:
                portFinal = savgol_filter(portFinal, 51, 3)
                starFinal = savgol_filter(starFinal, 51, 3)

            portDF['dep_m'] = portFinal * portDF['pix_m']
            starDF['dep_m'] = starFinal * starDF['pix_m']

            if adjDep != 0:
                adjBy = portDF['pix_m'][0]*adjDep
                portDF['dep_m'] += adjBy
                starDF['dep_m'] += adjBy

            if detectDep == 1:
                portDF['dep_m_Method'] = 'Zheng et al. 2021'
                starDF['dep_m_Method'] = 'Zheng et al. 2021'
            elif detectDep == 2:
                portDF['dep_m_Method'] = 'Threshold'
                starDF['dep_m_Method'] = 'Threshold'

            portDF['dep_m_smth'] = smthDep
            starDF['dep_m_smth'] = smthDep

            portDF['dep_m_adjBy'] = str(adjDep) + ' pixels'
            starDF['dep_m_adjBy'] = str(adjDep) + ' pixels'

        # Export to csv
        portDF.to_csv(self.port.sonMetaFile, index=False, float_format='%.14f')
        starDF.to_csv(self.star.sonMetaFile, index=False, float_format='%.14f')

        del portDF, starDF
        return self

    #=======================================================================
    def _plotBedPick(self, i, acousticBed = True, autoBed = True):

        # Load sonar intensity into memory
        self._getPortStarScanChunk(i)
        mergeSon = self.mergeSon

        ####################
        # Prepare depth data
        # Load sonar metadata file
        self.port._loadSonMeta()
        portDF = self.port.sonMetaDF
        portDF = portDF.loc[portDF['chunk_id'] == i, ['inst_dep_m', 'dep_m', 'pix_m']]

        self.star._loadSonMeta()
        starDF = self.star.sonMetaDF
        starDF = starDF.loc[starDF['chunk_id'] == i, ['inst_dep_m', 'dep_m', 'pix_m']]

        # Convert depth in meters to pixels
        portInst = (portDF['inst_dep_m'] / portDF['pix_m']).to_numpy(dtype=np.int, copy=True)
        portAuto = (portDF['dep_m'] / portDF['pix_m']).to_numpy(dtype=np.int, copy=True)

        starInst = (starDF['inst_dep_m'] / starDF['pix_m']).to_numpy(dtype=np.int, copy=True)
        starAuto = (starDF['dep_m'] / starDF['pix_m']).to_numpy(dtype=np.int, copy=True)

        # Ensure port/star same length
        if (portAuto.shape[0] != starAuto.shape[0]):
            pL = portAuto.shape[0]
            sL = starAuto.shape[0]
            # Add rows to shortest array from longest array
            if (pL > sL):
                starAuto = np.append(starAuto, portAuto[(sL-pL):])
                starInst = np.append(starInst, portInst[(sL-pL):])
            else:
                portAuto = np.append(portAuto, starAuto[(pL-sL):])
                portInst = np.append(portInst, starInst[(pL-sL):])

        # Relocate depths relative to horizontal center of image
        c = int(mergeSon.shape[1]/2)

        portInst = c - portInst
        portAuto = c - portAuto

        starInst = c + starInst
        starAuto = c + starAuto

        # maybe flip???
        portInst = np.flip(portInst)
        portAuto = np.flip(portAuto)

        starInst = np.flip(starInst)
        starAuto = np.flip(starAuto)

        #############
        # Export Plot
        y = np.arange(0, mergeSon.shape[0])
        # File name zero padding
        if i < 10:
            addZero = '0000'
        elif i < 100:
            addZero = '000'
        elif i < 1000:
            addZero = '00'
        elif i < 10000:
            addZero = '0'
        else:
            addZero = ''

        outDir = os.path.join(self.port.projDir, 'Bedpick')
        try:
            os.mkdir(outDir)
        except:
            pass

        projName = os.path.split(outDir)[-1]
        outFile = os.path.join(outDir, projName+'_Bedpick_'+addZero+str(i)+'.png')

        plt.imshow(mergeSon, cmap='gray')
        if acousticBed:
            plt.plot(portInst, y, 'y-.', lw=1, label='Acoustic Depth')
            plt.plot(starInst, y, 'y-.', lw=1)
        if autoBed:
            plt.plot(portAuto, y, 'b-.', lw=1, label='Auto Depth')
            plt.plot(starAuto, y, 'b-.', lw=1)

        plt.legend(loc = 'lower right', prop={'size':4}) # create the plot legend
        plt.savefig(outFile, dpi=300, bbox_inches='tight')
        plt.close()

    #=======================================================================
    def _cleanup(self):
        try:
            del self.bedpickModel
        except:
            pass

        try:
            del self.port.sonDat, self.star.sonDat
        except:
            pass

        try:
            del self.port.sonMetaDF, self.star.sonMetaDF
        except:
            pass

        return self
