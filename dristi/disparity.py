import cv2
import numpy as np
# from multiprocessing import process


class DisparityCreator(object):
    '''
    A class to handle disparity map generation
    Parameters:
    Lambda: Lambda parameter for WLS filter
    sigma: SigmaColor value for WLS filter
    Methods:
    get_disparity
    '''
    def __init__(self, Lambda, sigma):
        '''Initialise stereo matcher instances for left and right images.
            Use WLS filter to remove occlusion and noise in disparity map.
        '''
        self.left_matcher = cv2.StereoSGBM_create(
                                minDisparity = 0,
                                numDisparities = 64,
                                blockSize = 3,
                                P1 = 73,
                                P2 = 2600,
                                disp12MaxDiff = 43, #initial 43
                                uniquenessRatio = 10,
                                speckleRange = 1,
                                preFilterCap = 0,
                                speckleWindowSize = 10,
                                mode = 1
                                )

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter.setLambda(Lambda)
        self.wls_filter.setSigmaColor(sigma)

    def get_disparity(self, left_img, right_img):
        '''Get disparity map for corresponding left and right images.
        Parameters:
        left_img: Left image
        right_img: Right image
        '''
        left_disparity = self.left_matcher.compute(left_img, right_img)
        right_disparity = self.right_matcher.compute(right_img, left_img)
        filtered_disparity = self.wls_filter.filter(left_disparity, left_img, None, right_disparity)

        filtered_disparity[filtered_disparity > 1008] = 1008
        filtered_disparity[filtered_disparity < -16] = -16
        filtered_disparity = (((filtered_disparity + 16)/ 1008) * 255).astype(np.uint8)

        kernel = np.ones((3,3),np.uint8)
        disparity = cv2.morphologyEx(filtered_disparity,cv2.MORPH_OPEN,kernel, iterations = 2)
        coloured_disparity = cv2.applyColorMap((disparity).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imshow('disparity', coloured_disparity)
        return disparity
