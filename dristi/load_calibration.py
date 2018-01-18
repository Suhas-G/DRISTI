import os
import numpy as np
import cv2

class Calibration(object):
    '''
    A class to load all calibration files
    Methods defined:
    @public: load_calibration_files(folder_name)
    @private: _modify_undistort_and_rectify_maps()
    '''
    def __init__(self, size):
        '''
        Initialise all calibration parameters
        Parameters initialised:
        cam_mats, dist_coefs, rot_mat, trans_vec, e_mat, f_mat, rect_trans,
        proj_mats, disp_to_depth_mat, valid_boxes, undistortion_map,
        rectification_map
        @params: size - Size of the image.
        '''
        self.cam_mats = {"left": None, "right": None}
        #: Distortion coefficients (D)
        self.dist_coefs = {"left": None, "right": None}
        #: Rotation matrix (R)
        self.rot_mat = None
        #: Translation vector (T)
        self.trans_vec = None
        #: Essential matrix (E)
        self.e_mat = None
        #: Fundamental matrix (F)
        self.f_mat = None
        #: Rectification transforms (3x3 rectification matrix R1 / R2)
        self.rect_trans = {"left": None, "right": None}
        #: Projection matrices (3x4 projection matrix P1 / P2)
        self.proj_mats = {"left": None, "right": None}
        #: Disparity to depth mapping matrix (4x4 matrix, Q)
        self.disp_to_depth_mat = None
        #: Bounding boxes of valid pixels
        self.valid_boxes = {"left": None, "right": None}
        #: Undistortion maps for remapping
        self.undistortion_map = {"left": None, "right": None}
        #: Rectification maps for remapping
        self.rectification_map = {"left": None, "right": None}
        #: Size of the image
        self.size = size

    def load_calibration_files(self, folder):
        '''
        Description: Load all calibration files and also modify undistortion_maps and
        rectification_maps
        Parameters: folder: name of the calibration folder
        '''
        for key, item in self.__dict__.items():
            if key == 'size':
                continue
            if isinstance(item, dict):
                for side in ("left", "right"):
                    filename = os.path.join(folder,
                                            "{}_{}.npy".format(key, side))
                    self.__dict__[key][side] = np.load(filename)
            else:
                filename = os.path.join(folder, "{}.npy".format(key))
                self.__dict__[key] = np.load(filename)

        self._modify_undistort_and_rectify_maps()


    def _modify_undistort_and_rectify_maps(self):
        '''Description: Modification necessary to obtain proper disparity
        Important - Deviates from standard documentation present in OpenCV
        '''
        (self.undistortion_map['left'],
        self.rectification_map['left']) = cv2.initUndistortRectifyMap(
                                self.cam_mats['left'],
                                self.dist_coefs['left'],
                                self.rect_trans['left'],
                                self.proj_mats['left'], self.size,
                                cv2.CV_32FC1)
        (self.undistortion_map['right'],
         self.rectification_map['right']) = cv2.initUndistortRectifyMap(
                                    self.cam_mats['right'],
                                    self.dist_coefs['right'],
                                    self.rect_trans['right'],
                                    self.proj_mats['right'], self.size,
                                    cv2.CV_32FC1)
