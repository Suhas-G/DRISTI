import os
import sys
from threading import Thread

import cv2



class CaptureImage(object):
    '''A class to capture images from webcam in a separate thread. It allows
    access to only the recent most image pairs.
    Methods defined here:
    @public: start()
             stop()
             load_images()
    @private: _capture_frames()
    '''
    def __init__(self, LEFT_CAM, RIGHT_CAM):
        '''
        Description: Initialise 2 webcams to capture images. Take 20 images to be discared,
        so that a small amount of time is given for webcams to adjust
        Parameters: LEFT_CAM - Port number corresponding to left webcam
                 RIGHT_CAM - Port number corresponding to right webcam
        '''
        self.left_capture = cv2.VideoCapture(LEFT_CAM)
        self.right_capture = cv2.VideoCapture(RIGHT_CAM)

        if not self.left_capture.isOpened():
            self.left_capture.open()
        if not self.right_capture.isOpened():
            self.right_capture.open()

        self.stopped = False

        for _ in range(20):
            ret, self.left_frame = self.left_capture.read()
            ret, self.right_frame = self.right_capture.read()

    def start(self):
        '''Description: Start a thread to capture images
        '''
        self.t = Thread(target= self._capture_frames)
        self.t.setDaemon(True)
        self.t.start()

    def _capture_frames(self):
        '''Description: A method for thread to loop while capturing left and right images
        from webcams. The recent most image pairs are stored.
        '''
        while True:
            if self.stopped :
                self.t.stop()
                self.left_capture.close()
                self.right_capture.close()
                return
            ret, self.left_frame = self.left_capture.read()
            ret, self.right_frame = self.right_capture.read()


    def load_images(self):
        '''Description: Returns the recent most left and right image pairs.
        '''
        return self.left_frame, self.right_frame

    def stop(self):
        '''Description: Assign a thread termination indicating variable to True.
        '''
        self.stopped = True


class ReadImages(object):
    '''
    A class to load left and right image pairs from the given folder.
    Methods defined here:
    @public: load_images()
             start()
             stop()
    '''
    def __init__(self, folder):
        '''Description: Initiales the left and right image file names present in folder.
        Parameters: folder - Name of the folder which contains images in the
                          format left<sequence_number> and
                          right<sequence_number>.
        '''
        left_images = [file for file in os.listdir(folder) \
                                if file.startswith('left')]
        right_images = [file for file in os.listdir(folder) \
                                if file.startswith('right')]
        self.images_folder = []
        for file in left_images:
            string = 'right'+ file[4:]
            if string in right_images:
                self.images_folder.append(os.path.join(folder, file))
                self.images_folder.append(os.path.join(folder, string))

    def load_images(self):
        '''Description: Loads images 2 at a time corresponding to left and right images.
        '''
        if self.images_folder:
            left_frame, right_frame = [cv2.imread(file) \
                                        for file in self.images_folder[:2]]
            self.images_folder = self.images_folder[2:]

            return left_frame, right_frame

        else:
            sys.exit()

    def start(self):
        '''Description: A method just to discard initial 3 image pairs.
        '''
        self.images_folder = self.images_folder[6:]

    def stop(self):
        '''Description: Dummy method for compatibility.
        '''
        return
