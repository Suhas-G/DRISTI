'''
Description: A script to run laptop as server for collecting images from a Raspberry Pi client.
            It also computes the disparity map and returns appropriate control signals.
'''


import sys, struct, os, pickle, socket

import numpy as np
import cv2

from load_calibration import Calibration
from disparity import DisparityCreator

# Mention the folder which contains calibration parameters
CALIBRATION_FOLDER = 'calibration_parameters'

# Defined based on the resolution of webcam
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Decided empirically after looking at the histogram of disparity
NEAREST_LOWER_THRESH = 190
NEAREST_HIGHER_THRESH = 255
MIDDLE_LOWER_THRESH = 150
MIDDLE_HIGHER_THRESH = 180


def initialise_connection():
    '''
    Description: A function to initialise a streaming connection, using localhost, port 8000.
    Listens to raspberry pi client and accepts the connection.
    Returns:
    connection: Connection object to be used for all further communication.
    server_socket: The server socket
    '''
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 8000))
    print('Connection initiated...')
    server_socket.listen()
    connection, address = server_socket.accept()
    print('Connection accepted...')
    return connection, server_socket

def recieve(connection):
    '''
    Description: A simple protocol for receiving images.
    First obtain the size of images, then receive a single concatenated image, which is later split appropriately
    Parameters:
    connection: Connection object received from initialisation function.
    Returns:
    left_img, right_img : Left and right images captured from webcam connected to Raspberry Pi client.
    '''
    data_length = struct.unpack('<L', connection.recv(struct.calcsize('<L')))[0]
    if not data_length:
        sys.exit()
    data = b''
    length = 0
    while length < data_length:
        chunk = connection.recv(int(data_length/2))
        length += len(chunk)
        data += chunk
    data = np.fromstring(data, dtype = np.uint8)
    data = np.reshape(data, (-1, IMAGE_WIDTH * 2))

    left_img, right_img = np.array_split(data, 2, axis = 1)

    return left_img, right_img

def send(connection, signal):
    '''Description: A simple protocol to send control signals.
    First send the size of the signal, then send the serialised signal.
    Parameters:
    connection: Connection object received from initialisation function.
    signal: The signal to be sent to client.
    '''
    connection.send(struct.pack('<I', sys.getsizeof(signal)))
    connection.send(pickle.dumps(signal))

def preprocess(calibration, left_img, right_img):
    '''Description: Preprocess the images to remove distortion and rectify them.
    Parameters:
    calibration: Calibration object containing all undistortion maps.
    left_img: Left image received.
    right_img: Right image received.
    '''
    undistorted_left = cv2.remap(left_img,calibration.undistortion_map['left'],
                                calibration.rectification_map['left'],
                                cv2.INTER_LINEAR)
    undistorted_right = cv2.remap(right_img,calibration.undistortion_map['right'],
                                 calibration.rectification_map['right'],
                                 cv2.INTER_LINEAR)


    return undistorted_left,undistorted_right

def apply_thresholds(disparity):
    '''Description: Apply thresholds to grayscale image to segment objects present at 2 proximity levels.
    Parameters:
    disparity: disparity map
    Return:
    middle: Image with objects present at mid-level proximity
    nearest: Image with objects of nearest proximity
    '''
    nearest = np.full(disparity.shape, 255, dtype = np.float32)
    middle = np.full(disparity.shape, 255, dtype=np.float32)

    nearest[disparity > NEAREST_HIGHER_THRESH] = 0
    nearest[disparity < NEAREST_LOWER_THRESH] = 0

    middle[disparity > MIDDLE_HIGHER_THRESH] = 0
    middle[disparity < MIDDLE_LOWER_THRESH] = 0

    kernel = np.ones((5,5),np.uint8)
    nearest = cv2.morphologyEx(nearest, cv2.MORPH_OPEN, kernel)
    nearest = cv2.morphologyEx(nearest, cv2.MORPH_CLOSE, kernel)

    middle = cv2.morphologyEx(middle, cv2.MORPH_OPEN, kernel)
    middle = cv2.morphologyEx(middle, cv2.MORPH_CLOSE, kernel)

    return nearest, middle

def find_centroid(contour):
    '''Description: Finds centroid of contour.
    Parameters:
    contour: The contour to find the centroid of.
    Return:
    (cx, cy): x, y tuple of centroid if centroid is present, else None
    '''
    M = cv2.moments(contour)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)
    except:
        return None

def find_locations(img):
    '''Description: Finding contours and their centroids of image portion.
    Parameters:
    img : binary image portion
    Return:
    locations: list of centroids of contour regions if significantly big.
    '''
    contour_img, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours
                if cv2.contourArea(contour) >
                (img.shape[0]*img.shape[1]/3)] # if countour area is > (1/3)rd of img portion
    locations = [find_centroid(contour) for contour in contours]
    locations = [loc for loc in locations if not loc == None]

    return locations


def apply_segmentation(nearest, middle):
    '''Description: Apply segmentation to nearest and middle images after dividing each into 3 parts.
    Find appropriate control signal.
    Parameters:
    nearest: Image with objects of nearest proximity
    middle: Image with objects present at mid-level proximity
    Return:
    signal: Control signal.
    '''
    signal = {'nearest': [], 'middle':[]}
    nearest_parts = np.array_split(nearest, 3, axis = 1)

    for each in nearest_parts:
        loc = find_locations(each)
        if loc:
            signal['nearest'].append(1)
        else:
            signal['nearest'].append(0)

    middle_parts = np.array_split(middle, 4, axis = 1)
    for each in middle_parts:
        loc = find_locations(each)
        if loc:
            signal['middle'].append(1)
        else:
            signal['middle'].append(0)

    return signal




def main():
    calibration = Calibration((IMAGE_WIDTH, IMAGE_HEIGHT))
    calibration.load_calibration_files(CALIBRATION_FOLDER)
    
    disparity_handler = DisparityCreator(16000, 7)
    connection, server_socket = initialise_connection()

    while True:
        left_img, right_img = recieve(connection)
        left_img, right_img = preprocess(calibration, left_img, right_img)
        
        data = [left_img, right_img]
        for line in range(0, int(data[0].shape[0] / 20)):
            data[0][line * 20, :] = 255
            data[1][line * 20, :] = 255


        cv2.imshow('frames', np.hstack(data))

        disparity = disparity_handler.get_disparity(left_img, right_img)
        cv2.imshow('disparity', cv2.applyColorMap(disparity, cv2.COLORMAP_JET))
        nearest, middle = apply_thresholds(disparity)
        
        signal = apply_segmentation(nearest, middle)
        send(connection, pickle.dumps(signal))
        cv2.waitKey(10)





if __name__ == '__main__':
    main()
