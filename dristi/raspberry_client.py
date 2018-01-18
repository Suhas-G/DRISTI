import sys, os, struct, pickle, socket

import cv2
import numpy as np
import sounddevice as sd

from image_loader import CaptureImage
from audio import AudioFeedback

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
LEFT_CAM = 0
RIGHT_CAM = 1

def initialise_network(addr, port):
    '''Description: A function to initialise a streaming connection.
    Parameters:
    addr: Address of network to connect to.
    port: Port ID to connect to.
    '''
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect((addr, port))
    return connection

def send(connection, data):
    '''Description: A simple protocol to send images.
    First send the size of the images (2), then send the images.
    Parameters:
    connection: Connection object received from initialisation function.
    data: The image data to be sent to server.
    '''
    connection.send(struct.pack('<L', 640*480*2))
    connection.send(data)

def recieve(connection):
    '''
    Description: A simple protocol for receiving control signals.
    First obtain the size of signal, then receive the signal.
    Parameters:
    connection: Connection object received from initialisation function.
    Returns:
    signal: Control signal sent by the server.
    '''
    signal_len = struct.unpack('<I', connection.recv(struct.calcsize('<I')))[0]
    
    signal = b''
    length = 0
    while length < signal_len:
        chunk = connection.recv(int(signal_len/2))
        signal += chunk
        length += sys.getsizeof(chunk)
        
    signal = pickle.loads(signal)
    return signal

def preprocess(left_img, right_img):
    '''Description: Preprocess the images, by converting to gray scale and concatenating them.
    Parameters:
    left_img: Left image obtained from webcam
    right_img: Right image obtained from webcam
    Return:
    data: Left and right images concatenated.
    '''
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    data = np.hstack((left_img, right_img)).astype(np.uint8)
    return data


def main():
    image_loader = CaptureImage(LEFT_CAM, RIGHT_CAM)
    image_loader.start()
    connection = initialise_network('Suhas-G', 8000)
    feedback = AudioFeedback()
    print('Connection initialised...')
    while True:
        left_img, right_img = image_loader.load_images()
        data = preprocess(left_img, right_img)
        send(connection, data.tostring())
        signal = recieve(connection)
        feedback.update(signal)
        
        
    feedback.stop()
        
if __name__ == '__main__':
    main()
        
