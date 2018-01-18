import numpy as np
import sounddevice as sd



class AudioFeedback(object):
    '''A class to handle Audio feedback to the user.
    Methods:
    init_playback
    update
    stop
    '''
    def __init__(self, fs = 44100, duration = 10, volume = 0.3, channels = 2):
        '''
        Description: Initialise playback
        Parameters:
        fs : Sampling Frequency for the output audio.
        duration: Duration of each playback to be played, but can be changed in the middle
        volume: Volume for the audio output
        channels: No of channels being used.
        '''
        self.fs = fs
        self.duration = duration
        self.volume = volume
        self.channels = channels

        sound_1 = (np.sin(2*np.pi*np.arange(self.fs*self.duration)*440/self.fs)).astype(np.float32)
        sound_2 = (np.sin(2*np.pi*np.arange(self.fs*self.duration)*500/self.fs)).astype(np.float32)
        sound_3 = (np.sin(2*np.pi*np.arange(self.fs*self.duration)*540/self.fs)).astype(np.float32)

        self.frequency_map = {1: sound_1, 2: sound_2, 3: sound_3}
        self.size = sound_1.size
        self.init_playback()

    def init_playback(self):
        '''
        Description: Set some default values of playback.
        '''
        sd.default.samplerate = self.fs
        sd.default.channels = self.channels

    def update(self, signal):
        '''
        Description: Update the audio output depending on signal
        Parameters:
        signal: Control signal
        '''
        freq = signal.count(1)
        sound = np.zeros([self.size, 2])
        if signal[0] == 0 and signal[1] == 1 and signal[2] == 0:
            sound = np.hstack([self.frequency_map[freq], self.frequency_map[freq]])
        else:
            if signal[0] == 1:
                sound[:, 0] = self.frequency_map[freq]
            if signal[2] == 1:
                sound[:, 1] = self.frequency_map[freq]

        sd.play(self.volume* sound, loop = True, mapping = [1, 2])

    def stop(self):
        '''
        Description: Stop playback
        '''
        sd.stop()



if __name__ == '__main__':
    '''For testing purpose
    '''
    fs = 44100
    duration = 10.0
    volume = 0.3
    channels = 2
    feedback = AudioFeedback(fs = fs, duration = duration, volume = volume, channels = channels)

    while True:
        try:
            signal = list(map(float, input().strip().split(' ')))
            feedback.update(signal)

        except:
            pass
            break

    feedback.stop()
