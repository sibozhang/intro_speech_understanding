import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    # raise RuntimeError("You need to write this part!")
    nframes = int((len(speech) - frame_length) / frame_skip)
    frames = np.array([ speech[m*frame_skip : m*frame_skip + frame_length]
                        for m in range(nframes) ])
    A = librosa.lpc(frames, order=order, axis=-1)
    excitation = np.zeros((nframes, frame_length))
    for k in range(order + 1):
        excitation[:, order:] += A[:, k:k+1] * frames[:, order-k : frame_length-k]
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    # raise RuntimeError("You need to write this part!")
    nframes, order = A.shape
    synthesis = np.zeros(len(e))
    for i in range(len(synthesis)):
        synthesis[i] = e[i]
        frame = int(i / frame_skip)
        for k in range(1,order):                     
            synthesis[i] -= A[frame, k] * synthesis[i-k]
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    #raise RuntimeError("You need to write this part!")
    gain = np.sqrt(np.sum(np.square(excitation), axis=1))
    nframes = len(gain)
    e_robot = np.zeros(nframes*frame_skip)
    e_robot[::T0] = 1
    for i in range(len(e_robot)):
        e_robot[i] = e_robot[i] * gain[int(i/frame_skip)]
    return gain, e_robot

