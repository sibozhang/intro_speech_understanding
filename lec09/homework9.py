import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    # raise RuntimeError("You need to change this part")
    framelength = int(0.025*Fs)
    frameskip = int(0.01*Fs)
    frames = np.array([waveform[m:m+framelength]
                        for m in range(0, len(waveform)-framelength, frameskip)])
    if len(frames) == 0:
        return []
    energies = np.sum(np.square(frames), axis=1)
    threshold = 0.1*np.amax(energies)           
    is_speech = energies > threshold

    padded = np.concatenate([[0], is_speech.astype(int), [0]])
    diff = np.diff(padded)
    framestarts = np.flatnonzero(diff == 1)
    frameends = np.flatnonzero(diff == -1)

    segments = [waveform[frameskip*s : frameskip*e]
                for s, e in zip(framestarts, frameends)
                if frameskip*(e-s) >= framelength]     
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    raise RuntimeError("You need to change this part")

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    # raise RuntimeError("You need to change this part")
    segments = VAD(testspeech, Fs)
    test_models = segments_to_models(segments, Fs)      # 复用，不再重写一遍

    M = np.array(models)                                # (Y, D)
    T = np.array(test_models)                           # (K, D)
    M_n = M/np.linalg.norm(M, axis=1, keepdims=True)
    T_n = T/np.linalg.norm(T, axis=1, keepdims=True)
    sims = M_n @ T_n.T                                  # (Y, K)

    test_outputs = [labels[y] for y in np.argmax(sims, axis=0)]
    return sims, test_outputs


