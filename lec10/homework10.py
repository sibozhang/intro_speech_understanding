import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    # raise RuntimeError("You need to change this part")
    VAD_windowlen = int(0.025*Fs)
    VAD_windowskip = int(0.01*Fs)
    VAD_frames = np.array([waveform[m:m+VAD_windowlen]
                            for m in range(0, len(waveform)-VAD_windowlen, VAD_windowskip)])
    L = int(0.004*Fs)
    S = int(0.002*Fs)
    x_frames = np.array([waveform[m+1:m+1+L]-waveform[m:m+L]
                            for m in range(0, len(waveform)-L, S)])

    energy = np.sum(np.square(VAD_frames), 1)
    threshold = 0.1*np.amax(energy)
    VAD = (energy > threshold).astype(int)

    padded = np.concatenate([[0], VAD, [0]])          
    diff = np.diff(padded)
    startframes = np.flatnonzero(diff == 1)
    endframes = np.flatnonzero(diff == -1)

    labels = np.zeros(x_frames.shape[0], dtype=int)   
    for k in range(len(startframes)):
        labels[5*startframes[k] : 5*endframes[k]] = k+1   

    mstft = np.abs(np.fft.fft(x_frames, axis=1))
    features = 20*np.log10(np.maximum(0.001*np.amax(mstft), mstft))[:, 0:int(L/2)]
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    # raise RuntimeError("You need to change this part")
    NFRAMES, NFEATS = features.shape
    NCLASSES = int(np.amax(labels)) + 1
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(NFEATS, dtype=float),
        torch.nn.Linear(NFEATS, NCLASSES, dtype=float))
    lossfunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    X = torch.tensor(features, dtype=float)
    Y = torch.tensor(labels, dtype=int)
    lossvalues = np.zeros(iterations)
    for t in range(iterations):
        z = model(X)
        loss = lossfunction(z, Y)
        lossvalues[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    # raise RuntimeError("You need to change this part")
    testresults = model(torch.tensor(features, dtype=float))
    return testresults.softmax(-1, dtype=float).detach().numpy()

