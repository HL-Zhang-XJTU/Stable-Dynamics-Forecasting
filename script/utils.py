import numpy as np

def RMS_error(pred, real=None, axis=0):
    if real is None:
        return np.sqrt(np.mean(np.power(pred, 2), axis=axis))
    return np.sqrt(np.mean(np.power(pred - real, 2), axis=axis))

def resample(x, stride, H):
    length = x.shape[1]
    clips = int((length-H)/stride+1)
    matrix = np.zeros([clips*x.shape[0], x.shape[-1]])
    for k in range(x.shape[0]):
        for i in range(clips):
            matrix[i+k*clips] = x[k, i*stride+H-1]
    return matrix


def to_sequence(data, stride=1000 ,win=1000):
    if len(data.shape) == 3:
        length = data.shape[1]
        clips = int((length-win)/stride+1)
        matrix = np.zeros([clips*data.shape[0], win, data.shape[-1]])
        for k in range(data.shape[0]):
            for i in range(clips):
                matrix[i+k*clips] = data[k, i*stride:i*stride+win]
        return matrix
    else:
        length = data.shape[0]
        clips = int((length-win)/stride+1)
        matrix = np.zeros([clips, win, data.shape[-1]])
        for i in range(clips):
            matrix[i] = data[i*stride:i*stride+win]
        return matrix

def to_2Dtensor(data, stride=5, H=25):
    length = data.shape[1]
    clips = int((length-H)/stride+1)
    matrix = np.zeros([clips*data.shape[0], 1, H, 14])
    for k in range(data.shape[0]):
        for i in range(clips):
            matrix[i+k*clips, 0] = data[k, i*stride:i*stride+H]
    return matrix

def add_gaussian_noise(data, mean=0.0, std=0.1):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise
