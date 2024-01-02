import numpy as np

def make_noise(original_caption, noise_ratio):
    '''
    original_caption: The original caption file path of the dataset. Usually, it is a ".txt" file.
    noise_ratio: The noise proportion you need. (e.g., 0.2, 0.4, and 0.6)
    '''
    captions = []
    for line in open(original_caption, 'r', encoding='utf-8'):
        captions.append(line.strip())
    length = len(captions)
    idx = np.arange(length)
    np.random.shuffle(idx)
    # get noise's number
    noise_length = int(noise_ratio * length)
    shuffle_cap= np.array(captions)[idx[:noise_length]]
    np.random.shuffle(shuffle_cap)
    noise_cap = np.array(captions)
    noise_cap[idx[:noise_length]] = shuffle_cap
    with open('.${YOUR PATH}$/annotations/scan_split/{}_noise_train_caps.txt' .format(noise_ratio), mode='a', encoding='utf-8') as f:
            for cap in list(noise_cap):
                f.write(cap + '\n')
    return