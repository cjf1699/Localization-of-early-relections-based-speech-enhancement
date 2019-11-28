import numpy as np
import config as c
import torch
import warnings
from scipy.io import wavfile as wav
from scipy.io import loadmat
from scipy import signal
from hoa_params import get_encoder, get_y2
from encode import HOAencode

def peak_detection(data):
    # check pass
    assert len(data.shape) == 1

    N = len(data)
    peaks = []
    for i in range(N):
        if i - c.sigma < 0 or i + c.sigma > N:
            if i - c.sigma < 0:
                tmp = torch.cat((data[i - c.sigma:], data[:i+c.sigma]))
            else:
                tmp = torch.cat((data[i-c.sigma:], data[0:(i + c.sigma) % N + 1]))
        else:
            tmp = data[i-c.sigma:i+c.sigma]

        if data[i] == torch.max(tmp) and data[i] > c.thres:
            peaks.append(i)
        # if plot:
        #     plt.figure()
        #     plt.plot(data.cpu().detach().numpy())
        #     plt.savefig('normal' + time.asctime(time.localtime(time.time())) + '.jpg')
        #     plt.close()

    return peaks # 返回峰值的角度


def cal_recall(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')

    err = [5, 10, 15]
    
    recall = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for item in label:
            for yuce in pred:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        recall[idx] = cnt / len(label)

    return recall


def cal_precision(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')
    err = [5, 10, 15]
    prec = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for yuce in pred:
            for item in label:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        prec[idx] = cnt / len(pred)

    return prec


def cal_delay(s1, s2):
    """
    s1:reference signal
    s2:signal to be precessed
    return: sample points s2 preceeds s1. if negtive, it means that s2 is slower than s1
    """
    assert len(s1.shape) == len(s2.shape) == 1 # assume both signal has only one channel
    _s1, _s2 = list(s1), list(s2) # transform both signal to list for convenience
    # padding
    if len(_s1) < len(_s2):
        _s1 += ([0] * (len(_s2) - len(_s1)))
    elif len(_s2) < len(_s1):
        _s2 += ([0] * (len(_s1) - len(_s2)))
    _len = 2 * len(_s1) - 1
    corr = np.correlate(_s1, _s2, 'full')
    idx = np.argmax(np.abs(corr))
    maxlag = _len // 2
    lags = list(range(-maxlag, maxlag+1, 1))
    assert len(lags) == _len
    return lags[idx]



def time_domain_shift(signal, shift_point):
    """
    Shift signal in time domain, shift_time can be real number.

    For fraction part, shift it in frequency domain.
    """
    assert len(signal.shape) == 1 or (len(signal.shape) == 2 and signal.shape[1] == 1)  # must be mono signal.

    point_integer = int(shift_point)
    point_fraction = shift_point - point_integer
    s_len = signal.shape[0]

    _signal = np.zeros(s_len + 2 * c.tf_max_len, dtype=float)
    _signal[c.tf_max_len + point_integer:c.tf_max_len + point_integer + s_len] = signal

    _signal_fft = np.fft.fft(_signal)
    fft_len = _signal_fft.size
    half_len = (s_len + 1) // 2

    _signal_fft[:half_len] *= np.exp(- 1j * 2 * np.pi * point_fraction / fft_len * np.asarray(range(half_len)))
    _signal_fft[-1:-half_len:-1] = np.conj(_signal_fft[1:half_len])

    _signal = np.fft.ifft(_signal_fft).real
    return _signal


def load_mat(path):
    ori_data = loadmat(path)

    keys = list(ori_data.keys())
    for key in keys:
        if key[0] != '_':
            data = ori_data[key]
            break

    return data

def extract_sig(block, angles):
    
    freq_array, time_array, s_fft = signal.stft(
        block, c.fs, window='hamming', nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, axis=0, padded=True)
    n_block = time_array.size
    s_fft = s_fft[c.valid_freq_index, :, :]

    hoa = np.zeros((c.n_freq, c.hoa_num, n_block), dtype=np.complex64)
    encoder = get_encoder()
    for freq_index, freq in enumerate(c.valid_freq_array):
        hoa[freq_index, :, :] = encoder[freq_index, :, :].dot(s_fft[freq_index, :, :])  # 文献中的b （26）

    t_array, t_hoa = signal.istft(
            hoa, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, time_axis=-1, freq_axis=0)

    t_hoa = t_hoa.T
    
    #t_hoa, _fs = HOAencode(block, c.fs, 4)
    assert t_hoa.shape[1] == c.hoa_num

    rec_signal = np.zeros((t_hoa.shape[0], len(angles)), dtype=float) # each column is an enhanced signal

    Y = get_y2(np.array(angles) / 180 * np.pi, 0)   # get spherical harmonics
    for id, angle in enumerate(angles):
        rec_signal[:, id] = t_hoa.dot(Y[:, id])     # beamforming
    return rec_signal

def sig2frames(block):
    """
    assume block is a mono channel signal
    return: n*frames
    """
    assert len(block.shape) == 1
        
    frames = []
    start, end = 0, 0
    while start + c.frame_size <= len(block):
        end = start + c.frame_size
        frames.append(block[start:end])
        start += c.frame_step
    return np.array(frames)

def get_array_signal(mono_data, tf):
    '''
    mono_data: mono speech signal
    tf: the hrir for n_chans  shape: (tf_len, n_chan)
    n_chan: the number of mics.
    return: multichannel received signals stimulates by the mono_data
    '''

    try:
        assert (len(tf.shape) == 2 and tf.shape[1] == c.n_chan)
    except AssertionError:
        print(tf.shape, c.n_chan)

    L = len(mono_data) + tf.shape[0] - 1

    result = np.zeros((L, c.n_chan))
    # print(tf.shape)
    for i in range(c.n_chan):
        # print(tf.shape)
        result[:, i] = np.convolve(mono_data, tf[:, i])
    
    return result

def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / x.size
    npower = xpower / snr
    return (np.random.randn(*x.shape) * np.sqrt(npower) + x).astype(np.float32)

def cal_SDR_improve(clean, direct, enhance):
    """
    calculate a SDR1: direct to clean
    calculate a SDR2: enhance to clean
    return :SDR2 - SDR1 (improvement)
    """
    import sys
    sys.path.append('/home/cjf/workspace/201903_dereverLocEnhance/mir_eval_master/')
    from mir_eval import separation as sep

    SDR1, SIR1, SAR1, perm1 = sep.bss_eval_sources(clean, enhance, False)
    SDR2, SIR2, SAR2, perm2 = sep.bss_eval_sources(clean, direct, False)
    return SDR1 - SDR2

def cal_cmvn(path, num, prefix, suffix='.pt'):
    """
    given the data path, the total number of files and the prefix of data name, calculate the mean and the std
    """
    # assume the index begins at 1
    total_cnt = 0
    for i in range(1, num+1):
        print('mean:', i)
        data = torch.load(path + 'tr/' + prefix + str(i) + suffix)['X']
        total_cnt += data.shape[0]
        if i == 1:
            sum_temp = torch.sum(data, dim=0)
        else:
            sum_temp += torch.sum(data, dim=0)
    _mean = sum_temp / total_cnt
    torch.save(_mean, '_mean' + suffix)
    for i in range(1, num+1):
        print('std:', i)
        data = torch.load(path + 'tr/' + prefix + str(i) + suffix)['X']
        if i == 1:
            std_temp = torch.sum((data - _mean) ** 2, dim=0)
        else:
            std_temp += torch.sum((data - _mean) ** 2, dim=0)
    std = np.sqrt(1.0 / (total_cnt - 1) * std_temp) 
    if (std == 0).any():
        warnings.warn('There exists 0 in std, may occur ZeroDivisionError!')
    torch.save(std, '_std' + suffix)

# 20191128
def get_az(path):
    with open(path, 'rb') as f:
        records = np.fromfile(f)
        tmp = records.reshape(-1, 5)
        # 按照到达时间排序
        tmp = tmp[tmp[:, 0].argsort()]
        # aaa = tmp[0, :]
        index1 = np.where(tmp[:, 4] <= 1)[0]
        tmp1 = tmp[index1, :]
        index2 = np.where(tmp1[:, 3] == 0)[0]
        tmp2 = tmp1[index2]
        y = tmp2[:, 2]
        x = tmp2[:, 1]
        az = np.arctan2(y, x) / np.pi * 180
        # 小于0的角度，加上360
        az = np.where(az < 0, az + 360, az)
        az = np.round(az).astype(int)
        # 把角度归到距离最近的5的倍数
        for idx, angle in enumerate(az):
            if angle % 5 == 0:
                continue
            if angle % 5 <= 2:
                az[idx] = angle - angle % 5
            else:
                az[idx] = (angle + 5) - angle % 5
    return az

def label_gen(label):
    # direc_id = int(label) // 5
    direc_id = int(label) - 1
    aa = c.tf_list[0].split('_')

    read_path = (c.IMAGE_PATH + 'RT60_' + aa[4] + '/dist_' + aa[6] +
                 '/source_{}.binary').format(direc_id + 1)
    source_az = get_az(read_path)
    # print(source_az)
    all_az = np.linspace(0, 359, 360)[:, np.newaxis]
    gaussian_hot = None
    for idx, az in enumerate(source_az):
        if idx == 0:
            gaussian_hot = np.exp(-((all_az - az) ** 2) / (c.std ** 2))
        else:
            gaussian_hot = np.hstack((gaussian_hot, np.exp(-((all_az - az) ** 2) / (c.std ** 2))))
    gaussian_hot = np.max(gaussian_hot, axis=1)
    # plt.figure()
    # plt.plot(gaussian_hot)
    # plt.show()
    # plt.close()
    return gaussian_hot

def get_HOA(s_fft):
    encoder = get_encoder()
    nFrames = s_fft.shape[2]
    hoa = np.zeros((c.n_freq, c.hoa_num, nFrames), dtype=np.complex64)

    for freq_index, freq in enumerate(c.valid_freq_array):
        hoa[freq_index, :, :] = encoder[freq_index, :, :].dot(s_fft[freq_index, :, :])  # 文献中的b （26）
    #flag3 = (hoa == 0).any()
    res = hoa.astype(np.complex64)
    return res

def transform(wav_path, tf_path, index, snr, data_type='stft', cut=True):
    """
    Used during being-applied stage. This method transforms a wavform in time-domain into HOA or STFT format
    :param wav_path:
    :param index: index corresponding to direct sound direction
    :param tf_path: a random-picked RIR
    :param snr: a random-picked snr
    :param data_type: HOA or STFT
    :return: data in HOA or STFT format, the multi-channel audio data and the number of examples tranfromed from this wav_path

    """
    sample_rate, wav_mono = wav.read(wav_path)
    assert c.fs == sample_rate
    if cut:
        wav_mono = clippout_silence(wav_mono)

    TF = load_mat(tf_path)
    wav_data = get_array_signal(wav_mono, TF[index][0]).astype(np.float32)
    wav_data = awgn(wav_data, snr)

    dataset = {}
    dataset['X'] = []
    _, _, s_fft = signal.stft(
        wav_data, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, axis=0, padded=False)
    s_fft = s_fft[c.valid_freq_index, :, :]

    # flag1 = (s_fft == 0).any()
    if data_type == 'hoa':
        s_fft = get_HOA(s_fft)
        # flag2 = (s_fft == 0).any()
    start, cnt = 0, 0

    while start + c.frames_per_block + 2 <= s_fft.shape[2]:
        # print(cnt)
        cnt += 1  #
        temp1 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].real)
        temp2 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].imag)
        temp = torch.cat((temp1, temp2), dim=1).permute(
                [1, 2, 0])  # change (freq, chan, time) to (chan, time, freq)
        if start == 0:
            dataset['X'] = temp
            input_dim = list(temp.shape)
        else:
            dataset['X'] = torch.cat((dataset['X'], temp), dim=0)
        start += (c.frames_per_block + 2)

    input_dim.insert(0, cnt)

    dataset['X'] = dataset['X'].reshape(input_dim)
    dataset['Y'] = torch.from_numpy(label_gen(str(index+1))).repeat(cnt, 1)
    return dataset, wav_data, cnt

def file_gen(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            yield line


def clippout_silence(data):
    # ======= assert the data to be a mono signal ========
    assert len(data.shape) == 1
    silence_ids = np.where(np.abs(data) < c.gamma * np.mean(np.abs(data)))
    # print(silence_ids)
    new_data = np.delete(data, silence_ids)
    # check pass
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.plot(data)
    # ax2 = fig.add_subplot(212)
    # ax2.plot(new_data)
    # # print(len(data), len(new_data))
    # plt.show()
    return new_data


if __name__ == '__main__':
    a = np.array([1, 3, 2, 9, 0, 10])
    b = np.array([0, 0, 0, 0, 1, 3, 2, 9, 0, 10])
    res1 = cal_delay(a, b)
