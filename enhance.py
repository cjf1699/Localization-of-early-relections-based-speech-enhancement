from net import ResBlock, ResNet
from handler import peak_detection, time_domain_shift, cal_delay, extract_sig
from scipy import signal
from scipy.io import wavfile as wav
from hoa_params import get_encoder, get_y2
import torch
import config as c
import numpy as np
import matplotlib.pyplot as plt
import sys
from encode import HOAencode
from DataPreprocessor import array2HOA


sys.path.append('/home/cjf/workspace/201903_dereverLocEnhance/mir_eval_master/')
from mir_eval import separation as sep

bottle = False
DATA_TYPE = 'stft'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load('./models/ckpoint_epo30_STFT_speech_bot0_lr0.001_wd0.0001_#res3.tar')

block = ResBlock(256, 256, bottleneck=bottle)

model = ResNet(block, numOfResBlock=3, input_shape=(64, 22, 255), data_type=DATA_TYPE).to(device)
model.load_state_dict(checkpoint['state_dict'])
#print(checkpoint['state_dict'])

def enhance(block, angles):
    """
    :param block: given a block of signal
    :param angles: directions to be enchanced
    return: shifted signals according to the delay
    """
    rec_signal = extract_sig(block, angles)
    shifted_signals = np.zeros((rec_signal.shape[0] + 2 * c.tf_max_len, len(angles)), dtype=float)
    lags = np.zeros(len(angles))
    # lags = (np.array([3, 5, 8.544, 8.544, 11]) - 3) / c.speed_of_sound * c.fs   # ideal delay

    for id, angle in enumerate(angles):
        if id >= 1:
            lags[id] = cal_delay(rec_signal[:, 0], rec_signal[:, id]) # 这个值代表了该信号相对于第一个信号超前了多少个点

        shifted_signals[:, id] = time_domain_shift(rec_signal[:, id], lags[id])
        # savemat('./wavs/rec_signal.mat', {'data':rec_signal})
        # print(lags)
        # shifted_signals = shifted_signals[1600-372:shifted_signals.shape[0]-1600+372, :]
    # fig, ax = plt.subplots(4, 1)
    # for i in range(4):  ax[i].plot(shifted_signals[:, i])
    # plt.savefig('each_channel.jpg')
    return shifted_signals


if __name__ == '__main__':
    
    wav_path = '/mnt/hd8t/pchao/SpeechSeparation/mix/data/2speakers_new/wav8k/min/tt/s1/443o0303_2.6045_445o0306_3.2120.wav'
    tf_path = '/mnt/hd8t/cjf/TF_result_fs_8000/RT60_0.32684_dist_1.3816_TF_Matrix.mat'
    import time
    print('转换音频开始')
    sample_rate, s_mono = wav.read(wav_path)
    dataset, s_multi = array2HOA(wav_path, tf_path)
    #start = time.time()
    X = dataset['X']
    Y = dataset['Y']
    example = X[5].to(device)
    label = Y[5]

    output = model(example.unsqueeze(dim=0))
    angles = peak_detection(output.squeeze())
    real = peak_detection(label.squeeze())
    print('转换音频完毕')
    print(angles) # [7  73  173  281] 
    print(real)  # [[5, 30, 75, 175, 280]]
    
    '''
    dataset = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/speech/STFT/tr/DataSet_1.pt')
    X = dataset['X']
    Y = dataset['Y']
    example = X[10].to(device)
    label = Y[10]

    output = model(example.unsqueeze(dim=0))
    angles = peak_detection(output.squeeze())
    real = peak_detection(label.squeeze())
    #print(angles) # [9, 74, 175, 282]
    #print(real)  # [10, 20, 70, 175, 285]
    '''
    print('增强开始')
    shifted_signal = enhance(s_multi, angles)
    #end = time.time()
    print('增强结束')
    #print(end-start)
    
    enhanced_signal = np.mean(shifted_signal, axis=1)
    ref_signal = shifted_signal[:, 0]

    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(s_mono)
    # ax[1].plot(ref_signal)
    # ax[2].plot(enhanced_signal)
    delay_bet_ref_mono = cal_delay(s_mono, ref_signal)
    delay_bet_enhance_mono = cal_delay(s_mono, enhanced_signal)
    # plt.show()
    # 1.
    ref_signal1 = ref_signal[-delay_bet_ref_mono:-delay_bet_ref_mono+len(s_mono)]
    enhanced_signal1 = enhanced_signal[-delay_bet_enhance_mono:-delay_bet_enhance_mono+len(s_mono)]
    '''
    # record
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ref_signal1)
    ax[1].plot(enhanced_signal1)

    wav.write('ref_time.wav', c.fs, ref_signal1)
    wav.write('enhanced_time.wav', c.fs, enhanced_signal1)
    '''
    '''
    # 2.
    # enhanced_signal2 = enhanced_signal[0:len(s_mono)]
    # ref_signal2 = ref_signal[0:len(s_mono)]
    # ax[2].plot(ref_signal2)
    # ax[3].plot(enhanced_signal2)
    # # wav.write('ref2.wav', c.fs, ref_signal2)
    # wav.write('enhanced2.wav', c.fs, enhanced_signal2)
    plt.savefig('enhance_or_not.jpg')
    '''
    print('计算SDR开始')
    SDR, SIR, SAR, perm = sep.bss_eval_sources(s_mono, enhanced_signal1, False)
    SDR0, SIR0, SAR0, perm0 = sep.bss_eval_sources(s_mono, ref_signal1, False)
    print('计算SDR结束')
    print('反射声增强前：', SDR0)
    print('反射声增强后：', SDR)
