# 此文件用于将generator.py模块生成的数据处理成神经网络要求的输入格式，如进行STFT和HOA变换
import config as c
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import wavfile as wav
import torch
import torch.utils.data as data
from scipy import signal
from hoa_params import get_encoder
from scipy.special import hankel2
from handler import *


class DataProcessor(object):
    """
    Convert the time-domain data (.wav) to HOADataset format:
    dict fo tensors:
    X: (C, T, F)
    Y: (360,)
    """

    def __init__(self, path, image_path, tf_list, snr_list, net='res', is_tr='tr',
                 is_speech=False, data_type='hoa', normalize=True):

        # check pass
        self.anechoic_path = path
        self.image_path = image_path
        self.tf_list = tf_list      # tf of different rooms
        self.snr_list = snr_list       # various SNR
        self.net = net
        self.data_type = data_type   # hoa or stft
        self.is_tr = is_tr           # tr, cv, or te
        self.norm = normalize
        self.is_speech = is_speech

        term = '/mnt/hd8t/cjf/random_reverb_wavs/'
        if self.is_speech:
            term += 'speech/'
        if self.data_type == 'stft':
            term += 'STFT/'
        term += (self.is_tr + '/')
        self.save_path = term

        with open(self.anechoic_path, 'r') as f:
            _all_path = f.readlines()
        self.num_of_wavs = len(_all_path)

    def run(self):
        file_idx = 0
        sum_temp = None
        input_dim = None
        total_sample_cnt = 0

        for path in file_gen(self.anechoic_path):
            file_idx += 1
            # print(file_idx)

            print(self.data_type, self.is_tr, file_idx)
            content = path.strip().split(' ')
            adr, label = content[0], content[1]

            index = int(label) - 1
            random.shuffle(self.tf_list)
            TF_path = self.tf_list[0]

            if self.is_tr == 'tr':
                random.shuffle(self.snr_list)
                snr = self.snr_list[0]
            else:
                thres = self.num_of_wavs / len(self.snr_list)
                if file_idx < thres:    snr = self.snr_list[0]
                elif thres <= file_idx < 2 * thres: snr = self.snr_list[1]
                elif 2 * thres <= file_idx < 3 * thres: snr = self.snr_list[2]
                else:   snr = self.snr_list[3]
            
            dataset, s_multi, cnt = transform(adr, TF_path, index, snr, self.data_type, self.is_speech)

            total_sample_cnt += cnt
            torch.save(dataset, self.save_path + 'DataSet_' + str(file_idx) + '.pt')

            if self.norm:
                if file_idx == 1:
                    sum_temp = torch.sum(dataset['X'], dim=0)
                else:
                    sum_temp += torch.sum(dataset['X'], dim=0)

        if self.norm:
            _mean = sum_temp / total_sample_cnt
            mean_name = 'speech_' + ('' if self.data_type == 'hoa' else 'stft_') + 'mean.pt'
            torch.save(_mean, mean_name)
            # _mean = torch.load(mean_name)
            #
            std_temp = torch.zeros(input_dim[1:], dtype=torch.float32)
            # std_temp = torch.zeros((64, 22, 255), dtype=torch.float32)
            for i in range(file_idx):
                print('std...file {}'.format(i))
                data_dict = torch.load(self.save_path + 'DataSet_' + str(i + 1) + '.pt')
                std_temp += torch.sum((data_dict['X'] - _mean) ** 2, dim=0)

            # flag = (std_temp < 0).any()
            _std = np.sqrt(1.0 / (total_sample_cnt - 1) * std_temp)
            std_name = 'speech_' + ('' if self.data_type == 'hoa' else 'stft_') + 'std.pt'
            torch.save(_std, std_name)


class HOADataSet(data.Dataset):
    """
    Generate the appropriate format DataSet.

    """

    def __init__(self, path, index, data_type, is_speech=False):
        super(HOADataSet, self).__init__()
        self.readPath = path
        self.is_speech = is_speech
        self.data_type = data_type

        speech_term = 'speech_' if self.is_speech else ''
        data_type_term = 'stft_' if self.data_type == 'stft' else ''

        self.data_mean = torch.load(speech_term + data_type_term + 'mean.pt')
        self.data_std = torch.load(speech_term + data_type_term + 'std.pt')
        self.data_max = torch.load(speech_term + 'max.pt')
        flag = (self.data_max == 0).any()

        self.examples = torch.load(self.readPath + 'DataSet_' + str(index) + '.pt')
        self.X = self.examples['X']
        self.Y = self.examples['Y']
        # self.direct = self.examples['direct']
        if self.data_type == 'stft':
            self.X = (self.X - self.data_mean) / self.data_std
        elif self.data_type == 'hoa':
            # pass
            self.X = torch.from_numpy(np.where(self.X == 0, self.X, self.X / self.data_max))

    def __getitem__(self, index):
        _sample, _label = self.X[index], self.Y[index]

        return _sample, _label

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        self.X = torch.cat((self.X, other.X), dim=0)
        self.Y = torch.cat((self.Y, other.Y), dim=0)
        return self


class ERdataset(data.Dataset):
    def __init__(self, path, index, norm=None):
        super(ERdataset, self).__init__()
        self.readPath = path
        self.examples = torch.load(self.readPath + 'SEdata{}.pt'.format(index))
        self.X = self.examples['X']
        self.Y = self.examples['Y']
        if norm != None:
            self.data_mean = torch.load('SE_datamean.pt')
            self.data_std = torch.load('SE_datastd.pt')
            self.X = (self.X - self.data_mean) / self.data_std

    def __getitem__(self, index):
        _sample, _label = self.X[index], self.Y[index]
        return _sample, _label

    def __len__(self):
        return len(self.X)

    def __add__(self, other):
        self.X = torch.cat((self.X, other.X), dim=0)
        self.Y = torch.cat((self.Y, other.Y), dim=0)
        return self

def SE_data_gen(is_tr):
    """
    generate data for speech_enhancement
    """
    file_idx = 0
    sum_temp = None
    std_temp = None
    total_sample_cnt = 0
    if is_tr == 'tr':
        tf_list = c.TRAIN_TF_LIST
        random.shuffle(c.snr_list)
        snr = c.snr_list[0]
    elif is_tr == 'cv':
        tf_list = c.VALID_TF_LIST
    elif is_tr == 'tt':
        tf_list = c.TEST_TF_LIST

    anechoic_path = 'anechoic_mono_speech_' + is_tr + '.flist'
    with open(anechoic_path, 'r') as f:
        all_wavs = f.readlines()
        num_of_wavs = len(all_wavs)
    
    with open(anechoic_path, 'r') as f:
        for path in all_wavs:
            file_idx += 1
            # print(file_idx)
            # if file_idx <= 1735:
            #    continue
            print(is_tr, file_idx)
            content = path.strip().split(' ')
            adr, label = content[0], content[1]
            sample_rate, wav_data_temp = wav.read(adr)
            assert sample_rate == c.fs

            index = int(label) - 1
            random.shuffle(tf_list)
            TF_path = tf_list[0]
            TF = load_mat(TF_path)
            #_mmax = np.max(TF[index][0], axis=0)
            #_mmin = np.min(TF[index][0], axis=0)
            wav_data = get_array_signal(wav_data_temp, TF[index][0]).astype(np.float32)
            if is_tr == 'cv' or is_tr == 'tt':
                thres = num_of_wavs / len(c.snr_list)
                if file_idx < thres:
                    snr = c.snr_list[0]
                elif thres <= file_idx < 2 * thres:
                    snr = c.snr_list[1]
                elif 2 * thres <= file_idx < 3 * thres:
                    snr = c.snr_list[2]
                else:
                    snr = c.snr_list[3]
            #_mmax = np.max(wav_data, axis=0)
            #_mmin = np.min(wav_data, axis=0)
            wav_data = awgn(wav_data, snr)
            #mmax = np.max(wav_data, axis=0)
            #mmin = np.min(wav_data, axis=0)
            angles = get_goal_angle(index * 5)
            extracted_sig = extract_sig(wav_data, angles)
            assert extracted_sig.shape[1] == 5
            extracted_sig = extracted_sig[0:len(wav_data_temp), :]
            fig, ax = plt.subplots(6, 1)
            ax[0].plot(wav_data_temp)
            for i in range(5): ax[i+1].plot(extracted_sig[:, i])
            plt.show()
            # frames of multichannel, the 1st dimension is channel
            multi_frames = []
            for i in range(5):
                frames_each_chan = sig2frames(extracted_sig[:, i])
                multi_frames.append(frames_each_chan)
            X = torch.from_numpy(np.array(multi_frames).transpose([1, 0, 2]))
            Y = torch.from_numpy(sig2frames(wav_data_temp))
            torch.save({'X': X, 'Y': Y}, c.SE_data_save_path + is_tr + '/SEdata{}.pt'.format(file_idx))

def get_goal_angle(direct_angle):
    return c.check_ref_angle[direct_angle]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ser', type=str,
                        default='sh',  # sk : shengke 1 hao, sh: shrc
                        help='which server to use')

    args = parser.parse_args()

    SERVER = args.ser

    if SERVER == 'sk':
        IMAGE_PATH = '/gpfs/share/home/1801213778/workspace/2019summerholiday/stage2/RirsOfRooms/'
        TF_PATH = '/gpfs/share/home/1801213778/Dataset/TF_result_fs_16000/'
    elif SERVER == 'sh':
        IMAGE_PATH = '/home/cjf/workspace/Matlab/RirsOfRooms/'
        TF_PATH = '/mnt/hd8t/cjf/TF_result_fs_8000/'          # modified in 2019 11 11, change fs.
    elif SERVER == 'ship':
        IMAGE_PATH = '/home/cjf/workspace/stage2/RirsOfRooms/'
        TF_PATH = '/data/cjf/TF_result_fs_16000/'
    else:
        raise RuntimeError('Unrecognized server!')

    #SE_data_gen('tr')
    cal_cmvn(c.SE_data_save_path, 3340, 'SEdata')
    '''
    cnt = 0
    for i in range(3600):
        print('mean', i+1)
        if i == 0:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/tr/DataSet_{}.pt'.format(i+1))
            data = data['X']
            cnt += len(data)
            temp = torch.sum(data, dim=0)
        else:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/tr/DataSet_{}.pt'.format(i+1))
            data = data['X']
            temp += torch.sum(data, dim=0)
            cnt += len(data)
    _mean = temp / cnt
    torch.save(_mean, 'data_stft_mean.pt')
    '''
    '''
    _mean = torch.load('speech_mean.pt')
    cnt = 0
    for i in range(3340):
        print(i + 1)
        if i == 0:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/speech/tr/DataSet_{}.pt'.format(i+1))
            data = data['X']
            _max = torch.max(torch.abs(data), dim=0).values
            # std_temp = torch.sum((data - _mean) ** 2, dim=0)
            # cnt += data.shape[0]

        else:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/speech/tr/DataSet_{}.pt'.format(i + 1))
            data = data['X']
            _max_temp = torch.max(torch.abs(data), dim=0).values
            # std_temp += torch.sum((data - _mean) ** 2, dim=0)
            _max = torch.from_numpy(np.where(_max_temp > _max, _max_temp, _max))
            # cnt += data.shape[0]
        # if (std_temp == 0).any():
        #     print('std 0000000000000{}'.format(i+1))
    # _std = np.sqrt(1.0 / (cnt - 1) * std_temp)
    # torch.save(_std, 'speech_stft_std.pt')
    torch.save(_max, 'speech_max.pt')
    print('finished!')
    '''
    # _mean = torch.load('data_mean.pt')
    #
    # std_temp = torch.zeros(50, 22, 255, dtype=torch.float64)
    # total_sample_cnt = 0
    # for i in range(3600):
    #     HOAdata_dict = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/tr/DataSet_' + str(i + 1) + '.pt')
    #     num = len(HOAdata_dict['X'])
    #     total_sample_cnt += num
    #     for j in range(num):
    #         print('std...file {}, No.{}'.format(i, j))
    #         # print(_mean.dtype)
    #         # print(HOAdata_dict['X'][j].dtype)
    #         std_temp += (HOAdata_dict['X'][j] - _mean) ** 2
    # _std = np.sqrt(1.0 / (total_sample_cnt - 1) * std_temp)
    # torch.save(_std, 'data_std.pt')

    #d_tr = DataProcessor(path=c.TRAIN_FILE_PATH, image_path=c.IMAGE_PATH, tf_list=c.TRAIN_TF_LIST,
     #                         snr_list=c.snr_list, net='res', is_tr='tr',
      #                        is_speech=True, data_type='hoa', normalize=True)
    #d_tr.run()
    # d_cv = DataProcessor(path=VALID_FILE_PATH, image_path=IMAGE_PATH, tf_list=VALID_TF_LIST,
    #                           snr_list=[10, 5, 0, -5], net='res', is_tr='cv',
    #                           is_speech=True, data_type='hoa', normalize=False)
    # d_cv.run()
    # d_tt = DataProcessor(path=TEST_FILE_PATH, image_path=IMAGE_PATH, tf_list=TEST_TF_LIST,
    #                           snr_list=[10, 5, 0, -5], net='res', is_tr='tt',
    #                           is_speech=True, data_type='hoa', normalize=False)
    # d_tt.run()
    # d_tr_stft = DataProcessor(path=TRAIN_FILE_PATH, image_path=IMAGE_PATH, tf_list=TRAIN_TF_LIST,
    #                           snr_list=[10, 5, 0, -5], net='res', is_tr='tr',
    #                           is_speech=True, data_type='stft', normalize=True)
    # d_tr_stft.run()
    # d_cv_stft = DataProcessor(path=VALID_FILE_PATH, image_path=IMAGE_PATH, tf_list=VALID_TF_LIST,
    #                           snr_list=[10, 5, 0, -5], net='res', is_tr='cv',
    #                           is_speech=True, data_type='stft', normalize=False)
    # d_cv_stft.run()
    # d_tt_stft = DataProcessor(path=TEST_FILE_PATH, image_path=IMAGE_PATH, tf_list=TEST_TF_LIST,
    #                      snr_list=[10, 5, 0, -5], net='res', is_tr='tt',
    #                      is_speech=True, data_type='stft', normalize=False)
    # d_tt_stft.run()


'''
# get_az函数测试通过，标签没问题

'''
