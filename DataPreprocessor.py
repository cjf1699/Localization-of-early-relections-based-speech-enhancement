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


class DataProcessor(object):
    """
    First, slice the original data into blocks.
    Second, convert the time-domain data to HOA efficents
    """

    def __init__(self, path, image_path, tf_list, snr_list, net='res', is_tr='tr',
                 is_speech=False, data_type='hoa', normalize=True):
        # 音频文件的路径 和其角度标签
        self.anechoic_path = path
        self.image_path = image_path
        self.tf_list = tf_list
        self.snr_list = snr_list
        self.encoder = get_encoder()  # (n_freq, 25, 32)
        self.net = net
        self.data_type = data_type
        self.is_tr = is_tr
        self.is_speech = is_speech
        self.norm = normalize

        speech_term = 'speech/' if self.is_speech else ''
        data_type_term = 'STFT/' if self.data_type == 'stft' else ''
        tr_term = self.is_tr + '/'
        term = '/'.join((speech_term, data_type_term, tr_term))
        self.save_path = '/mnt/hd8t/cjf/random_reverb_wavs/' + term

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
            print(self.data_type, self.is_tr, file_idx)
            content = path.strip().split(' ')
            adr, label = content[0], content[1]
            sample_rate, wav_data_temp = wav.read(adr)
            assert sample_rate == c.fs

            if self.is_speech:
                wav_data_temp = clippout_silence(wav_data_temp)

            index = int(label) - 1
            random.shuffle(self.tf_list)
            TF_path = self.tf_list[0]
            TF = self.load_mat(TF_path)
            wav_data = self.get_array_signal(wav_data_temp, TF[index][0])
            if self.is_tr == 'tr':
                random.shuffle(self.snr_list)
                snr = self.snr_list[0]
                wav_data = self.awgn(wav_data, snr)
            else:
                thres = self.num_of_wavs / len(self.snr_list)
                if file_idx < thres:
                    snr = self.snr_list[0]
                elif thres <= file_idx < 2 * thres:
                    snr = self.snr_list[1]
                elif 2 * thres <= file_idx < 3 * thres:
                    snr = self.snr_list[2]
                else:
                    snr = self.snr_list[3]
                wav_data = self.awgn(wav_data, snr)

            dataset = {}
            dataset['X'] = []
            _, _, s_fft = signal.stft(
                wav_data, c.fs, nperseg=c.frame_size, noverlap=c.n_overlap, nfft=c.fft_point, axis=0, padded=False)
            s_fft = s_fft[c.valid_freq_index, :, :]
            if self.data_type == 'hoa':
                s_fft = self.get_HOA(s_fft)
            start, cnt = 0, 0

            while start + c.frames_per_block + 2 <= s_fft.shape[2]:
                # print(cnt)
                cnt += 1  #
                temp1 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].real)
                temp2 = torch.from_numpy(s_fft[:, :, start:start + c.frames_per_block + 2].imag)
                if self.net == 'hoa':
                    temp = torch.cat((temp1, temp2), dim=1).permute(
                        [0, 2, 1])  # change (freq, chan, time) to (freq, time, chan)
                elif self.net == 'res':
                    temp = torch.cat((temp1, temp2), dim=1).permute(
                        [1, 2, 0])  # change (freq, chan, time) to (chan, time, freq)
                if start == 0:
                    dataset['X'] = temp
                    input_dim = list(temp.shape)
                else:
                    dataset['X'] = torch.cat((dataset['X'], temp), dim=0)
                start += (c.frames_per_block + 2)

            total_sample_cnt += cnt
            # tt1 = time.time()
            input_dim.insert(0, cnt)

            dataset['X'] = dataset['X'].reshape(input_dim)
            dataset['Y'] = torch.from_numpy(self.label_gen(label)).repeat(cnt, 1)
            torch.save(dataset, self.save_path + 'DataSet_' + str(file_idx) + '.pt')

            if self.norm:
                if file_idx == 1:
                    sum_temp = torch.sum(dataset['X'], dim=0)
                else:
                    sum_temp += torch.sum(dataset['X'], dim=0)
            # tt2 = time.time()
            # print(tt2 - tt1)
            # print(dataset['X'].shape)
            # print(dataset['Y'].shape)
        if self.norm:
            _mean = sum_temp / total_sample_cnt
            mean_name = 'Spdata_' + ('' if self.data_type == 'hoa' else 'stft_') + 'mean.pt'
            torch.save(_mean, mean_name)

            std_temp = torch.zeros(input_dim[1:], dtype=torch.float64)
            for i in range(file_idx):
                HOAdata_dict = torch.load(self.save_path + 'DataSet_' + str(i + 1) + '.pt')
                num = len(HOAdata_dict['X'])
                for j in range(num):
                    print('std...file {}, No.{}'.format(i, j))
                    std_temp += (HOAdata_dict['X'][j] - _mean) ** 2
            _std = np.sqrt(1.0 / (total_sample_cnt - 1) * std_temp)
            std_name = 'Spdata_' + ('' if self.data_type == 'hoa' else 'stft_') + 'std.pt'
            torch.save(_std, std_name)

    def label_gen(self, label):
        # direc_id = int(label) // 5
        direc_id = int(label) - 1
        aa = self.tf_list[0].split('_')

        read_path = (self.image_path + 'RT60_' + aa[4] + '/dist_' + aa[6] +
                     '/source_{}.binary').format(direc_id + 1)
        source_az = self.get_az(read_path)
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

    def get_az(self, path):
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
            for idx, angle in enumerate(az):
                if angle % 5 == 0:
                    continue
                if angle % 5 <= 2:
                    az[idx] = angle - angle % 5
                else:
                    az[idx] = (angle + 5) - angle % 5
        return az

    def get_HOA(self, s_fft):
        nFrames = s_fft.shape[2]
        hoa = np.zeros((c.n_freq, c.hoa_num, nFrames), dtype="complex")

        for freq_index, freq in enumerate(c.valid_freq_array):
            hoa[freq_index, :, :] = self.encoder[freq_index, :, :].dot(s_fft[freq_index, :, :])  # 文献中的b （26）
        return hoa

    def load_mat(self, path):
        ori_data = loadmat(path)

        keys = list(ori_data.keys())
        for key in keys:
            if key[0] != '_':
                data = ori_data[key]
                break

        return data

    def get_array_signal(self, mono_data, tf):
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

    def awgn(self, x, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(x ** 2) / x.size
        npower = xpower / snr
        return np.random.randn(*x.shape) * np.sqrt(npower) + x


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


class HOADataSet(data.Dataset):
    """
    Generate the appropriate format DataSet.

    """

    def __init__(self, path, index, data_type, is_speech=False):
        super(HOADataSet, self).__init__()
        self.readPath = path
        self.is_speech = is_speech
        self.data_type = data_type

        speech_term = 'Sp' if self.is_speech else ''
        data_type_term = 'stft_' if self.data_type == 'stft' else ''

        self.data_mean = torch.load(speech_term + 'data_' + data_type_term + 'mean.pt')
        self.data_std = torch.load(speech_term + 'data_' + data_type_term + 'std.pt')

        self.examples = torch.load(self.readPath + 'DataSet_' + str(index) + '.pt')
        self.X = self.examples['X']
        self.Y = self.examples['Y']

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
        TF_PATH = '/mnt/hd8t/cjf/TF_result_fs_16000/'
    elif SERVER == 'ship':
        IMAGE_PATH = '/home/cjf/workspace/stage2/RirsOfRooms/'
        TF_PATH = '/data/cjf/TF_result_fs_16000/'
    else:
        raise RuntimeError('Unrecognized server!')

    TRAIN_TF_LIST = [TF_PATH + i for i in [
        'RT60_0.583_dist_1.9572_TF_Matrix.mat',
        'RT60_0.47149_dist_1.6555_TF_Matrix.mat',
        'RT60_0.31077_dist_1.6948_TF_Matrix.mat']]
    VALID_TF_LIST = [TF_PATH + i for i in [
        'RT60_0.3687_dist_1.6557_TF_Matrix.mat',
        'RT60_0.56688_dist_1.3804_TF_Matrix.mat']]
    TEST_TF_LIST = [TF_PATH + i for i in [
        'RT60_0.32684_dist_1.3816_TF_Matrix.mat',
        'RT60_0.52589_dist_1.6324_TF_Matrix.mat']]

    TRAIN_FILE_PATH = './anechoic_mono_tr.flist'
    VALID_FILE_PATH = './anechoic_cv_speech.flist'
    TEST_FILE_PATH = './anechoic_mono_te.flist'
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


    for i in range(3600):
        print('std', i + 1)
        if i == 0:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/STFT/tr/DataSet_{}.pt'.format(i+1))
            data = data['X']
            std_temp = torch.sum((data - _mean) ** 2, dim=0)

        else:
            data = torch.load('/mnt/hd8t/cjf/random_reverb_wavs/STFT/tr/DataSet_{}.pt'.format(i + 1))
            data = data['X']
            std_temp += torch.sum((data - _mean) ** 2, dim=0)
    _std = np.sqrt(1.0 / (cnt - 1) * std_temp)
    torch.save(_std, 'data_stft_std.pt')
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
    '''
    # d_tr = DataProcessor(path=TRAIN_FILE_PATH, image_path=IMAGE_PATH, tf_list=TRAIN_TF_LIST,
    #                      snr_list=[10, 5, 0, -5], ser='sh', net='res',
    #                      is_tr='tr', data_type='hoa')
    # d_tr.run()
    # d_cv = DataProcessor(path=VALID_FILE_PATH, image_path=IMAGE_PATH, tf_list=VALID_TF_LIST,
    #                      snr_list=[10, 5, 0, -5], net='res', is_tr='cv',
    #                      is_speech=True, data_type='hoa', normalize=True)
    # d_cv.run()

    # d_tr_stft = DataProcessor(path=TRAIN_FILE_PATH, image_path=IMAGE_PATH, tf_list=TRAIN_TF_LIST,
    #                      snr_list=[10, 5, 0, -5], ser='sh', net='res',
    #                      is_tr='tr', data_type='stft')
    # d_tr_stft.run()
    # d_te = DataProcessor(path=TEST_FILE_PATH, image_path=IMAGE_PATH, tf_list=TEST_TF_LIST,
    #                      snr_list=[10, 5, 0, -5], net='res', is_tr='te',
    #                      is_speech=False, data_type='hoa', normalize=False)
    # d_te.run()
    d_te_stft = DataProcessor(path=TEST_FILE_PATH, image_path=IMAGE_PATH, tf_list=TEST_TF_LIST,
                         snr_list=[10, 5, 0, -5], net='res', is_tr='te',
                         is_speech=False, data_type='stft', normalize=False)
    d_te_stft.run()

'''
# get_az函数测试通过，标签没问题

直达声处于0, 标签[  0   0  70 290 180]
直达声处于5, 标签[  5   5  70 290 180]
直达声处于10, 标签[ 10   5  70 290 175]
直达声处于15, 标签[ 15  10  70 290 175]
直达声处于20, 标签[ 20  10  70 285 175]
直达声处于25, 标签[ 25  15  70 285 175]
直达声处于30, 标签[ 30  15  70 285 170]
直达声处于35, 标签[ 35  15  70 285 170]
直达声处于40, 标签[ 40  20  70 285 170]
直达声处于45, 标签[ 45  20  70 280 170]
直达声处于50, 标签[ 50  70  20 165 280]
直达声处于55, 标签[ 55  75  20 165 280]
直达声处于60, 标签[ 60  75  20 165 280]
直达声处于65, 标签[ 65  75  20 165 275]
直达声处于70, 标签[ 70  80  20 165 275]
直达声处于75, 标签[ 75  80  20 160 275]
直达声处于80, 标签[ 80  85  20 160 275]
直达声处于85, 标签[ 85  85  20 160 270]
直达声处于90, 标签[ 90  90 160  20 270]
直达声处于95, 标签[ 95  95 160  20 270]
直达声处于100, 标签[100  95 160  20 265]
直达声处于105, 标签[105 100 160  20 265]
直达声处于110, 标签[110 100 160  15 265]
直达声处于115, 标签[115 105 160  15 265]
直达声处于120, 标签[120 105 160  15 260]
直达声处于125, 标签[125 105 160  15 260]
直达声处于130, 标签[130 110 160  15 260]
直达声处于135, 标签[135 110 160  10 260]
直达声处于140, 标签[140 160 110 255  10]
直达声处于145, 标签[145 165 110 255  10]
直达声处于150, 标签[150 165 110 255  10]
直达声处于155, 标签[155 165 110 255   5]
直达声处于160, 标签[160 170 110 255   5]
直达声处于165, 标签[165 170 110 250   5]
直达声处于170, 标签[170 175 110 250   5]
直达声处于175, 标签[175 175 110 250   0]
直达声处于180, 标签[180 180 110 250   0]
直达声处于185, 标签[185 185 250 110 360]
直达声处于190, 标签[190 185 250 110 355]
直达声处于195, 标签[195 190 250 110 355]
直达声处于200, 标签[200 190 250 105 355]
直达声处于205, 标签[205 195 250 105 355]
直达声处于210, 标签[210 195 250 105 350]
直达声处于215, 标签[215 195 250 105 350]
直达声处于220, 标签[220 200 250 105 350]
直达声处于225, 标签[225 200 250 100 350]
直达声处于230, 标签[230 250 200 345 100]
直达声处于235, 标签[235 255 200 345 100]
直达声处于240, 标签[240 255 200 345 100]
直达声处于245, 标签[245 255 200 345  95]
直达声处于250, 标签[250 260 200 345  95]
直达声处于255, 标签[255 260 200 340  95]
直达声处于260, 标签[260 265 200 340  95]
直达声处于265, 标签[265 265 200 340  90]
直达声处于270, 标签[270 270 340 200  90]
直达声处于275, 标签[275 275 340 200  90]
直达声处于280, 标签[280 275 340 200  85]
直达声处于285, 标签[285 280 340 200  85]
直达声处于290, 标签[290 280 340 195  85]
直达声处于295, 标签[295 285 340 195  85]
直达声处于300, 标签[300 285 340 195  80]
直达声处于305, 标签[305 285 340 195  80]
直达声处于310, 标签[310 290 340 195  80]
直达声处于315, 标签[315 290 340 190  80]
直达声处于320, 标签[320 340 290  75 190]
直达声处于325, 标签[325 345 290  75 190]
直达声处于330, 标签[330 345 290  75 190]
直达声处于335, 标签[335 345 290  75 185]
直达声处于340, 标签[340 350 290  75 185]
直达声处于345, 标签[345 350 290  70 185]
直达声处于350, 标签[350 355 290  70 185]
直达声处于355, 标签[355 355 290  70 180]
'''
