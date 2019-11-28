# 此文件用于仿真数据生成，利用源数据和RIR卷积得到wav文件，并在.flist文件中记录wav文件的位置
# check pass
# 2019/9/24 version . From now on, I will save mono data instead of array data in .flist file,
# the corresponding operation, such as STFT or HOA, will be done in the DataProcessor class.
# So now, the degree value in the name of the wav file path in the .flist does not
# represents for its real direction anymore, instead, it will be used to generate signals of that direction.

import numpy as np 
import config as c
import random
import os
from scipy.io import wavfile as wav
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ser', type=str,
                    default='sh',  # sk : shengke 1 hao, sh: shrc
                    help='use which server to run')

args = parser.parse_args()
SERVER = args.ser


class DataGen(object):
    '''
    For white noise, this class generates both audio wavs and the wav_index_file;
    For speech, up to now, this class only generates the wav_index_file, because the speech data is already availiable from others.
    '''
    def __init__(self, data_type, num_of_wavs=None, time=None, is_speech=False):
        # 初始化时指定每个list文件包含多少个wav音频（不是总数，总数还需乘以方向数）， 每个音频的时长， 数据集的类型
        if num_of_wavs is not None:
            self.num_of_wavs = num_of_wavs
        if time is not None:
            self.time = time
            if SERVER == 'sh':
                self.save_path = '/mnt/hd8t/cjf/anechoic_wavs/'
            elif SERVER == 'sk':
                self.save_path = '/gpfs/share/home/1801213778/Dataset/wavs/'
            elif SERVER == 'ship':
                self.save_path = '/data/cjf/anechoic_wavs/'
            else:
                raise RuntimeError('Unrecognized server!')
        self.is_speech = is_speech

        self.type = data_type
        if not self.type in ['tr', 'cv', 'tt']:
            raise RuntimeError('Unrecognized data type!')

    def run(self):
        if not self.is_speech:
            with open('anechoic_mono_{}.flist'.format(self.type), 'w') as f:
                for i in range(c.scan_num):
                    for j in range(self.num_of_wavs):
                        print('{} {} {}'.format(self.type, i, j))
                        s_mono = np.random.randn(int(c.fs * self.time)) * np.sqrt(0.01)

                        # 角度deg的第idx个音频
                        file_name = 'wgn_deg{}_{}_idx_{}.wav'.format(i * 5, self.type, j)
                        all_path = self.save_path + file_name
                        wav.write(all_path, c.fs, s_mono)
                        # 将音频信息写入列表文件
                        f.write(all_path + ' ' + str(i+1) + '\n')
            # shuffle
            with open('anechoic_mono_{}.flist'.format(self.type), 'r') as f:
                _list = f.readlines()
                random.shuffle(_list)
            with open('anechoic_mono_{}.flist'.format(self.type), 'w') as f:
                for line in _list:
                    f.write(line)
        else:
            with open('anechoic_mono_speech_{}.flist'.format(self.type), 'w') as f:
                for _, _, files in os.walk('/mnt/hd8t/pchao/SpeechSeparation/mix/data/2speakers_new/wav8k/min/' + self.type + '/s1'):
                    file_index = -1
                    for name in files:
                        file_index += 1
                        print(file_index)
                        full_name = '/mnt/hd8t/pchao/SpeechSeparation/mix/data/2speakers_new/wav8k/min/' + self.type + '/s1/' + name
                        f.write(full_name + ' ' + str(file_index % 72 + 1) + '\n')
            # shuffle
            with open('anechoic_mono_speech_{}.flist'.format(self.type), 'r') as f:
                _list = f.readlines()
                random.shuffle(_list)
            with open('anechoic_mono_speech_{}.flist'.format(self.type), 'w') as f:
                for line in _list:
                    f.write(line)


if __name__ == '__main__':
    # data_gen_tr = DataGen(num_of_wavs=50, time=5, data_type='tr')
    # data_gen_cv = DataGen(num_of_wavs=20, time=5, data_type='cv')
    # data_gen_tr.run()
    # data_gen_cv.run()
    # print(data_gen.tf[10][0].shape)
    # pass
    data_gen_tr = DataGen(data_type='tr', is_speech=True)
    data_gen_tr.run()
    data_gen_cv = DataGen(data_type='cv', is_speech=True)
    data_gen_cv.run()
    data_gen_tt = DataGen(data_type='tt', is_speech=True)
    data_gen_tt.run()