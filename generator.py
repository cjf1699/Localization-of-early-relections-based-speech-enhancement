# 此文件用于仿真数据生成，利用源数据和RIR卷积得到wav文件，并在.flist文件中记录wav文件的位置
# check pass
# 2019/9/24 version . From now on, I will save mono data instead of array data in .flist file,
# the corresponding operation, such as STFT or HOA, will be done in the DataProcessor class.
# So now, the degree value in the name of the wav file path in the .flist does not
# represents for its real direction anymore, instead, it will be used to generate signals of that direction.

import numpy as np 
import config as c
import random
from scipy.io import wavfile as wav
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ser', type=str,
                    default='sh',  # sk : shengke 1 hao, sh: shrc
                    help='use which server to run')

args = parser.parse_args()
SERVER = args.ser


class DataGen(object):
    def __init__(self, num_of_wavs, time, data_type):
        # 初始化时指定每个list文件包含多少个wav音频（不是总数，总数还需乘以方向数）， 每个音频的时长， 数据集的类型
        self.num_of_wavs = num_of_wavs
        self.time = time
        if SERVER == 'sh':
            self.save_path = '/mnt/hd8t/cjf/anechoic_wavs/'
        elif SERVER == 'sk':
            self.save_path = '/gpfs/share/home/1801213778/Dataset/wavs/'
        elif SERVER == 'ship':
            self.save_path = '/data/cjf/anechoic_wavs/'
        else:
            raise RuntimeError('Unrecognized server!')
        self.type = data_type

    def run(self):
        with open('anechoic_mono_{}.flist'.format(self.type), 'w') as f:
            for i in range(c.scan_num):
                for j in range(self.num_of_wavs):
                    print('{} {} {}'.format(self.type, i, j))
                    # s_mono = np.random.randn(int(c.fs * self.time)) * np.sqrt(0.01)

                    # 角度deg的第idx个音频
                    file_name = 'wgn_deg{}_{}_idx_{}.wav'.format(i * 5, self.type, j)
                    all_path = self.save_path + file_name
                    # wav.write(all_path, c.fs, s_mono)
                    # 将音频信息写入列表文件
                    f.write(all_path + ' ' + str(i+1) + '\n')
        # shuffle
        with open('anechoic_mono_{}.flist'.format(self.type), 'r') as f:
            _list = f.readlines()
            random.shuffle(_list)
        with open('anechoic_mono_{}.flist'.format(self.type), 'w') as f:
            for line in _list:
                f.write(line)


if __name__ == '__main__':
    # data_gen_tr = DataGen(num_of_wavs=50, time=5, data_type='tr')
    # data_gen_cv = DataGen(num_of_wavs=20, time=5, data_type='cv')
    # data_gen_tr.run()
    # data_gen_cv.run()
    # print(data_gen.tf[10][0].shape)
    # pass
    data_gen_te = DataGen(num_of_wavs=20, time=5, data_type='te')
    data_gen_te.run()