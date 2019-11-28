from const import *
import torch
# SOME HYPER
fft_point = 1024 # for consistence with hoa, equavlent to NFFT.
fs = 8000
frame_time = 20e-3
frame_size = 1024 # int(frame_time * fs)

frame_step = 512 # int(10e-3 * fs)
n_overlap = frame_size - frame_step
frames_per_block = 20
block_size = frame_step * (frames_per_block - 1) + frame_size

n_chan = 32
tf_len = 8192 / 2

# ===========================================================================
# computer weights for each freq
freq_array = np.linspace(0, fs/2, fft_point//2+1)
valid_freq_min = 0
valid_freq_max = fs/2
valid_freq_index = (freq_array > valid_freq_min) & (freq_array < valid_freq_max)
valid_freq_array = freq_array[valid_freq_index]
n_freq = valid_freq_array.size
weight_array = np.ones_like(valid_freq_index)
# ============================================================================
# for HOA
min_freq, max_freq = 0, fs // 2
hoa_order = 4
hoa_num = 25

# scan params ==============================
az_num, el_num = 361, 1   # for compatibility with original code of decoder
scan_num = 72             # really used in this module
ref_el_index = 0  # this program mainly run on equator.
az_max, az_min = 2*np.pi, 0
el_max, el_min = 0, np.pi
az_array = np.linspace(az_min, az_max, scan_num)
el_array = np.linspace(el_min, el_max, el_num) if el_num > 1 else np.asarray([np.pi / 2])

# ============================
mic_position = mic_position_32mic
array_radius = array_radius_32mic

speed_of_sound = 344  # m/s

# for time domain shift
tf_max_len = int(fs * 0.1)

# for HOANet
first_out_chan = 64
sec_out_chan = 128
thi_out_chan = 256
first_out_fc = 1080
sec_out_fc = 360

std = 8
# for normalize
# data_hoa_mean = torch.load('data_hoa_mean.pt')
# data_hoa_std = torch.load('data_hoa_std.pt')
#
# data_stft_mean = torch.load('data_stft_mean.pt')
# data_stft_mean = torch.load('data_stft_mean.pt')

# for peak detection
sigma = 8
thres = 0.5

# snr
snr = 10
snr_list = [10, 5, 0, -5]
# for clipping out the silence in a speech
gamma = 0.1

# paths

IMAGE_PATH = '/home/cjf/workspace/Matlab/RirsOfRooms/'
TF_PATH = '/mnt/hd8t/cjf/TF_result_fs_8000/'
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
SE_data_save_path = '/mnt/hd8t/cjf/SE_data/'
data_read_path = '/mnt/hd8t/cjf/SE_data/'

# dictionary for direct sound to reflection
check_ref_angle = {
0: [0, 0, 70, 290, 180],
5: [5, 5, 70, 290, 180],
10: [10, 5, 70, 290, 175],
15: [15, 10, 70, 290, 175],
20: [20, 10, 70, 285, 175],
25: [25, 15, 70, 285, 175],
30: [30, 15, 70, 285, 170],
35: [35, 15, 70, 285, 170],
40: [40, 20, 70, 285, 170],
45: [45, 20, 70, 280, 170],
50: [50, 70, 20, 165, 280],
55: [55, 75, 20, 165, 280],
60: [60, 75, 20, 165, 280],
65: [65, 75, 20, 165, 275],
70: [70, 80, 20, 165, 275],
75: [75, 80, 20, 160, 275],
80: [80, 85, 20, 160, 275],
85: [85, 85, 20, 160, 270],
90: [90, 90, 160, 20, 270],
95: [95, 95, 160, 20, 270],
100: [100, 95, 160, 20, 265],
105: [105, 100, 160, 20, 265],
110: [110, 100, 160, 15, 265],
115: [115, 105, 160, 15, 265],
120: [120, 105, 160, 15, 260],
125: [125, 105, 160, 15, 260],
130: [130, 110, 160, 15, 260],
135: [135, 110, 160, 10, 260],
140: [140, 160, 110, 255, 10],
145: [145, 165, 110, 255, 10],
150: [150, 165, 110, 255, 10],
155: [155, 165, 110, 255,  5],
160: [160, 170, 110, 255,  5],
165: [165, 170, 110, 250,  5],
170: [170, 175, 110, 250,  5],
175: [175, 175, 110, 250,  0],
180: [180, 180, 110, 250,  0],
185: [185, 185, 250, 110, 0],
190: [190, 185, 250, 110, 355],
195: [195, 190, 250, 110, 355],
200: [200, 190, 250, 105, 355],
205: [205, 195, 250, 105, 355],
210: [210, 195, 250, 105, 350],
215: [215, 195, 250, 105, 350],
220: [220, 200, 250, 105, 350],
225: [225, 200, 250, 100, 350],
230: [230, 250, 200, 345, 100],
235: [235, 255, 200, 345, 100],
240: [240, 255, 200, 345, 100],
245: [245, 255, 200, 345,  95],
250: [250, 260, 200, 345,  95],
255: [255, 260, 200, 340,  95],
260: [260, 265, 200, 340,  95],
265: [265, 265, 200, 340,  90],
270: [270, 270, 340, 200,  90],
275: [275, 275, 340, 200,  90],
280: [280, 275, 340, 200,  85],
285: [285, 280, 340, 200,  85],
290: [290, 280, 340, 195,  85],
295: [295, 285, 340, 195,  85],
300: [300, 285, 340, 195,  80],
305: [305, 285, 340, 195,  80],
310: [310, 290, 340, 195,  80],
315: [315, 290, 340, 190,  80],
320: [320, 340, 290, 75, 190],
325: [325, 345, 290, 75, 190],
330: [330, 345, 290, 75, 190],
335: [335, 345, 290, 75, 185],
340: [340, 350, 290, 75, 185],
345: [345, 350, 290, 70, 185],
350: [350, 355, 290, 70, 185],
355: [355, 355, 290, 70, 180]}
