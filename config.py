from const import *
import torch
# SOME HYPER
fft_point = 512 # for consistence with hoa, equavlent to NFFT.
fs = 16000
frame_time = 20e-3
frame_size = int(frame_time * fs)

frame_step = int(10e-3 * fs)
n_overlap = frame_size - frame_step
frames_per_block = 20
block_size = frame_step * (frames_per_block - 1) + frame_size

n_chan = 32
tf_len = 8192

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

# for clipping out the silence in a speech
gamma = 0.1