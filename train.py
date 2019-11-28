import torch
import torch.nn as nn
import torch.utils.data as data
import logging
import config as c
import argparse
import warnings
import random
from net import HOANet, ResBlock, ResNet
from DataPreprocessor import HOADataSet
from draw import *
from handler import peak_detection, cal_precision, cal_recall

# ================= read parameters from cmd, for run mode ====================

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,
                    default='test',
                    help='The name or function of the current task')
parser.add_argument('--ser', type=str,
                    default='sh',  # sk : shengke 1 hao, sh: shrc
                    help='use which server to run')
parser.add_argument('--net', type=str,
                    default='res',  # sk : shengke 1 hao, sh: shrc
                    help='use which network')
# parser.add_argument('--snr', type=int,
#                     default=0,  # sk : shengke 1 hao, sh: shrc
#                     help='set a snr for validation')
parser.add_argument('--data', type=str,
                    default='hoa',  # sk : shengke 1 hao, sh: shrc
                    help='use hoa data or stft data')
parser.add_argument('--issp', type=int,
                    default='1',  # sk : shengke 1 hao, sh: shrc
                    help='speech or not')
parser.add_argument('--debug', type=int,
                    default=0,
                    help='debug mode or not')
parser.add_argument('--gpu', type=int,
                    default=1,  # sk : shengke 1 hao, sh: shrc
                    help='use GPU or CPU')

args = parser.parse_args()

CUR_TASK = args.name
SERVER = args.ser
NET_TYPE = args.net
# SNR = args.snr
SPEECH = args.issp
DATA_TYPE = args.data
DEVICE_TYPE = args.gpu
DEBUG = args.debug
# some directory and  Device configuration
if SERVER == 'sk':
    DATA_PATH = '/gpfs/share/home/1801213778/Dataset/random_reverb_wavs/' + ('speech/' if SPEECH == 1 else '')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
elif SERVER == 'sh':
    DATA_PATH = '/mnt/hd8t/cjf/random_reverb_wavs/' + ('speech/' if SPEECH == 1 else '')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
elif SERVER == 'ship':
    DATA_PATH = '/data/cjf/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if DEVICE_TYPE == 1 else torch.device('cpu')
else:
    raise RuntimeError('Unrecognized server!')

# Hyper
num_epochs = 30
batch_size = 128
running_lr = False
decay = 0.04  # lr decay
numOfEth = 3
snr_list = [10, 5, 0, -5]
TRAIN_FILE_NUM = 50 * 72 if SPEECH == 0 else 3340
VALID_FILE_NUM = 20 * 72 if SPEECH == 0 else 840
TEST_FILE_NUM = 20 * 72 if SPEECH == 0 else 500
bottle = False
lr = 0.001  # + list(1.0 / np.random.randint(1000, 2000, size=2))
weight_decay = 0.0001
num_of_res_blocks = 3

name = CUR_TASK + '_bot' + str(int(bottle)) + \
       '_lr' + str(lr) + '_wd' + str(weight_decay) + '_#res' + str(num_of_res_blocks)
# ========================================================
# recording config
logging.basicConfig(level=logging.DEBUG,
                    filename=name + '.log',
                    filemode='w',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

train_file_order = list(range(TRAIN_FILE_NUM))
valid_file_order = list(range(VALID_FILE_NUM))

# set model
input_shape_dict = {'hoa hoa': (c.n_freq, c.frames_per_block + 2, c.hoa_num * 2),
                    'hoa stft': (c.n_freq, c.frames_per_block + 2, c.n_chan * 2),
                    'res hoa': (c.hoa_num * 2, c.frames_per_block + 2, c.n_freq),
                    'res stft': (c.n_chan * 2, c.frames_per_block + 2, c.n_freq)
                    }
input_shape = input_shape_dict[' '.join((NET_TYPE, DATA_TYPE))]

index_offset_dict = {10: 0, 5: int(TEST_FILE_NUM / 4), 0: int(TEST_FILE_NUM / 2), -5: int(TEST_FILE_NUM / 4 * 3)}


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def awgn_gpu(x, snr, device):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / tensor_size(x)
    npower = xpower / snr
    return torch.randn(x.shape).to(device) * np.sqrt(npower.cpu().detach().numpy()) + x


def tensor_size(x):
    a = list(x.size())
    _mul = 1
    for item in a:
        _mul *= item
    return _mul


# Train the model
def train_and_valid(learning_rate=lr, weight_decay=weight_decay,
                    num_of_res=num_of_res_blocks, if_bottleneck=bottle, plot=True):
    """
    Train the model and run it on the valid set every epoch
    :param weight_decay: for L2 regularzition
    :param bottleneck:
    :param num_of_res:
    :param learning_rate: lr
    :param plot: draw the train/valid loss curve or not
    :return:
    """
    curr_lr = learning_rate
    # model define
    if NET_TYPE == 'res':
        if DATA_TYPE == 'hoa':
            block = ResBlock(128, 128, bottleneck=if_bottleneck)
        else:
            block = ResBlock(256, 256, bottleneck=if_bottleneck)
        model = ResNet(block, numOfResBlock=num_of_res, input_shape=input_shape, data_type=DATA_TYPE).to(device)
    elif NET_TYPE == 'hoa':
        model = HOANet(input_shape=input_shape).to(device)
    else:
        raise RuntimeError('Unrecognized net type!')
    # print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # These parameters are for searching the best epoch to early stopping
    train_loss_curve, valid_loss_curve = [], []
    best_loss, avr_valid_loss = 10000.0, 0.0

    best_epoch = 0
    best_model = None  # the best parameters

    for epoch in range(num_epochs):
        # 每一轮的 训练集/验证集 误差
        train_loss_per_epoch, valid_loss_per_epoch = 0.0, 0.0
        train_step_cnt, valid_step_cnt = 0, 0

        train_data, valid_data = [], []
        # 进入训练模式
        model.train()
        random.shuffle(train_file_order)

        for idx, train_idx in enumerate(train_file_order):
            if len(train_data) < batch_size:
                train_data_temp = HOADataSet(path=DATA_PATH + ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'tr/',
                                             index=train_idx + 1, data_type=DATA_TYPE, is_speech=SPEECH)
                if len(train_data) == 0:
                    train_data = train_data_temp
                else:
                    train_data += train_data_temp
                continue

            train_loader = data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

            for step, (examples, labels) in enumerate(train_loader):
                # if step == 1:
                #     break
                train_step_cnt += 1
                # print(train_step_cnt)
                examples = examples.float().to(device)
                labels = labels.float().to(device)
                outputs = model(examples)
                train_loss = criterion(outputs, labels)
                train_loss_per_epoch += train_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                logger.info("Epoch [{}/{}], Step {}, train Loss: {:.4f}"
                            .format(epoch + 1, num_epochs, train_step_cnt, train_loss.item()))

            train_data = HOADataSet(path=DATA_PATH + ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'tr/',
                                    index=train_idx + 1, data_type=DATA_TYPE, is_speech=SPEECH)

        if plot:
            train_loss_curve.append(train_loss_per_epoch / train_step_cnt)

        if running_lr and epoch > 1 and (epoch + 1) % 2 == 0:
            curr_lr = curr_lr * (1 - decay)
            update_lr(optimizer, curr_lr)

        # valid every epoch
        # 进入验证模式

        model.eval()
        with torch.no_grad():
            for idx, valid_idx in enumerate(valid_file_order):
                if len(valid_data) < batch_size:
                    valid_data_temp = HOADataSet(path=DATA_PATH + ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'cv/',
                                                 index=valid_idx + 1, data_type=DATA_TYPE, is_speech=SPEECH)
                    if len(valid_data) == 0:
                        valid_data = valid_data_temp
                    else:
                        valid_data += valid_data_temp
                    continue

                valid_loader = data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)

                for step, (examples, labels) in enumerate(valid_loader):
                    valid_step_cnt += 1
                    # print(valid_step_cnt)
                    examples = examples.float().to(device)
                    labels = labels.float().to(device)

                    outputs = model(examples)
                    valid_loss = criterion(outputs, labels)
                    valid_loss_per_epoch += valid_loss.item()

                    logger.info('The loss for the current batch:{}'.format(valid_loss))
                
                valid_data = HOADataSet(path=DATA_PATH + ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'cv/',
                                    index=valid_idx + 1, data_type=DATA_TYPE, is_speech=SPEECH)

            avr_valid_loss = valid_loss_per_epoch / valid_step_cnt

            logger.info('Epoch {}, the average loss on the valid set: {} '.format(epoch, avr_valid_loss))

            valid_loss_curve.append(avr_valid_loss)
            if avr_valid_loss < best_loss:
                best_loss = avr_valid_loss
                best_epoch, best_model = epoch, model.state_dict()

    # end for loop of epoch
    torch.save({
        'epoch': best_epoch,
        'state_dict': best_model,
        'loss': best_loss,
    }, './models/ckpoint_' + CUR_TASK + '_bot' + str(int(if_bottleneck)) +
       '_lr' + str(learning_rate) + '_wd' + str(weight_decay) + '_#res' + str(num_of_res) + '.tar')

    logger.info('best epoch:{}, valid loss:{}'.format(best_epoch, best_loss))
    if plot:
        x = np.arange(num_epochs)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, train_loss_curve, 'b', label='Train Loss')
        ax.plot(x, valid_loss_curve, 'r', label='Valid Loss')
        plt.legend(loc='upper right')
        plt.savefig(name + '.jpg')
        plt.close()


# verify the model checkpoint
if __name__ == '__main__':
    # for random searching hyper
    train_and_valid()

    logger.info('\n\n====================Training finished=========================')
    logger.info('========================Validation on valid set====================')

    # 读取checkpoint保存的模型，在验证集上跑一遍，计算准确率和召回率
    path = './models/ckpoint_' + name + '.tar'
    checkpoint = torch.load(path)
    # input_shape = (c.hoa_num * 2, c.frames_per_block+2, c.n_freq)
    if DATA_TYPE == 'hoa':
        block = ResBlock(128, 128, bottleneck=bottle)
    else:
        block = ResBlock(256, 256, bottleneck=bottle)
    model = ResNet(block, numOfResBlock=num_of_res_blocks, input_shape=input_shape, data_type=DATA_TYPE).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    criterion = nn.MSELoss()
    model.eval()
    f = open(name + '.txt', 'w')
    file_ = open('anechoic_mono_speech_tt.flist', 'r')
    all_test_files = file_.readlines()
    with torch.no_grad():
        for snr in snr_list:
            offset = index_offset_dict[snr]
            total_correct = np.zeros(numOfEth)  # 每个误差容忍度都对应一个准确个数
            total_recall = np.zeros(numOfEth)
            total_peaks, total_predict = 0, 0  # 总共的真实峰值数，总共预测出来的峰值数

            valid_step_cnt = 0
            total_valid_loss = 0.0
            no_peak = np.zeros(c.scan_num)  # 记录无峰值输出（全0输出）的样本个数
            total = 0  # 验证集的总样本个数

            total_correct_each = np.zeros((360, 3))
            total_recall_each = np.zeros((360, 3))
            total_peaks_each = np.zeros(360)
            total_predict_each = np.zeros(360)

            for idx in range(offset + 1, int(offset + TEST_FILE_NUM / 4 + 1)):
                direct_label = int(all_test_files[idx - 1].strip().split(' ')[1]) - 1
                _str = ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'tt/'
                valid_data = HOADataSet(path=DATA_PATH + ('' if DATA_TYPE == 'hoa' else 'STFT/') + 'tt/',
                                        index=idx, data_type=DATA_TYPE, is_speech=SPEECH)
                valid_loader = data.DataLoader(dataset=valid_data,
                                               batch_size=1,
                                               shuffle=True)
                total += len(valid_data)
                for step, (examples, labels) in enumerate(valid_loader):
                    valid_step_cnt += 1
                    # print(valid_step_cnt)
                    examples = examples.float().to(device)
                    labels = labels.float().to(device)
                    outputs = model(examples)
                    valid_loss = criterion(outputs, labels.squeeze())
                    total_valid_loss += valid_loss.item()

                    real_peaks = peak_detection(labels.squeeze())
                    pred_peaks = peak_detection(outputs.squeeze())

                    if len(pred_peaks) == 0:
                        no_peak[direct_label] += 1
                        warnings.warn('没有峰值！\n')
                        '''
                        plt.figure()
                        plt.plot(outputs.squeeze().cpu().detach().numpy())
                        term = 'STFT/' if DATA_TYPE == 'stft' else ''
                        plt.savefig(
                                './picture/' + term + 'abnormal{}deg_{}_{}.jpg'.format(direct_label * 5,
                                                                           no_peak[direct_label],
                                                                           real_peaks))
                        plt.close()
                        '''
                        # logger.info('no peak, direct sound at {}'.format((int(label) - 1) * 5))
                        continue

                    prec = cal_precision(pred_peaks, real_peaks)
                    recall = cal_recall(pred_peaks, real_peaks)

                    total_correct += prec * len(pred_peaks)
                    total_recall += recall * len(real_peaks)
                    total_peaks += len(real_peaks)
                    total_predict += len(pred_peaks)

                    if DEBUG:
                        total_correct_each[direct_label] += prec * len(pred_peaks)
                        total_recall_each[direct_label] += recall * len(real_peaks)
                        total_peaks_each[direct_label] += len(real_peaks)
                        total_predict_each[direct_label] += len(pred_peaks)

                    logger.info('====================================')
                    # logger.info('直达声处于{}°'.format((int(label) - 1) * 5))
                    logger.info('real peaks:{}'.format(real_peaks))
                    logger.info('pred peaks:{}'.format(pred_peaks))
                    logger.info('precision:{}, recall:{}'.format(prec, recall))
                    logger.info('The loss for the current batch:{}'.format(valid_loss))

            avr_valid_loss = total_valid_loss / valid_step_cnt
            assert total == valid_step_cnt
            # assert total_predict_each.sum() == total_predict
            # assert total_peaks_each.sum() == total_peaks
            # assert (total_recall_each.sum(axis=0) == total_recall).all()
            # assert (total_correct_each.sum(axis=0) == total_correct).all()

            _prec = total_correct / total_predict
            _recall = total_recall / total_peaks
            if DEBUG:
                prec_each = total_correct_each / total_predict_each[:, np.newaxis]
                recall_each = total_recall_each / total_peaks_each[:, np.newaxis]
                term = 'STFT' if DATA_TYPE == 'stft' else ''
                np.save(term + 'prec_rec_for_each_snr_{}.npy'.format(snr), {'p': prec_each, 'r': recall_each},
                        allow_pickle=True)
                np.save(term + 'np_peak_for_each_deg_snr{}.npy'.format(snr), no_peak, allow_pickle=True)

            f.write('SNR:{}dB, Epoch {}, the average precision on the valid set:{}\n'.format(snr, checkpoint['epoch'],
                                                                                             _prec))
            f.write('SNR:{}dB, Epoch {}, the average recall on the valid set:{}\n'.format(snr, checkpoint['epoch'],
                                                                                          _recall))
            f.write('SNR:{}dB, no-peak examples:[{}/{}]\n'.format(snr, int(no_peak.sum()), total))
    f.close()
    file_.close()
