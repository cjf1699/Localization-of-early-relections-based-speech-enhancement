import torch
import time
import numpy as np

device = torch.device('cuda')


def awgn_tensor(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / tensor_size(x)
    npower = xpower / snr
    return torch.randn(x.shape) * np.sqrt(npower) + x


def awgn_gpu(x, snr, device):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x ** 2) / tensor_size(x)
    npower = xpower / snr
    print(torch.randn(x.shape).to(device) * np.sqrt(0.01) + x)

    return torch.randn(x.shape).to(device) * np.sqrt(npower) + x


def tensor_size(x):
    a = list(x.size())
    _mul = 1
    for item in a:
        _mul *= item
    return _mul


def awgn(x, snr):

    snr = 10 ** (snr / 10.0)

    xpower = np.sum(x ** 2) / x.size

    npower = xpower / snr

    return np.random.randn(*x.shape) * np.sqrt(npower) + x


if __name__ == '__main__':
    # verify awgn_tensor, check pass
    # a = torch.randn(128, 50, 22, 255)
    # b = awgn_tensor(a, 10)
    # snr = 10 * np.log10(torch.sum(a ** 2) / torch.sum((b-a) ** 2))
    # print(snr, 'dB')

    t1 = time.time()
    a = torch.randn(128, 50, 22, 255)
    b = awgn_gpu(a, 10).to(device)
    t2 = time.time()
    print('tensor, gpu:', t2-t1)

    t1 = time.time()
    a = torch.randn(128, 50, 22, 255).to(device)
    b = awgn_tensor(a, 10)

    t2 = time.time()
    print('tensor, cpu:', t2-t1)
    print(b.device)

    t1 = time.time()
    a = torch.randn(128, 50, 22, 255).to(device)
    b = awgn(a.cpu().detach().numpy(), 10)
    c = torch.from_numpy(b).to(device)
    t2 = time.time()
    print('gpu:', t2-t1)
    t1 = time.time()
    a = torch.randn(128, 50, 22, 255)
    b = awgn(a.numpy(), 10)
    b = torch.from_numpy(b).to(device)
    t2 = time.time()
    print('cpu:', t2-t1)

