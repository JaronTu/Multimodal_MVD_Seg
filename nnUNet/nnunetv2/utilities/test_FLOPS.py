import torch
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from thop import profile
from torchvision.models import resnet50

# device = torch.device('cuda:1')
# model = resnet50().to(device)
#
# input1 = torch.randn(4, 3, 224, 224).to(device)
# flops, params = profile(model, inputs=(input1, ))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

def walltime(func, x, y):
    # print(func)
    start_time = time.time()
    result = func(x, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def apply_function(func, x, y):
    return func(x, y)

device = torch.device('cuda:1')

matmul_tflops = defaultdict(lambda: {})
for n in [128, 512, 2048, 8192]:  # 四种大小的矩阵
    for dtype in (torch.float32, torch.float16):
        a = torch.randn(n, n, dtype=dtype).to(device)
        b = torch.randn(n, n, dtype=dtype).to(device)
        t = walltime(torch.matmul, a, b)  # 计算两个矩阵相乘的时间
        matmul_tflops[f'n={n}'][
            dtype] = 2 * n ** 3 / t / 1e12  # 计算TFLOPS：两个n*n的矩阵相乘会继续2*n**3次计算，再除以计算的时间，和1e12(1tplops=1e12),得到tflops
        del a, b

df = pd.DataFrame(matmul_tflops)
print(df)

vector = defaultdict(lambda: {})
for n in [1024 * 64, 1024 * 256, 1024 * 1024, 1024 * 1024 * 4]:
    a = torch.randn(n).to(device)
    t = walltime(torch.mul, a, 1.2)  # 进行向量乘法操作
    vector[n]['TFLOPS'] = n / t / 1e12  # 计算TFPLOS
    vector[n]['GB/s'] = 8 * n / t / 1e9  # 计算带宽：进行一个向量乘法需要将数据从gpu拿得到计算单元（4byte），再从计算单元再拿回来（4byte），所以是8*n

df = pd.DataFrame(vector)
print(df)