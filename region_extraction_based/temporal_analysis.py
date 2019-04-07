# -*- coding:utf-8 -*-
__author__ = 'shichao'
from video_frequency import *






def fft_demo():
    import numpy as np
    from scipy.fftpack import fft, ifft, dct
    import matplotlib.pyplot as plt
    freq_function = dct
    #采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
    # x=np.linspace(0,1,1400)

    x=np.linspace(0,100,20)
    # y = np.sin(2*np.pi*x)
    # y = np.random.randint(10,15,20)
    # y = [1,100,90,95,3,2,4,1,5,1,3,4,5,1,4,5,2,4,5,3]
    # y = np.ones((20,1))
    # y = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    # y = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    y = [13,11,12,14,15,14,14,17,13,15,12,11,14,17,15,11,13,12,15,11]
    # y = [50,25,38,60,70,55,55,80,50,70,38,25,60,80,70,25,50,38,70,25]
    #设置需要采样的信号，频率分量有180，390和600
    # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)


    yy=freq_function(y)                     #快速傅里叶变换
    yreal = yy.real               # 获取实数部分
    yimag = yy.imag               # 获取虚数部分

    yf=abs(freq_function(y))                # 取绝对值
    yf1=abs(freq_function(y))/len(x)           #归一化处理
    yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

    xf = np.arange(len(y))        # 频率
    xf1 = xf
    xf2 = xf[range(int(len(x)/2))]  #取一半区间


    plt.subplot(221)
    plt.plot(x[0:50],y[0:50])
    plt.title('Original wave')

    plt.subplot(222)
    plt.plot(xf,yf,'r')
    plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

    plt.subplot(223)
    plt.plot(xf1,yf1,'g')
    plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')

    plt.subplot(224)
    plt.plot(xf2,yf2,'b')
    plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')


    plt.show()

def stepSig_demo():
    from scipy.signal import lti, step2, impulse2
    import matplotlib.pyplot as plt

    s1 = lti([3], [1, 2, 10])  # 以分子分母的最高次幂降序的系数构建传递函数，s1=3/(s^2+2s+10）
    s2 = lti([1], [1, 0.4, 1])  # s2=1/(s^2+0.4s+1)
    s3 = lti([5], [1, 2, 5])  # s3=5/(s^2+2s+5)

    t1, y1 = step2(s1)  # 计算阶跃输出，y1是Step response of system.
    t2, y2 = step2(s2)
    t3, y3 = step2(s3)
    t11, y11 = impulse2(s1)
    t22, y22 = impulse2(s2)
    t33, y33 = impulse2(s3)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')  # 开启subplots模式
    ax1.plot(t1, y1, 'r', label='s1 Step Response', linewidth=0.5)
    ax1.set_title('s1 Step Response', fontsize=9)
    ax2.plot(t2, y2, 'g', label='s2 Step Response', linewidth=0.5)
    ax2.set_title('s2 Step Response', fontsize=9)
    ax3.plot(t3, y3, 'b', label='s3 Step Response', linewidth=0.5)
    ax3.set_title('s3 Step Response', fontsize=9)

    ax4.plot(t11, y11, 'm', label='s1 Impulse Response', linewidth=0.5)
    ax4.set_title('s1 Impulse Response', fontsize=9)
    ax5.plot(t22, y22, 'y', label='s2 Impulse Response', linewidth=0.5)
    ax5.set_title('s2 Impulse Response', fontsize=9)
    ax6.plot(t33, y33, 'k', label='s3 Impulse Response', linewidth=0.5)
    ax6.set_title('s3 Impulse Response', fontsize=9)

    ##plt.xlabel('Times')
    ##plt.ylabel('Amplitude')
    # plt.legend()
    plt.show()

def freqFilter_demo():
    from scipy import signal
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    N = 500
    fs = 5
    n = [2 * math.pi * fs * t / N for t in range(N)]
    axis_x = np.linspace(0, 1, num=N)
    # 频率为5Hz的正弦信号


    x = [math.sin(i) for i in n]
    plt.subplot(221)
    plt.plot(axis_x, x)
    plt.title(u'5Hz sin')
    plt.axis('tight')

    xx = []
    x1 = [math.sin(i * 10) for i in n]

    for i in range(len(x)):

        xx.append(x[i] + x1[i])

    plt.subplot(222)
    plt.plot(axis_x, xx)
    plt.title(u'5Hz and 50Hz mix sin')
    plt.axis('tight')

    b, a = signal.butter(3, 0.08, 'low')

    sf = signal.filtfilt(b, a, xx)

    plt.subplot(223)
    plt.plot(axis_x, sf)
    plt.title(u'low pass filter')
    plt.axis('tight')

    b, a = signal.butter(3, 0.10, 'high')
    sf = signal.filtfilt(b, a, xx)

    plt.subplot(224)
    plt.plot(axis_x, sf)
    plt.title(u'high pass filter')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    # fft_demo()
    # stepSig_demo()
    freqFilter_demo()