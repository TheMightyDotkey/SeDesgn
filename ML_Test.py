
print('test')

import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

def datasetmaker(offset, matfilename):
    """offset is multiple of 256 matfile name is output name"""

    Fs = 12000
    T = 1/Fs
    L = 256
    t = np.arange(0, L) * T
    a = 0 + 256 * offset
    b = 255 + 256 * offset
    matfilenamedat = "{}.mat" .format(matfilename)
    vectorname = "fftdata{}" .format(matfilename)
    S = data[a:b]

    f = Fs * np.arange(0, L//2 + 1) / L

    Y = np.fft.fft(S, axis = 0)
    P2 = abs(Y/L)
    P1 = P2[0:L//2+1]
    P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]
    P1 = P1/max(P1)

    sp.savemat(matfilenamedat, {vectorname:P1})
    sp.savemat('timefile.mat', {'time':f})


bearingdata = sp.loadmat('normal.mat')
sorted(bearingdata.keys())
print(bearingdata)

data = bearingdata['X097_DE_time']
print(data)

datasetmaker(0, 'normalset1')
datasetmaker(1, 'normalset2')
datasetmaker(2, 'normalset3')
datasetmaker(3, 'normalset4')
datasetmaker(4, 'normalset5')
datasetmaker(5, 'normalset6')
datasetmaker(6, 'normalset7')
datasetmaker(7, 'normalset8')
datasetmaker(8, 'normalset9')
datasetmaker(9, 'normalset10')




#N = 1000
#T = 1 / 800
#x = np.linspace(0, N*T, N-1)
#y = 2*np.sin(20*x*2*np.pi) + 6*np.sin(150*x*2*np.pi)
#plt.plot(y)
#ffty = np.fft.fft(y)
#Spec = abs(ffty/N)
#Dat = Spec[1:N//2+1]
#Dat[2:len(Dat)-1] = 2 * Dat[2:len(Dat)-1]
#f = 800 * np.linspace(0, 1, N//2) / N
#plt.plot(f, Dat)


#Fs = 1000
#T = 1/Fs
#L = 1500
#t = np.arange(0, L) * T

#S = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)

#f = Fs * np.arange(0, L//2 + 1) / L

#Y = np.fft.fft(S)
#P2 = abs(Y/L)
#P1 = P2[0:L//2+1]
#P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]

#plt.plot(f,P1)


#plt.plot(fftx, np.abs(ffty[0:N//2]) * 2 / len(ffty))

#plt.show()

#Fs = 12000
#T = 1/Fs
#L = 256
#t = np.arange(0, L) * T

#S = data[0:256]

#f = Fs * np.arange(0, L//2 + 1) / L

#Y = np.fft.fft(S, axis = 0)
#P2 = abs(Y/L)
#P1 = P2[0:L//2+1]
#P1[1:len(P1)-1] = 2*P1[1:len(P1)-1]

#sp.savemat('normaldata1.mat', {'fftdata':P1})


#plt.plot(f, P1)
#plt.show()



#plt.plot(data[0:100])

#plt.show()

#fftdata = np.fft.fft(data)

#plt.plot(fftdata[0:100])

#plt.show()
