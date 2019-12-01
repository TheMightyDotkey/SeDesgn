
Fs = 12000;
T = 1/Fs
L = 6000
t = (0:L-1)*T;
x = X119_DE_time(1:6000)
%plot(t,x)
Q = fft(x)
Y = fft(x)/L;
f = Fs/2*linspace(0,1,L/2+1);
k = 2*abs(Y(1:L/2+1))
plot(f, k)