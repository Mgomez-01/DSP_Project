%% Problem 10.1 a
h_w = [ones(1,5) zeros(1,15) ones(1,5)]
h_n = fft(h_w)

figure()
plot(real(h_n))
hold on
plot(imag(h_n))

h_n = [h_n zeros(1,100)]
h_w = fft(h_n)
figure()
plot(abs(h_w))
figure()
plot(angle(h_w))

%% trial 2
close all
N = 25;
pos = zeros(1,N);
h_n = zeros(1,N);

mid = 12;

for n = 1:N
    pos(n) = (n-1)/N*pi*2;
    if (n-1-mid)==0
        h_n(n) = 1/6;
    else
        h_n(n) = sin(pi/6*(n-1-mid))/(pi*(n-1-mid));
    end
end

figure()
stem(h_n)
h_w = fft(h_n);
figure()
% plot(abs(h_w))
plot(pos, 10.*log10(abs(h_w)))
figure()
stem(pos, angle(h_w))
