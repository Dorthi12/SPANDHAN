function [f, Y] = runFFT(signal, fs)
    % RUNFFT Computes Fast Fourier Transform of input signal.
    % Inputs:
    %   signal - Time-domain signal
    %   fs     - Sampling frequency
    % Outputs:
    %   f      - Frequency vector (Hz)
    %   Y      - Single-sided amplitude spectrum

    L = length(signal);
    NFFT = 2^nextpow2(L);
    Y_raw = fft(signal, NFFT);
    P2 = abs(Y_raw / L);
    Y = P2(1:NFFT/2+1);
    Y(2:end-1) = 2*Y(2:end-1);
    f = fs * (0:(NFFT/2)) / NFFT;
end
