function [S, F, T] = runSTFT(signal, fs)
    % RUNSTFT Computes Short-Time Fourier Transform of signal.
    
    window = hamming(256);
    noverlap = 128;
    nfft = 512;
    [S, F, T] = spectrogram(signal, window, noverlap, nfft, fs);
end
