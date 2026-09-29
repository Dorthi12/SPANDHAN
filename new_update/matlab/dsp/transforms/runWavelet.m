function [c, l] = runWavelet(signal, waveletName)
    % RUNWAVELET Computes Continuous/Discrete Wavelet Transform.
    if nargin < 2
        waveletName = 'db4';
    end
    [c, l] = wavedec(signal, 3, waveletName);
end
