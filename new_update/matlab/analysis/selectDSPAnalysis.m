function res = selectDSPAnalysis(signal, fs, dspType)
    % SELECTDSPANALYSIS Routes signal to requested DSP algorithm.
    switch lower(dspType)
        case 'fft'
            [res.f, res.Y] = runFFT(signal, fs);
        case 'stft'
            [res.S, res.F, res.T] = runSTFT(signal, fs);
        case 'wavelet'
            [res.c, res.l] = runWavelet(signal, 'db4');
        otherwise
            error('Unknown DSP Analysis type: %s', dspType);
    end
end
