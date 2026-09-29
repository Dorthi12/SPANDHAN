function cfg = config()
    % CONFIG Returns project-wide configuration parameters for SPANDHAN.
    
    cfg.sampleRate = 44100; % Default audio sampling rate (Hz)
    cfg.fftSize = 1024;     % Default FFT window size
    cfg.imageSize = [128, 128]; % Default image target dimension
    cfg.dataInputDir = '../data/input/';
    cfg.dataOutputDir = '../data/output/';
    cfg.modelsDir = '../models/';
end
