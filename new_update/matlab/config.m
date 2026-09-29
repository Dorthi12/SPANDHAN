function cfg = config()
%CONFIG  Project-wide configuration parameters for SPANDHAN.

    % ---- Audio preprocessing (MATLAB stage) -------------------
    cfg.targetFs       = 16000;  % Unified sampling rate (Hz)
    cfg.targetDuration = 2.0;    % seconds
    cfg.targetSamples  = 32000;  % = targetFs * targetDuration

    % ---- DSP analysis (post-ML stage) -------------------------
    cfg.fftSize     = 1024;      % Default FFT window size

    % ---- Image pipeline ----------------------------------------
    cfg.imageSize   = [128, 128];

    % ---- Dataset directories -----------------------------------
    cfg.audioRawDir       = '../datasets/audio/';
    cfg.audioProcessedDir = '../datasets/audio_processed/';
    cfg.modelsDir         = '../models/';

    % ---- Signal classes ----------------------------------------
    cfg.classes = ["impulse"; "sinusoidal"; "white_noise"; "step"; "chirp"];

end
