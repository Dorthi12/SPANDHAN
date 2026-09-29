function [processed, info] = preprocessAudio(signal, fs, className)
%PREPROCESSAUDIO  SPANDHAN single-file audio conditioner (convenience wrapper).
%
%   Wraps preprocessAudioFile with standard SPANDHAN defaults:
%     TARGET_FS       = 16000 Hz
%     TARGET_SAMPLES  = 32000  (2 seconds)
%
%   For batch preprocessing of the full dataset, use:
%     preprocessAudioDataset.m
%
%   Inputs
%   ------
%   signal    : Raw audio signal matrix [samples × channels]
%   fs        : Input sampling frequency (Hz)
%   className : (optional) Signal class for smart crop/pad
%               "impulse" | "sinusoidal" | "white_noise" | "step" | "chirp"
%               Default: "sinusoidal" (centre crop/pad behaviour)
%
%   Outputs
%   -------
%   processed : Conditioned audio column vector [32000 × 1]
%   info      : Preprocessing metadata struct

    if nargin < 3
        className = "sinusoidal";   % safe default: centre crop/pad
    end

    TARGET_FS      = 16000;
    TARGET_SAMPLES = 32000;

    [processed, info] = preprocessAudioFile( ...
        signal, fs, TARGET_FS, TARGET_SAMPLES, className);

end
