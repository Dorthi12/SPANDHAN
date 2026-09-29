function cleanSignal = removeNoise(signal, fs)
%REMOVENOISE  Signal conditioning for SPANDHAN audio preprocessing.
%
%   IMPORTANT: This function performs ONLY DC offset removal.
%   Aggressive frequency-domain filtering (FIR high-pass, etc.)
%   is intentionally NOT applied here because it would destroy
%   the class-defining characteristics of:
%     - impulse    : shape distortion
%     - chirp      : frequency trajectory alteration
%     - step       : transition smoothing
%     - white_noise: broadband destruction
%     - sinusoidal : amplitude/frequency modification
%
%   Full standardization pipeline: use preprocessAudioFile.m
%
%   Inputs
%   ------
%   signal  : Audio column vector
%   fs      : Sampling frequency (Hz) [kept for API compatibility]
%
%   Output
%   ------
%   cleanSignal : DC-removed signal

    signal = signal(:);

    % Remove NaN / Inf defensively
    signal(~isfinite(signal)) = 0;

    % Remove DC offset (mean subtraction only)
    cleanSignal = signal - mean(signal);

end
