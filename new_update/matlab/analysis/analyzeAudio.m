function res = analyzeAudio(audioPath, signalClass, options)
%ANALYZEAUDIO Full SPANDHAN audio analysis pipeline.
%
%   res = analyzeAudio(audioPath)
%   res = analyzeAudio(audioPath, signalClass)
%   res = analyzeAudio(audioPath, signalClass, options)
%
%   Loads an audio file, applies preprocessing, then runs the
%   class-specific DSP pipeline via selectDSPAnalysis.
%
%   Inputs
%   ------
%   audioPath   : Path to audio file (.wav, .mp3, etc.)
%   signalClass : (optional) Signal class string — one of:
%                 "impulse" | "sinusoidal" | "white_noise" | "step" | "chirp"
%                 Default: "sinusoidal"
%   options     : (optional) Parameter override struct (see selectDSPAnalysis)
%
%   Output
%   ------
%   res : struct from selectDSPAnalysis, with additional fields:
%       .audioPath   : Source file path
%       .Fs          : Sampling frequency (Hz)
%       .rawSignal   : Signal after loading, before preprocessing
%       .signal      : Preprocessed signal passed into DSP chain

    narginchk(1, 3);

    if nargin < 2 || isempty(signalClass)
        signalClass = "sinusoidal";
    end

    if nargin < 3
        options = struct();
    end

    % Load audio
    [rawSignal, Fs] = loadAudio(audioPath);

    % Preprocessing (noise removal + normalization)
    signal = preprocessAudio(rawSignal, Fs);

    % Route through class-specific DSP pipeline
    res = selectDSPAnalysis(signal, Fs, signalClass, options);

    % Attach audio metadata
    res.audioPath  = audioPath;
    res.Fs         = Fs;
    res.rawSignal  = rawSignal;
    res.signal     = signal;

end
