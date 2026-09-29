function [signal, fs] = loadAudio(filepath)
    % LOADAUDIO Loads audio signal from a given file path.
    % Inputs:
    %   filepath - Path to audio file (.wav, .mp3, etc.)
    % Outputs:
    %   signal   - Audio samples vector
    %   fs       - Sampling frequency (Hz)

    if ~exist(filepath, 'file')
        error('File not found: %s', filepath);
    end
    [signal, fs] = audioread(filepath);
end
