function processed = preprocessAudio(signal, fs)
    % PREPROCESSAUDIO Applies filtering and normalization to raw audio.
    % Inputs:
    %   signal    - Raw audio signal
    %   fs        - Sampling frequency
    % Outputs:
    %   processed - Preprocessed audio signal

    clean = removeNoise(signal, fs);
    processed = normalizeAudio(clean);
end
