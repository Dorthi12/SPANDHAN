function normSignal = normalizeAudio(signal)
%NORMALIZEAUDIO  Peak-normalize audio signal to range [-1, 1].
%   Used by legacy preprocessAudio.m path.
%   The full preprocessing pipeline uses preprocessAudioFile.m directly.

    signal  = signal(:);
    maxVal  = max(abs(signal));

    if maxVal > 0
        normSignal = signal / maxVal;
    else
        normSignal = signal;
    end
end
