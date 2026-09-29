function normSignal = normalizeAudio(signal)
    % NORMALIZEAUDIO Normalizes audio signal amplitude to range [-1, 1].
    maxVal = max(abs(signal));
    if maxVal == 0
        normSignal = signal;
    else
        normSignal = signal / maxVal;
    end
end
