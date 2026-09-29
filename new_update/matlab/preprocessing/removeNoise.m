function cleanSignal = removeNoise(signal, fs)
    % REMOVENOISE Basic noise attenuation for audio signals using high-pass filtering.
    fc = 50; % Cutoff frequency 50Hz to remove DC / hum
    firResult = runFIR(signal, fs, 100, fc, "high", "hamming");
    cleanSignal = firResult.output;
end
