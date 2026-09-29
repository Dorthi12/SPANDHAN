function cleanSignal = removeNoise(signal, fs)
    % REMOVENOISE Basic noise attenuation for audio signals using high-pass filtering.
    fc = 50; % Cutoff frequency 50Hz to remove DC / hum
    cleanSignal = runFIR(signal, fc, fs);
end
