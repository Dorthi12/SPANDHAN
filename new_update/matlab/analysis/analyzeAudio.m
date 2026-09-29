function res = analyzeAudio(audioPath)
    % ANALYZEAUDIO Full analysis pipeline for an audio file.
    [signal, fs] = loadAudio(audioPath);
    processed = preprocessAudio(signal, fs);
    res = selectDSPAnalysis(processed, fs, 'fft');
end
