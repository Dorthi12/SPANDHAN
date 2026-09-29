% TEST_CLASSIFY_IMPULSE - Test the full MATLAB -> Python ML classification flow
% Loads impulse_0001.wav, runs preprocessAudio(), then calls classifyAudio().

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(genpath(scriptDir));

% 1. Load Audio
inputFile = fullfile(projectRoot, 'datasets', 'audio', 'impulse', 'impulse_0001.wav');
fprintf('Loading audio: %s\n', inputFile);
[rawSignal, fs] = audioread(inputFile);

% 2. MATLAB Preprocessing (DC removal, resample 16kHz, length 32000, normalize)
[processedSignal, info] = preprocessAudio(rawSignal, fs, 'impulse');
fprintf('Preprocessing complete: length=%d, fs=%d Hz\n', length(processedSignal), info.processedFs);

% 3. ML Classification via Python Bridge
prediction = classifyAudio(processedSignal, info.processedFs);

% 4. Display standardized output
fprintf('\n======================================\n');
fprintf('SPANDHAN MATLAB Audio Classification\n');
fprintf('======================================\n');
fprintf('Detected Class : %s\n', prediction.class);
fprintf('Class ID       : %d\n', prediction.class_id);
fprintf('Confidence     : %.2f%%\n', prediction.confidence * 100);
fprintf('\nProbabilities:\n');
fields = fieldnames(prediction.probabilities);
for i = 1:numel(fields)
    fName = fields{i};
    fProb = prediction.probabilities.(fName);
    fprintf('  %-12s: %.4f\n', fName, fProb);
end
fprintf('======================================\n');
