function result = runAudioPipeline(filepath)
% runAudioPipeline
%
% Complete SPANDHAN audio processing pipeline.
%
% Pipeline:
%   1. Load audio
%   2. Preprocess audio in MATLAB
%   3. Classify preprocessed audio using Python ML
%   4. Run class-aware DSP analysis in MATLAB
%   5. Return a unified result structure
%
% Input:
%   filepath - Path to input audio file
%
% Output:
%   result - Structure containing:
%       result.input
%       result.preprocessing
%       result.ml
%       result.dsp
%
% Example:
%   result = runAudioPipeline("sample.wav");

    fprintf("\n");
    fprintf("============================================\n");
    fprintf("        SPANDHAN AUDIO PIPELINE\n");
    fprintf("============================================\n");

    %% ---------------------------------------------------------
    % 1. Validate input
    % ----------------------------------------------------------

    if nargin < 1 || isempty(filepath)
        error("runAudioPipeline:MissingInput", ...
            "An audio file path must be provided.");
    end

    filepath = char(filepath);

    if ~isfile(filepath)
        error("runAudioPipeline:FileNotFound", ...
            "Audio file not found: %s", filepath);
    end

    fprintf("\n[1/4] Loading audio...\n");
    fprintf("      File: %s\n", filepath);

    %% ---------------------------------------------------------
    % 2. Load audio
    % ----------------------------------------------------------

    [rawSignal, originalFs] = loadAudio(filepath);

    fprintf("      Original sampling rate: %d Hz\n", originalFs);
    fprintf("      Original samples: %d\n", length(rawSignal));

    %% ---------------------------------------------------------
    % 3. MATLAB preprocessing
    % ----------------------------------------------------------

    fprintf("\n[2/4] Preprocessing audio in MATLAB...\n");

    [processedSignal, prepInfo] = preprocessAudio( ...
        rawSignal, ...
        originalFs);

    % Make sure processed sampling rate is available.
    if isfield(prepInfo, "processedFs")
        processedFs = prepInfo.processedFs;
    else
        % Current SPANDHAN configuration uses 16 kHz.
        cfg = config();
        processedFs = cfg.targetFs;
        prepInfo.processedFs = processedFs;
    end

    fprintf("      Processed sampling rate: %d Hz\n", processedFs);
    fprintf("      Processed samples: %d\n", length(processedSignal));

    %% ---------------------------------------------------------
    % 4. Python ML classification
    % ----------------------------------------------------------

    fprintf("\n[3/4] Classifying audio using Python ML...\n");

    prediction = classifyAudio( ...
        processedSignal, ...
        processedFs);

    fprintf("      Predicted class: %s\n", ...
        char(prediction.class));

    fprintf("      Confidence: %.2f%%\n", ...
        prediction.confidence * 100);

    %% ---------------------------------------------------------
    % 5. MATLAB DSP analysis
    % ----------------------------------------------------------

    fprintf("\n[4/4] Running class-aware DSP analysis...\n");

    dspResult = analyzeAudio( ...
        processedSignal, ...
        processedFs, ...
        prediction.class);

    %% ---------------------------------------------------------
    % 6. Build unified result structure
    % ----------------------------------------------------------

    result = struct();

    % Input information
    result.input = struct();
    result.input.file = filepath;
    result.input.originalFs = originalFs;
    result.input.originalSamples = length(rawSignal);

    % Preprocessing information
    result.preprocessing = prepInfo;
    result.preprocessing.processedFs = processedFs;
    result.preprocessing.processedSamples = length(processedSignal);

    % ML result
    result.ml = prediction;

    % DSP result
    result.dsp = dspResult;

    % Store processed signal for downstream visualization
    result.signal = processedSignal;

    fprintf("\n============================================\n");
    fprintf("       AUDIO PIPELINE COMPLETED\n");
    fprintf("============================================\n");

end
