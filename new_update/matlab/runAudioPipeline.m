function result = runAudioPipeline(filepath, varargin)
% RUNAUDIOPIPELINE
% End-to-end SPANDHAN audio analysis pipeline:
% Load -> Preprocess -> ML Classification -> Class-Aware DSP -> Feature Extraction -> Visualization
%
% USAGE:
%   result = runAudioPipeline(filepath)
%   result = runAudioPipeline(filepath, "Visualize", true)
%   result = runAudioPipeline(filepath, "Visualize", false, "ImpulseResponse", ir)
%
% INPUTS:
%   filepath - Absolute or relative path to the input audio file (.wav, .mp3, etc.)
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   "Visualize"       - true/false: whether to generate DSP result figures (default: true)
%   "FigureVisible"   - "on"/"off": whether figures are displayed on screen (default: "on")
%   "ImpulseResponse" - System impulse response vector for convolution/deconvolution
%   "FFTSize"         - FFT size (default: 4096)
%   "STFTWindow"      - STFT window length (default: 1024)
%   "STFTOverlap"     - STFT overlap (default: 512)
%   "STFTNFFT"        - STFT FFT points (default: 2048)
%   "Wavelet"         - Wavelet family (default: "db4")
%   "WaveletLevel"    - Wavelet decomposition level (default: 5)
%   "FIROrder"        - FIR filter order (default: 100)
%   "FIRCutoff"       - FIR cutoff frequency in Hz (default: 3000)
%   "IIRPassband"     - IIR passband frequency in Hz (default: 3000)
%   "IIRStopband"     - IIR stopband frequency in Hz (default: 4000)
%
% OUTPUT:
%   result - Unified struct containing:
%       .modality        - "audio"
%       .input           - Struct with .file, .originalFs, .originalSamples, .rawSignal
%       .preprocessing   - Struct with .processedSignal, .processedFs, .processedSamples, etc.
%       .ml              - Struct with .class, .class_id, .confidence, .probabilities
%       .dsp             - Unified DSP result structure from analyzeAudio()
%       .features        - Quantitative features extracted by calculateFeatures()
%       .figures         - Figure handles returned by plotDSPResults()
%       .signal          - Preprocessed signal vector

    fprintf("\n");
    fprintf("============================================\n");
    fprintf("        SPANDHAN AUDIO PIPELINE\n");
    fprintf("============================================\n");

    %% ---------------------------------------------------------
    % 1. Validate input & parse parameters
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

    visualize = true;
    figVisible = "on";
    dspArgs = {};

    if mod(length(varargin), 2) ~= 0
        error("Optional arguments must be supplied as name-value pairs.");
    end

    for k = 1:2:length(varargin)
        param = lower(string(varargin{k}));
        val   = varargin{k+1};

        switch param
            case "visualize"
                visualize = logical(val);
            case "figurevisible"
                figVisible = string(val);
            otherwise
                % Forward all other parameters to analyzeAudio
                dspArgs = [dspArgs, {varargin{k}, val}]; %#ok<AGROW>
        end
    end

    fprintf("\n[1/5] Loading audio...\n");
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

    fprintf("\n[2/5] Preprocessing audio in MATLAB...\n");

    [processedSignal, prepInfo] = preprocessAudio( ...
        rawSignal, ...
        originalFs);

    % Make sure processed sampling rate is available.
    if isfield(prepInfo, "processedFs")
        processedFs = prepInfo.processedFs;
    else
        cfg = config();
        processedFs = cfg.targetFs;
        prepInfo.processedFs = processedFs;
    end

    fprintf("      Processed sampling rate: %d Hz\n", processedFs);
    fprintf("      Processed samples: %d\n", length(processedSignal));

    %% ---------------------------------------------------------
    % 4. Python ML classification
    % ----------------------------------------------------------

    fprintf("\n[3/5] Classifying audio using Python ML...\n");

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

    fprintf("\n[4/5] Running class-aware DSP analysis...\n");

    dspResult = analyzeAudio( ...
        processedSignal, ...
        processedFs, ...
        prediction.class, ...
        dspArgs{:});

    fprintf("      Completed DSP Modules: %s\n", ...
        strjoin(dspResult.completedAnalyses, ", "));
    if ~isempty(dspResult.failedAnalyses)
        fprintf("      Failed DSP Modules   : %s\n", ...
            strjoin(dspResult.failedAnalyses, ", "));
    end
    fprintf("      DSP Status           : %s\n", dspResult.status);

    %% ---------------------------------------------------------
    % 6. Feature Extraction
    % ----------------------------------------------------------

    features = struct();
    try
        features = calculateFeatures(dspResult, "audio");
    catch ME
        warning("Feature extraction skipped: %s", ME.message);
    end

    %% ---------------------------------------------------------
    % 7. Visualization
    % ----------------------------------------------------------

    figures = struct();
    if visualize
        fprintf("\n[5/5] Generating DSP visualizations...\n");
        try
            figures = plotDSPResults(dspResult, "audio", "Visible", figVisible);
        catch ME
            warning("Visualization error: %s", ME.message);
        end
    else
        fprintf("\n[5/5] Visualization skipped (Visualize=false).\n");
    end

    %% ---------------------------------------------------------
    % 8. Build unified result structure
    % ----------------------------------------------------------

    result = struct();
    result.modality = "audio";

    % Input information
    result.input = struct();
    result.input.file = filepath;
    result.input.originalFs = originalFs;
    result.input.originalSamples = length(rawSignal);
    result.input.rawSignal = rawSignal;
    result.input.rawFs = originalFs;

    % Preprocessing information
    result.preprocessing = prepInfo;
    result.preprocessing.processedFs = processedFs;
    result.preprocessing.processedSamples = length(processedSignal);
    result.preprocessing.signal = processedSignal;
    result.preprocessing.processedSignal = processedSignal;

    % ML result
    result.ml = prediction;

    % DSP result
    result.dsp = dspResult;

    % Features & Figures
    result.features = features;
    result.figures = figures;

    % Store processed signal for downstream use
    result.signal = processedSignal;

    fprintf("\n============================================\n");
    fprintf("       AUDIO PIPELINE COMPLETED\n");
    fprintf("============================================\n");

end
