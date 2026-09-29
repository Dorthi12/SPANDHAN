%% ============================================================
%  SPANDHAN - AUDIO PREPROCESSING PIPELINE
%  File: preprocessAudioDataset.m
%
%  Purpose:
%  Master script. Processes all 5000 raw WAV files from
%  datasets/audio/ and outputs standardized WAV files to
%  datasets/audio_processed/ for the Python ML pipeline.
%
%  MATLAB responsibilities (this script):
%    1.  Load WAV
%    2.  Convert stereo → mono
%    3.  Remove DC offset
%    4.  Resample to TARGET_FS
%    5.  Remove NaN / Inf
%    6.  Amplitude normalization
%    7.  Length standardization (class-aware cropping)
%    8.  Final normalization
%    9.  Save processed WAV
%   10.  Generate metadata CSV
%   11.  QA validation
%   12.  QA plots
%
%  Python responsibilities (downstream):
%    - Feature extraction (RMS, ZCR, spectral centroid, etc.)
%    - ML training / evaluation / inference
%
%  Usage:
%    cd <project_root>
%    run matlab/preprocessing/preprocessAudioDataset.m
%
%  ============================================================

clc;
clear;
close all;

%% ------------------------------------------------------------
%  PATH SETUP
%  Add all MATLAB subfolders so helper functions are visible.
% ------------------------------------------------------------

scriptDir  = fileparts(mfilename('fullpath'));
matlabRoot = fileparts(scriptDir);          % .../matlab/
addpath(genpath(matlabRoot));

%% ============================================================
%  CONFIGURATION
% =============================================================

% ------ Source / output directories -------------------------
PROJECT_ROOT = fileparts(matlabRoot);       % .../new_update/
SOURCE_DIR   = fullfile(PROJECT_ROOT, 'datasets', 'audio');
OUTPUT_DIR   = fullfile(PROJECT_ROOT, 'datasets', 'audio_processed');
PLOTS_DIR    = fullfile(PROJECT_ROOT, 'datasets', 'audio_processed', 'qa_plots');

% ------ Target signal parameters ----------------------------
TARGET_FS       = 16000;   % Hz  — unified sampling rate for ML
TARGET_DURATION = 2.0;     % seconds
TARGET_SAMPLES  = TARGET_FS * TARGET_DURATION;   % 32 000

% ------ Classes to process ----------------------------------
CLASSES = ["impulse"; "sinusoidal"; "white_noise"; "step"; "chirp"];

% ------ QA plot control -------------------------------------
GENERATE_PLOTS   = true;   % set false to skip plots (faster)
PLOTS_PER_CLASS  = 3;      % how many example plots per class

% ------ Verbosity -------------------------------------------
VERBOSE = true;

%% ============================================================
%  CREATE OUTPUT DIRECTORY TREE
% =============================================================

if ~exist(OUTPUT_DIR, 'dir');  mkdir(OUTPUT_DIR);  end
if GENERATE_PLOTS && ~exist(PLOTS_DIR, 'dir');  mkdir(PLOTS_DIR);  end

for c = 1:numel(CLASSES)
    d = fullfile(OUTPUT_DIR, CLASSES(c));
    if ~exist(d, 'dir');  mkdir(d);  end
end

%% ============================================================
%  METADATA STORAGE
% =============================================================

% Pre-allocate metadata as cell arrays for speed, convert to
% table at the end.
metaFilename        = {};
metaClass           = {};
metaOriginalFs      = [];
metaProcessedFs     = [];
metaOriginalSamples = [];
metaProcessedSamples= [];
metaOriginalDur     = [];
metaProcessedDur    = [];
metaOriginalCh      = [];
metaOriginalPeak    = [];
metaProcessedPeak   = [];
metaOriginalMean    = [];
metaProcessedMean   = [];
metaWasResampled    = [];
metaWasPadded       = [];
metaWasCropped      = [];
metaSkipped         = [];

%% ============================================================
%  MAIN PROCESSING LOOP
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' SPANDHAN AUDIO PREPROCESSING PIPELINE\n');
fprintf('============================================================\n');
fprintf(' Source  : %s\n', SOURCE_DIR);
fprintf(' Output  : %s\n', OUTPUT_DIR);
fprintf(' Target  : %d Hz  |  %.1f s  |  %d samples\n', ...
        TARGET_FS, TARGET_DURATION, TARGET_SAMPLES);
fprintf('============================================================\n\n');

totalProcessed = 0;
totalSkipped   = 0;

for c = 1:numel(CLASSES)

    className  = CLASSES(c);
    inputDir   = fullfile(SOURCE_DIR,  className);
    outputDir  = fullfile(OUTPUT_DIR,  className);

    fprintf('------------------------------------------------------------\n');
    fprintf(' Class: %s\n', className);
    fprintf('------------------------------------------------------------\n');

    files = dir(fullfile(inputDir, '*.wav'));

    if isempty(files)
        fprintf('  [WARN] No WAV files found in: %s\n', inputDir);
        continue;
    end

    fprintf('  Files found : %d\n\n', numel(files));

    plotCount = 0;   % QA plots generated for this class

    for k = 1:numel(files)

        inputFile  = fullfile(inputDir,  files(k).name);
        outputFile = fullfile(outputDir, files(k).name);

        if VERBOSE
            fprintf('  [%4d/%4d] %s\n', k, numel(files), files(k).name);
        end

        %% --------------------------------------------------
        %  STEP 1: READ AUDIO
        % ---------------------------------------------------

        try
            [x, originalFs] = audioread(inputFile);
        catch ME
            fprintf('    [ERROR] Cannot read file: %s\n', ME.message);
            totalSkipped = totalSkipped + 1;

            % Still log the skipped entry
            metaFilename{end+1,1}         = files(k).name;
            metaClass{end+1,1}            = char(className);
            metaOriginalFs(end+1,1)       = NaN;
            metaProcessedFs(end+1,1)      = TARGET_FS;
            metaOriginalSamples(end+1,1)  = NaN;
            metaProcessedSamples(end+1,1) = NaN;
            metaOriginalDur(end+1,1)      = NaN;
            metaProcessedDur(end+1,1)     = NaN;
            metaOriginalCh(end+1,1)       = NaN;
            metaOriginalPeak(end+1,1)     = NaN;
            metaProcessedPeak(end+1,1)    = NaN;
            metaOriginalMean(end+1,1)     = NaN;
            metaProcessedMean(end+1,1)    = NaN;
            metaWasResampled(end+1,1)     = false;
            metaWasPadded(end+1,1)        = false;
            metaWasCropped(end+1,1)       = false;
            metaSkipped(end+1,1)          = true;
            continue;
        end

        %% --------------------------------------------------
        %  STEP 2-8: PREPROCESS
        % ---------------------------------------------------

        [xProcessed, info] = preprocessAudioFile( ...
            x, originalFs, TARGET_FS, TARGET_SAMPLES, className);

        %% --------------------------------------------------
        %  STEP 9: SAVE PROCESSED WAV
        % ---------------------------------------------------

        try
            audiowrite(outputFile, xProcessed, TARGET_FS);
        catch ME
            fprintf('    [ERROR] Cannot write file: %s\n', ME.message);
            totalSkipped = totalSkipped + 1;
            continue;
        end

        %% --------------------------------------------------
        %  ACCUMULATE METADATA
        % ---------------------------------------------------

        metaFilename{end+1,1}         = files(k).name;
        metaClass{end+1,1}            = char(className);
        metaOriginalFs(end+1,1)       = originalFs;
        metaProcessedFs(end+1,1)      = TARGET_FS;
        metaOriginalSamples(end+1,1)  = info.originalLength;
        metaProcessedSamples(end+1,1) = info.processedLength;
        metaOriginalDur(end+1,1)      = info.originalDuration;
        metaProcessedDur(end+1,1)     = info.processedDuration;
        metaOriginalCh(end+1,1)       = info.originalChannels;
        metaOriginalPeak(end+1,1)     = info.originalPeak;
        metaProcessedPeak(end+1,1)    = info.processedPeak;
        metaOriginalMean(end+1,1)     = info.originalMean;
        metaProcessedMean(end+1,1)    = info.processedMean;
        metaWasResampled(end+1,1)     = info.wasResampled;
        metaWasPadded(end+1,1)        = info.wasPadded;
        metaWasCropped(end+1,1)       = info.wasCropped;
        metaSkipped(end+1,1)          = false;

        totalProcessed = totalProcessed + 1;

        %% --------------------------------------------------
        %  QA PLOT (first PLOTS_PER_CLASS files per class)
        % ---------------------------------------------------

        if GENERATE_PLOTS && plotCount < PLOTS_PER_CLASS
            plotCount = plotCount + 1;
            visualizePreprocessing( ...
                x, xProcessed, originalFs, TARGET_FS, ...
                files(k).name, className, ...
                PLOTS_DIR, plotCount);
        end

    end % file loop

    fprintf('\n  Processed: %d | Skipped: %d\n\n', ...
        sum(~metaSkipped(end - numel(files) + 1 : end)), ...
        sum( metaSkipped(end - numel(files) + 1 : end)));

end % class loop

%% ============================================================
%  STEP 10: SAVE METADATA CSV
% =============================================================

metadata = table( ...
    metaFilename, ...
    metaClass, ...
    metaOriginalFs, ...
    metaProcessedFs, ...
    metaOriginalSamples, ...
    metaProcessedSamples, ...
    metaOriginalDur, ...
    metaProcessedDur, ...
    metaOriginalCh, ...
    metaOriginalPeak, ...
    metaProcessedPeak, ...
    metaOriginalMean, ...
    metaProcessedMean, ...
    metaWasResampled, ...
    metaWasPadded, ...
    metaWasCropped, ...
    metaSkipped, ...
    'VariableNames', { ...
        'filename', ...
        'class', ...
        'originalFs', ...
        'processedFs', ...
        'originalSamples', ...
        'processedSamples', ...
        'originalDuration', ...
        'processedDuration', ...
        'originalChannels', ...
        'originalPeak', ...
        'processedPeak', ...
        'originalMean', ...
        'processedMean', ...
        'wasResampled', ...
        'wasPadded', ...
        'wasCropped', ...
        'wasSkipped' ...
    });

metadataFile = fullfile(OUTPUT_DIR, 'audio_preprocessing_metadata.csv');
writetable(metadata, metadataFile);

%% ============================================================
%  STEP 11: QA VALIDATION SUMMARY
% =============================================================

validatePreprocessedAudio(metadata, OUTPUT_DIR, TARGET_FS, TARGET_SAMPLES);

%% ============================================================
%  FINAL SUMMARY
% =============================================================

fprintf('\n============================================================\n');
fprintf(' PREPROCESSING COMPLETE\n');
fprintf('============================================================\n');
fprintf(' Target sampling rate  : %d Hz\n',      TARGET_FS);
fprintf(' Target duration       : %.2f s\n',     TARGET_DURATION);
fprintf(' Target samples        : %d\n',         TARGET_SAMPLES);
fprintf('\n');
fprintf(' Total processed       : %d\n',  totalProcessed);
fprintf(' Total skipped         : %d\n',  totalSkipped);
fprintf('\n');
fprintf(' Metadata CSV          : %s\n', metadataFile);
fprintf(' Processed dataset     : %s\n', OUTPUT_DIR);
if GENERATE_PLOTS
    fprintf(' QA plots              : %s\n', PLOTS_DIR);
end
fprintf('============================================================\n\n');
fprintf(' Next step: Python ML Pipeline\n');
fprintf('   python/audio/audio_features.py  <-- reads audio_processed/\n');
fprintf('============================================================\n\n');
