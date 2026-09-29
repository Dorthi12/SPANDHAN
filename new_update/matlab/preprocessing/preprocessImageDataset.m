%% ============================================================
%  SPANDHAN - IMAGE PREPROCESSING PIPELINE
%  File: preprocessImageDataset.m
%
%  Purpose:
%  Master script. Processes all 5000 raw PNG files from
%  datasets/image/ and outputs standardized PNG files to
%  datasets/image_processed/ for the Python ML pipeline.
%
%  MATLAB responsibilities (this script):
%    1.  Load PNG
%    2.  Validate file is readable
%    3.  Convert to grayscale (if RGB/RGBA)
%    4.  Convert to double [0,1]  via  double(I)/255
%    5.  Handle NaN / Inf pixels
%    6.  Clamp to [0,1]
%    7.  Resize to 128x128 if necessary (bicubic, aspect-preserving pad)
%    8.  Final clamp and sanity check
%    9.  Save processed PNG as uint8
%   10.  Generate metadata CSV
%   11.  QA validation
%   12.  QA plots
%
%  What this pipeline does NOT do (intentional):
%    - Gaussian / median / bilateral denoising
%    - Sharpening or edge enhancement
%    - Histogram equalization or CLAHE
%    - Adaptive thresholding
%    - Sobel / Canny / Laplacian preprocessing
%    - FFT, wavelet, or any frequency-domain modification
%
%  Rationale: the spatial-frequency content IS the class signal.
%  Modifying it before ML destroys discriminative information.
%
%  Python responsibilities (downstream):
%    - float32 rescale  (pixel / 255)
%    - CNN feature extraction
%    - ML training / evaluation / inference
%
%  Usage:
%    cd <project_root>
%    run matlab/preprocessing/preprocessImageDataset.m
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
SOURCE_DIR   = fullfile(PROJECT_ROOT, 'datasets', 'image');
OUTPUT_DIR   = fullfile(PROJECT_ROOT, 'datasets', 'image_processed');
PLOTS_DIR    = fullfile(PROJECT_ROOT, 'datasets', 'image_processed', 'qa_plots');

% ------ Target image parameters -----------------------------
TARGET_HEIGHT = 128;
TARGET_WIDTH  = 128;

% ------ Classes to process ----------------------------------
CLASSES = ["impulse"; "sinusoidal"; "white_noise"; "step"; "chirp"];

% ------ QA plot control -------------------------------------
GENERATE_PLOTS  = true;   % set false to skip (faster)
PLOTS_PER_CLASS = 3;      % example plots saved per class

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
%  Pre-allocate as cell/numeric arrays; convert to table at end.
% =============================================================

metaFilename             = {};
metaClass                = {};
metaOriginalHeight       = [];
metaOriginalWidth        = [];
metaOriginalChannels     = [];
metaOriginalDatatype     = {};
metaOriginalMin          = [];
metaOriginalMax          = [];
metaProcessedHeight      = [];
metaProcessedWidth       = [];
metaProcessedDatatype    = {};
metaProcessedMin         = [];
metaProcessedMax         = [];
metaProcessedMean        = [];
metaProcessedStd         = [];
metaWasGrayscaleConverted = [];
metaWasPadded            = [];
metaWasResized           = [];
metaInvalidPixels        = [];
metaSkipped              = [];

%% ============================================================
%  MAIN PROCESSING LOOP
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' SPANDHAN IMAGE PREPROCESSING PIPELINE\n');
fprintf('============================================================\n');
fprintf(' Source  : %s\n', SOURCE_DIR);
fprintf(' Output  : %s\n', OUTPUT_DIR);
fprintf(' Target  : %d x %d  |  1 channel  |  float [0,1]\n', ...
        TARGET_WIDTH, TARGET_HEIGHT);
fprintf('============================================================\n\n');

totalProcessed = 0;
totalSkipped   = 0;

for c = 1:numel(CLASSES)

    className = CLASSES(c);
    inputDir  = fullfile(SOURCE_DIR,  className);
    outputDir = fullfile(OUTPUT_DIR,  className);

    fprintf('------------------------------------------------------------\n');
    fprintf(' Class: %s\n', className);
    fprintf('------------------------------------------------------------\n');

    files = dir(fullfile(inputDir, '*.png'));

    if isempty(files)
        fprintf('  [WARN] No PNG files found in: %s\n', inputDir);
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
        %  STEP 1: READ IMAGE
        % ---------------------------------------------------

        try
            I = imread(inputFile);
        catch ME
            fprintf('    [ERROR] Cannot read file: %s\n', ME.message);
            totalSkipped = totalSkipped + 1;

            % Log the skipped entry
            metaFilename{end+1,1}              = files(k).name;
            metaClass{end+1,1}                 = char(className);
            metaOriginalHeight(end+1,1)        = NaN;
            metaOriginalWidth(end+1,1)         = NaN;
            metaOriginalChannels(end+1,1)      = NaN;
            metaOriginalDatatype{end+1,1}      = 'unknown';
            metaOriginalMin(end+1,1)           = NaN;
            metaOriginalMax(end+1,1)           = NaN;
            metaProcessedHeight(end+1,1)       = NaN;
            metaProcessedWidth(end+1,1)        = NaN;
            metaProcessedDatatype{end+1,1}     = 'unknown';
            metaProcessedMin(end+1,1)          = NaN;
            metaProcessedMax(end+1,1)          = NaN;
            metaProcessedMean(end+1,1)         = NaN;
            metaProcessedStd(end+1,1)          = NaN;
            metaWasGrayscaleConverted(end+1,1) = false;
            metaWasPadded(end+1,1)             = false;
            metaWasResized(end+1,1)            = false;
            metaInvalidPixels(end+1,1)         = NaN;
            metaSkipped(end+1,1)               = true;
            continue;
        end

        %% --------------------------------------------------
        %  STEP 2-8: PREPROCESS
        % ---------------------------------------------------

        [IProcessed, info] = preprocessImageFile( ...
            I, TARGET_HEIGHT, TARGET_WIDTH);

        %% --------------------------------------------------
        %  STEP 9: SAVE PROCESSED PNG
        %  Store as uint8 to keep PNG file sizes manageable.
        %  Python will rescale back to float32 [0,1] via /255.
        % ---------------------------------------------------

        try
            imwrite(uint8(round(IProcessed * 255)), outputFile);
        catch ME
            fprintf('    [ERROR] Cannot write file: %s\n', ME.message);
            totalSkipped = totalSkipped + 1;
            continue;
        end

        %% --------------------------------------------------
        %  ACCUMULATE METADATA
        % ---------------------------------------------------

        metaFilename{end+1,1}              = files(k).name;
        metaClass{end+1,1}                 = char(className);
        metaOriginalHeight(end+1,1)        = info.originalHeight;
        metaOriginalWidth(end+1,1)         = info.originalWidth;
        metaOriginalChannels(end+1,1)      = info.originalChannels;
        metaOriginalDatatype{end+1,1}      = info.originalDatatype;
        metaOriginalMin(end+1,1)           = info.originalMin;
        metaOriginalMax(end+1,1)           = info.originalMax;
        metaProcessedHeight(end+1,1)       = info.processedHeight;
        metaProcessedWidth(end+1,1)        = info.processedWidth;
        metaProcessedDatatype{end+1,1}     = info.processedDatatype;
        metaProcessedMin(end+1,1)          = info.processedMin;
        metaProcessedMax(end+1,1)          = info.processedMax;
        metaProcessedMean(end+1,1)         = info.processedMean;
        metaProcessedStd(end+1,1)          = info.processedStd;
        metaWasGrayscaleConverted(end+1,1) = info.wasGrayscaleConverted;
        metaWasPadded(end+1,1)             = info.wasPadded;
        metaWasResized(end+1,1)            = info.wasResized;
        metaInvalidPixels(end+1,1)         = info.invalidPixels;
        metaSkipped(end+1,1)               = false;

        totalProcessed = totalProcessed + 1;

        %% --------------------------------------------------
        %  QA PLOT (first PLOTS_PER_CLASS files per class)
        % ---------------------------------------------------

        if GENERATE_PLOTS && plotCount < PLOTS_PER_CLASS
            plotCount = plotCount + 1;
            visualizeImagePreprocessing( ...
                I, IProcessed, ...
                files(k).name, className, ...
                PLOTS_DIR, plotCount);
        end

    end  % file loop

    fprintf('\n  Processed: %d | Skipped: %d\n\n', ...
        sum(~metaSkipped(end - numel(files) + 1 : end)), ...
        sum( metaSkipped(end - numel(files) + 1 : end)));

end  % class loop

%% ============================================================
%  STEP 10: SAVE METADATA CSV
% =============================================================

metadata = table( ...
    metaFilename, ...
    metaClass, ...
    metaOriginalHeight, ...
    metaOriginalWidth, ...
    metaOriginalChannels, ...
    metaOriginalDatatype, ...
    metaOriginalMin, ...
    metaOriginalMax, ...
    metaProcessedHeight, ...
    metaProcessedWidth, ...
    metaProcessedDatatype, ...
    metaProcessedMin, ...
    metaProcessedMax, ...
    metaProcessedMean, ...
    metaProcessedStd, ...
    metaWasGrayscaleConverted, ...
    metaWasPadded, ...
    metaWasResized, ...
    metaInvalidPixels, ...
    metaSkipped, ...
    'VariableNames', { ...
        'filename', ...
        'class', ...
        'originalHeight', ...
        'originalWidth', ...
        'originalChannels', ...
        'originalDatatype', ...
        'originalMin', ...
        'originalMax', ...
        'processedHeight', ...
        'processedWidth', ...
        'processedDatatype', ...
        'processedMin', ...
        'processedMax', ...
        'processedMean', ...
        'processedStd', ...
        'wasGrayscaleConverted', ...
        'wasPadded', ...
        'wasResized', ...
        'invalidPixels', ...
        'wasSkipped' ...
    });

metadataFile = fullfile(OUTPUT_DIR, 'image_preprocessing_metadata.csv');
writetable(metadata, metadataFile);

%% ============================================================
%  STEP 11: QA VALIDATION SUMMARY
% =============================================================

validatePreprocessedImages(metadata, OUTPUT_DIR, TARGET_HEIGHT, TARGET_WIDTH);

%% ============================================================
%  FINAL SUMMARY
% =============================================================

fprintf('\n============================================================\n');
fprintf(' PREPROCESSING COMPLETE\n');
fprintf('============================================================\n');
fprintf(' Target size           : %d x %d (grayscale)\n', TARGET_WIDTH, TARGET_HEIGHT);
fprintf(' Target datatype       : uint8 (stored)  /  float32 (ML)\n');
fprintf(' Target range          : [0, 255] stored  /  [0.0, 1.0] ML\n');
fprintf('\n');
fprintf(' Total processed       : %d\n', totalProcessed);
fprintf(' Total skipped         : %d\n', totalSkipped);
fprintf('\n');
fprintf(' Metadata CSV          : %s\n', metadataFile);
fprintf(' Processed dataset     : %s\n', OUTPUT_DIR);
if GENERATE_PLOTS
    fprintf(' QA plots              : %s\n', PLOTS_DIR);
end
fprintf('============================================================\n\n');
fprintf(' Next step: Python ML Pipeline\n');
fprintf('   python/image/image_classifier.py  <-- reads image_processed/\n');
fprintf('============================================================\n\n');
