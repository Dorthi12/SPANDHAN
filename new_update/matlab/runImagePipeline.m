function result = runImagePipeline(filepath, varargin)
% RUNIMAGEPIPELINE
% End-to-end SPANDHAN 2-D image analysis pipeline:
% Load -> Preprocess -> ML Classification -> Class-Aware Image DSP -> Feature Extraction -> Visualization
%
% USAGE:
%   result = runImagePipeline(filepath)
%   result = runImagePipeline(filepath, "Visualize", true)
%   result = runImagePipeline(filepath, "Visualize", false, "FilterType", "gaussian")
%
% INPUTS:
%   filepath - Absolute or relative path to the input image (.png, .jpg, .bmp, etc.)
%
% OPTIONAL PARAMETERS (Name-Value pairs):
%   "Visualize"             - true/false: whether to generate DSP result figures (default: true)
%   "FigureVisible"         - "on"/"off": whether figures are displayed on screen (default: "on")
%   "FFTAnalysis"           - true/false: run 2D FFT if selected by class (default: true)
%   "WaveletName"           - 2-D Wavelet family name (default: "db4")
%   "WaveletLevel"          - 2-D Wavelet decomposition level (default: 3)
%   "FilterType"            - Filter type ("gaussian", "average", "laplacian", "sobel", "prewitt", "custom")
%   "FilterKernelSize"      - Spatial filter kernel size (default: 5)
%   "FilterSigma"           - Gaussian filter sigma (default: 1)
%   "ConvolutionKernel"     - 2-D spatial convolution kernel matrix
%   "PSF"                   - Point Spread Function for image deconvolution
%   "DeconvolutionMethod"   - Method: "wiener", "regularized", "lucy", "blind" (default: "wiener")
%   "NSR"                   - Noise-to-Signal ratio for Wiener filter (default: 0.01)
%
% OUTPUT:
%   result - Unified struct containing:
%       .modality        - "image"
%       .input           - Struct with .file, .rawImage, .rawSize
%       .preprocessing   - Struct with .image (128x128 double [0,1]) and metadata
%       .ml              - Struct with .class, .class_id, .confidence, .probabilities
%       .dsp             - Unified image DSP result structure from analyzeImage()
%       .features        - Quantitative features extracted by calculateFeatures()
%       .figures         - Figure handles returned by plotDSPResults()

    %% ===============================================================
    % 1. INPUT VALIDATION & PARAMETER PARSING
    % ===============================================================

    if nargin < 1 || isempty(filepath)
        error("runImagePipeline requires an image file path.");
    end

    filepath = char(filepath);
    if ~exist(filepath, "file")
        error("Image file does not exist: %s", filepath);
    end

    visualize  = true;
    figVisible = "on";
    dspArgs    = {};

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
                % Forward all other parameters to analyzeImage
                dspArgs = [dspArgs, {varargin{k}, val}]; %#ok<AGROW>
        end
    end

    fprintf("\n------------------------------------------------------------\n");
    fprintf(" SPANDHAN IMAGE PIPELINE: %s\n", filepath);
    fprintf("------------------------------------------------------------\n");

    %% ===============================================================
    % 2. STEP 1: LOAD IMAGE
    % ===============================================================

    fprintf("[1/5] Loading raw image...\n");
    rawImage = loadImage(filepath);
    rawSize  = size(rawImage);
    fprintf("      Loaded image of size: %s (%s)\n", ...
        mat2str(rawSize), class(rawImage));

    %% ===============================================================
    % 3. STEP 2: PREPROCESS IMAGE
    % ===============================================================

    fprintf("[2/5] Standardizing image (grayscale, 128x128, normalized [0, 1])...\n");
    [processedImage, prepInfo] = preprocessImage(rawImage);
    fprintf("      Conditioned: %dx%d %s (Range: [%.3f, %.3f])\n", ...
        size(processedImage, 1), size(processedImage, 2), ...
        class(processedImage), min(processedImage(:)), max(processedImage(:)));

    %% ===============================================================
    % 4. STEP 3: ML CLASSIFICATION
    % ===============================================================

    fprintf("[3/5] Classifying image via Python ML classifier...\n");
    prediction = classifyImage(processedImage);
    fprintf("      Predicted Class: %s (ID: %d)\n", ...
        upper(prediction.class), prediction.class_id);
    fprintf("      Confidence     : %.2f%%\n", prediction.confidence * 100);

    %% ===============================================================
    % 5. STEP 4: CLASS-AWARE IMAGE DSP ANALYSIS
    % ===============================================================

    fprintf("[4/5] Running class-aware DSP pipeline for class '%s'...\n", prediction.class);
    dspResult = analyzeImage(processedImage, prediction.class, dspArgs{:});
    fprintf("      Completed DSP Modules: %s\n", ...
        strjoin(dspResult.completedAnalyses, ", "));
    if ~isempty(dspResult.failedAnalyses)
        fprintf("      Failed DSP Modules   : %s\n", ...
            strjoin(dspResult.failedAnalyses, ", "));
    end
    fprintf("      DSP Status           : %s\n", dspResult.status);

    %% ===============================================================
    % 6. STEP 5: FEATURE EXTRACTION
    % ===============================================================

    features = struct();
    try
        features = calculateFeatures(dspResult, "image");
    catch ME
        warning("Feature extraction skipped: %s", ME.message);
    end

    %% ===============================================================
    % 7. STEP 6: VISUALIZATION
    % ===============================================================

    figures = struct();
    if visualize
        fprintf("[5/5] Generating Image DSP visualizations...\n");
        try
            figures = plotDSPResults(dspResult, "image", "Visible", figVisible);
        catch ME
            warning("Visualization error: %s", ME.message);
        end
    else
        fprintf("[5/5] Visualization skipped (Visualize=false).\n");
    end

    %% ===============================================================
    % 8. STEP 7: PACKAGE UNIFIED RESULT
    % ===============================================================

    result = struct();
    result.modality               = "image";
    result.input                  = struct();
    result.input.file             = filepath;
    result.input.rawImage         = rawImage;
    result.input.rawSize          = rawSize;
    result.input.originalSize     = rawSize;

    result.preprocessing          = prepInfo;
    result.preprocessing.image    = processedImage;
    result.preprocessing.processedSize = size(processedImage);
    result.preprocessing.dataType = class(processedImage);

    result.ml                     = prediction;
    result.dsp                    = dspResult;
    result.features               = features;
    result.figures                = figures;

    fprintf("------------------------------------------------------------\n");
    fprintf(" SPANDHAN Image Pipeline Complete.\n");
    fprintf("------------------------------------------------------------\n\n");

end
