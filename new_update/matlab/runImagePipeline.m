function result = runImagePipeline(filepath)
% runImagePipeline
%
% Complete SPANDHAN image processing pipeline.
%
% Pipeline:
%   1. Load image
%   2. Preprocess image in MATLAB
%   3. Classify preprocessed image using Python ML
%   4. Run class-aware image DSP analysis in MATLAB
%   5. Return a unified result structure
%
% Input:
%   filepath - Path to input image file
%
% Output:
%   result - Structure containing:
%       result.input
%       result.preprocessing
%       result.ml
%       result.dsp
%
% Example:
%   result = runImagePipeline("sample.png");

    fprintf("\n");
    fprintf("============================================\n");
    fprintf("        SPANDHAN IMAGE PIPELINE\n");
    fprintf("============================================\n");

    %% ---------------------------------------------------------
    % 1. Validate input
    % ----------------------------------------------------------

    if nargin < 1 || isempty(filepath)
        error("runImagePipeline:MissingInput", ...
            "An image file path must be provided.");
    end

    filepath = char(filepath);

    if ~isfile(filepath)
        error("runImagePipeline:FileNotFound", ...
            "Image file not found: %s", filepath);
    end

    fprintf("\n[1/4] Loading image...\n");
    fprintf("      File: %s\n", filepath);

    %% ---------------------------------------------------------
    % 2. Load image
    % ----------------------------------------------------------

    rawImage = loadImage(filepath);

    fprintf("      Original image size: ");

    originalSize = size(rawImage);

    fprintf("%d x %d", ...
        originalSize(1), ...
        originalSize(2));

    if ndims(rawImage) == 3
        fprintf(" x %d", originalSize(3));
    end

    fprintf("\n");

    %% ---------------------------------------------------------
    % 3. MATLAB preprocessing
    % ----------------------------------------------------------

    fprintf("\n[2/4] Preprocessing image in MATLAB...\n");

    processedImage = preprocessImage(rawImage);

    processedSize = size(processedImage);

    fprintf("      Processed image size: %d x %d\n", ...
        processedSize(1), ...
        processedSize(2));

    fprintf("      Data type: %s\n", ...
        class(processedImage));

    %% ---------------------------------------------------------
    % 4. Python ML classification
    % ----------------------------------------------------------

    fprintf("\n[3/4] Classifying image using Python ML...\n");

    prediction = classifyImage(processedImage);

    fprintf("      Predicted class: %s\n", ...
        char(prediction.class));

    fprintf("      Confidence: %.2f%%\n", ...
        prediction.confidence * 100);

    %% ---------------------------------------------------------
    % 5. MATLAB image DSP analysis
    % ----------------------------------------------------------

    fprintf("\n[4/4] Running class-aware image DSP analysis...\n");

    dspResult = analyzeImage( ...
        processedImage, ...
        prediction.class);

    %% ---------------------------------------------------------
    % 6. Build unified result structure
    % ----------------------------------------------------------

    result = struct();

    % Input information
    result.input = struct();
    result.input.file = filepath;
    result.input.originalSize = originalSize;

    % Preprocessing information
    result.preprocessing = struct();
    result.preprocessing.processedSize = processedSize;
    result.preprocessing.dataType = class(processedImage);

    % ML result
    result.ml = prediction;

    % DSP result
    result.dsp = dspResult;

    % Store processed image for visualization
    result.image = processedImage;

    fprintf("\n============================================\n");
    fprintf("       IMAGE PIPELINE COMPLETED\n");
    fprintf("============================================\n");

end
