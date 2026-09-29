function result = runImageFilter(imageInput, filterType, varargin)
% RUNIMAGEFILTER
% General-purpose 2-D image filtering.
%
% INPUTS:
%   imageInput - Grayscale or RGB image
%   filterType - "gaussian", "average", "median", "sobel",
%                "prewitt", "laplacian", "log", "motion", "custom"
%
% OPTIONAL NAME-VALUE PAIRS:
%   "KernelSize"   - Filter size, default 3
%   "Sigma"        - Gaussian/LoG sigma, default 1
%   "MotionLength" - Motion blur length, default 9
%   "MotionAngle"  - Motion blur angle, default 0
%   "Kernel"       - Custom filter kernel
%   "Boundary"     - "symmetric", "replicate", "circular", "zero"
%   "Operation"    - "convolution" or "correlation"
%
% OUTPUT:
%   result - Structure containing filtered image, kernel,
%            response metrics and configuration.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 2
        error("runImageFilter requires imageInput and filterType.");
    end

    if isempty(imageInput)
        error("Input image cannot be empty.");
    end

    if ~isnumeric(imageInput) && ~islogical(imageInput)
        error("Input must be a numeric or logical image.");
    end

    if ndims(imageInput) ~= 2 && ndims(imageInput) ~= 3
        error("Input must be a grayscale or RGB image.");
    end

    if ndims(imageInput) == 3 && size(imageInput,3) ~= 3
        error("RGB input must have exactly 3 channels.");
    end

    %% ---------------------------------------------------------------
    % 2. DEFAULT PARAMETERS
    % ---------------------------------------------------------------

    kernelSize   = 3;
    sigma        = 1;
    motionLength = 9;
    motionAngle  = 0;
    customKernel = [];
    boundary     = "symmetric";
    operation    = "convolution";

    %% ---------------------------------------------------------------
    % 3. PARSE NAME-VALUE PARAMETERS
    % ---------------------------------------------------------------

    if mod(length(varargin), 2) ~= 0
        error("Optional arguments must be supplied as name-value pairs.");
    end

    for k = 1:2:length(varargin)

        parameter = lower(string(varargin{k}));
        value     = varargin{k+1};

        switch parameter
            case "kernelsize"
                kernelSize = value;
            case "sigma"
                sigma = value;
            case "motionlength"
                motionLength = value;
            case "motionangle"
                motionAngle = value;
            case "kernel"
                customKernel = value;
            case "boundary"
                boundary = lower(string(value));
            case "operation"
                operation = lower(string(value));
            otherwise
                error("Unknown parameter: %s", parameter);
        end

    end

    %% ---------------------------------------------------------------
    % 4. VALIDATE PARAMETERS
    % ---------------------------------------------------------------

    filterType = lower(string(filterType));

    validTypes = [ ...
        "gaussian", ...
        "average", ...
        "median", ...
        "sobel", ...
        "prewitt", ...
        "laplacian", ...
        "log", ...
        "motion", ...
        "custom" ...
    ];

    if ~any(filterType == validTypes)
        error("Unsupported filter type: %s", filterType);
    end

    validBoundaries = [ ...
        "symmetric", ...
        "replicate", ...
        "circular", ...
        "zero" ...
    ];

    if ~any(boundary == validBoundaries)
        error("Unsupported boundary mode: %s", boundary);
    end

    if ~any(operation == ["convolution", "correlation"])
        error("Operation must be convolution or correlation.");
    end

    %% ---------------------------------------------------------------
    % 5. CONVERT IMAGE TO DOUBLE
    % ---------------------------------------------------------------

    originalImage = imageInput;
    imageDouble   = im2double(imageInput);

    %% ---------------------------------------------------------------
    % 6. CREATE FILTER KERNEL
    % ---------------------------------------------------------------

    switch filterType

        case "gaussian"

            if ~isscalar(kernelSize) || kernelSize < 1
                error("KernelSize must be positive.");
            end

            kernelSize = round(kernelSize);

            if mod(kernelSize, 2) == 0
                kernelSize = kernelSize + 1;
            end

            if sigma <= 0
                error("Sigma must be positive.");
            end

            % Explicit Gaussian kernel.
            radius = floor(kernelSize / 2);
            [X, Y] = meshgrid(-radius:radius, -radius:radius);
            kernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
            kernel = kernel / sum(kernel(:));

        case "average"

            kernelSize = round(kernelSize);

            if mod(kernelSize, 2) == 0
                kernelSize = kernelSize + 1;
            end

            kernel = ones(kernelSize) / kernelSize^2;

        case "median"

            kernel = [];

        case "sobel"

            kernel = fspecial("sobel");

        case "prewitt"

            kernel = fspecial("prewitt");

        case "laplacian"

            kernel = fspecial("laplacian");

        case "log"

            kernelSize = round(kernelSize);

            if mod(kernelSize, 2) == 0
                kernelSize = kernelSize + 1;
            end

            kernel = fspecial("log", kernelSize, sigma);

        case "motion"

            kernel = fspecial("motion", motionLength, motionAngle);

        case "custom"

            if isempty(customKernel)
                error("Custom filter requires the Kernel parameter.");
            end

            if ~ismatrix(customKernel)
                error("Custom kernel must be a 2-D matrix.");
            end

            kernel = double(customKernel);

    end

    %% ---------------------------------------------------------------
    % 7. APPLY FILTER
    % ---------------------------------------------------------------

    tic;

    if filterType == "median"

        if ndims(imageDouble) == 2

            filteredImage = medfilt2( ...
                imageDouble, ...
                [kernelSize kernelSize], ...
                "symmetric");

        else

            filteredImage = zeros(size(imageDouble));

            for channel = 1:3
                filteredImage(:,:,channel) = medfilt2( ...
                    imageDouble(:,:,channel), ...
                    [kernelSize kernelSize], ...
                    "symmetric");
            end

        end

    else

        if boundary == "zero"
            boundaryArgument = 0;
        else
            boundaryArgument = char(boundary);
        end

        if operation == "convolution"
            operationArgument = "conv";
        else
            operationArgument = "corr";
        end

        filteredImage = imfilter( ...
            imageDouble, ...
            kernel, ...
            boundaryArgument, ...
            operationArgument, ...
            "same");

    end

    computationTime = toc;

    %% ---------------------------------------------------------------
    % 8. CALCULATE BASIC METRICS
    % ---------------------------------------------------------------

    inputMean  = mean(imageDouble(:));
    outputMean = mean(filteredImage(:));
    inputStd   = std(imageDouble(:));
    outputStd  = std(filteredImage(:));

    inputEnergy  = sum(imageDouble(:).^2);
    outputEnergy = sum(filteredImage(:).^2);

    %% ---------------------------------------------------------------
    % 9. DIFFERENCE IMAGE
    % ---------------------------------------------------------------

    differenceImage  = filteredImage - imageDouble;
    differenceEnergy = sum(differenceImage(:).^2);

    %% ---------------------------------------------------------------
    % 10. BUILD RESULT
    % ---------------------------------------------------------------

    result = struct();

    result.input          = originalImage;
    result.inputDouble    = imageDouble;
    result.filteredImage  = filteredImage;
    result.differenceImage = differenceImage;

    result.filterType  = filterType;
    result.kernel      = kernel;
    result.kernelSize  = size(kernel);
    result.sigma       = sigma;
    result.motionLength = motionLength;
    result.motionAngle  = motionAngle;
    result.boundary    = boundary;
    result.operation   = operation;

    result.inputMean   = inputMean;
    result.outputMean  = outputMean;
    result.inputStandardDeviation  = inputStd;
    result.outputStandardDeviation = outputStd;

    result.inputEnergy       = inputEnergy;
    result.outputEnergy      = outputEnergy;
    result.differenceEnergy  = differenceEnergy;
    result.computationTime   = computationTime;

end
