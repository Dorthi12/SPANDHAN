function result = runImageConvolution(imageInput, kernel, outputMode)
% RUNIMAGECONVOLUTION
% Performs explicit 2-D image convolution.
%
% INPUTS:
%   imageInput - Grayscale or RGB image
%   kernel     - 2-D convolution kernel
%   outputMode - "same", "full", or "valid"
%
% OUTPUT:
%   result - Structure containing convolution output,
%            kernel, dimensions, energy and computation metrics.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 2
        error("runImageConvolution requires imageInput and kernel.");
    end

    if nargin < 3 || isempty(outputMode)
        outputMode = "same";
    end

    if isempty(imageInput)
        error("Input image cannot be empty.");
    end

    if isempty(kernel)
        error("Convolution kernel cannot be empty.");
    end

    if ~ismatrix(kernel)
        error("Kernel must be a 2-D matrix.");
    end

    outputMode = lower(string(outputMode));

    if ~any(outputMode == ["same", "full", "valid"])
        error("Output mode must be same, full, or valid.");
    end

    %% ---------------------------------------------------------------
    % 2. CONVERT TO DOUBLE
    % ---------------------------------------------------------------

    originalImage = imageInput;
    imageDouble   = im2double(imageInput);
    kernel        = double(kernel);

    %% ---------------------------------------------------------------
    % 3. FLIP KERNEL
    % ---------------------------------------------------------------

    % conv2 performs mathematical convolution.
    % Explicitly retain the flipped kernel for transparency.

    flippedKernel = rot90(kernel, 2);

    %% ---------------------------------------------------------------
    % 4. PERFORM CONVOLUTION
    % ---------------------------------------------------------------

    tic;

    if ndims(imageDouble) == 2

        output = conv2( ...
            imageDouble, ...
            kernel, ...
            char(outputMode));

    else

        output = [];

        for channel = 1:size(imageDouble, 3)

            channelOutput = conv2( ...
                imageDouble(:,:,channel), ...
                kernel, ...
                char(outputMode));

            if isempty(output)
                [r, c] = size(channelOutput);
                output = zeros(r, c, size(imageDouble, 3));
            end

            output(:,:,channel) = channelOutput;

        end

    end

    computationTime = toc;

    %% ---------------------------------------------------------------
    % 5. CALCULATE METRICS
    % ---------------------------------------------------------------

    inputEnergy  = sum(imageDouble(:).^2);
    outputEnergy = sum(output(:).^2);
    kernelEnergy = sum(kernel(:).^2);

    inputPeak  = max(abs(imageDouble(:)));
    outputPeak = max(abs(output(:)));

    %% ---------------------------------------------------------------
    % 6. BUILD RESULT
    % ---------------------------------------------------------------

    result = struct();

    result.input         = originalImage;
    result.inputDouble   = imageDouble;
    result.kernel        = kernel;
    result.flippedKernel = flippedKernel;
    result.output        = output;
    result.outputMode    = outputMode;

    result.inputSize  = size(imageDouble);
    result.kernelSize = size(kernel);
    result.outputSize = size(output);

    result.inputEnergy   = inputEnergy;
    result.outputEnergy  = outputEnergy;
    result.kernelEnergy  = kernelEnergy;
    result.inputPeak     = inputPeak;
    result.outputPeak    = outputPeak;
    result.computationTime = computationTime;

end
