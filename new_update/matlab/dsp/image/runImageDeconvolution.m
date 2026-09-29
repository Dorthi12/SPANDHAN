function result = runImageDeconvolution( ...
        blurredImage, psf, method, varargin)
% RUNIMAGEDECONVOLUTION
% Restores a blurred image using Wiener or regularized deconvolution.
%
% INPUTS:
%   blurredImage - Observed blurred/noisy image
%   psf          - Point Spread Function
%   method       - "wiener", "regularized", or "inverse"
%
% OPTIONAL NAME-VALUE:
%   "NSR"        - Noise-to-signal ratio for Wiener filtering
%   "NoisePower" - Noise power for regularized deconvolution
%   "Original"   - Original image for evaluation metrics
%   "EdgeTaper"  - true/false, default true
%
% OUTPUT:
%   result - Structure containing restored image,
%            PSF, method, error metrics and configuration.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 2
        error("runImageDeconvolution requires blurredImage and PSF.");
    end

    if nargin < 3 || isempty(method)
        method = "wiener";
    end

    if isempty(blurredImage)
        error("Blurred image cannot be empty.");
    end

    if isempty(psf)
        error("PSF cannot be empty.");
    end

    if ~ismatrix(psf)
        error("PSF must be a 2-D matrix.");
    end

    method = lower(string(method));

    validMethods = ["wiener", "regularized", "inverse"];

    if ~any(method == validMethods)
        error("Method must be wiener, regularized, or inverse.");
    end

    %% ---------------------------------------------------------------
    % 2. DEFAULT PARAMETERS
    % ---------------------------------------------------------------

    nsr           = 0.01;
    noisePower    = 0.01;
    originalImage = [];
    edgeTaper     = true;

    %% ---------------------------------------------------------------
    % 3. PARSE NAME-VALUE PARAMETERS
    % ---------------------------------------------------------------

    if mod(length(varargin), 2) ~= 0
        error("Optional arguments must be name-value pairs.");
    end

    for k = 1:2:length(varargin)

        parameter = lower(string(varargin{k}));
        value     = varargin{k+1};

        switch parameter
            case "nsr"
                nsr = value;
            case "noisepower"
                noisePower = value;
            case "original"
                originalImage = value;
            case "edgetaper"
                edgeTaper = logical(value);
            otherwise
                error("Unknown parameter: %s", parameter);
        end

    end

    %% ---------------------------------------------------------------
    % 4. CONVERT INPUT TO DOUBLE
    % ---------------------------------------------------------------

    blurredOriginal = blurredImage;
    blurredImage    = im2double(blurredImage);
    psf             = double(psf);

    if ~isempty(originalImage)
        originalImage = im2double(originalImage);
    end

    %% ---------------------------------------------------------------
    % 5. NORMALIZE PSF
    % ---------------------------------------------------------------

    psfSum = sum(psf(:));

    if abs(psfSum) > eps
        psf = psf / psfSum;
    end

    %% ---------------------------------------------------------------
    % 6. OPTIONAL EDGE TAPER
    % ---------------------------------------------------------------

    if edgeTaper
        processedInput = edgetaper(blurredImage, psf);
    else
        processedInput = blurredImage;
    end

    %% ---------------------------------------------------------------
    % 7. PERFORM DECONVOLUTION
    % ---------------------------------------------------------------

    tic;

    if method == "wiener"

        restoredImage = deconvwnr( ...
            processedInput, ...
            psf, ...
            nsr);

    elseif method == "regularized"

        restoredImage = deconvreg( ...
            processedInput, ...
            psf, ...
            noisePower);

    else

        % -----------------------------------------------------------
        % Frequency-domain inverse filtering.
        % Protected against division by values close to zero.
        % -----------------------------------------------------------

        [rows, cols, channels] = size(processedInput);

        psfOTF = psf2otf(psf, [rows cols]);

        denominator = abs(psfOTF).^2;

        threshold = max(denominator(:)) * 1e-8;

        restoredImage = zeros(size(processedInput));

        if ndims(processedInput) == 2

            G = fft2(processedInput);
            X = zeros(size(G));
            valid = denominator > threshold;
            X(valid) = G(valid) ./ psfOTF(valid);
            restoredImage = real(ifft2(X));

        else

            for channel = 1:channels

                G = fft2(processedInput(:,:,channel));
                X = zeros(size(G));
                valid = denominator > threshold;
                X(valid) = G(valid) ./ psfOTF(valid);
                restoredImage(:,:,channel) = real(ifft2(X));

            end

        end

    end

    computationTime = toc;

    %% ---------------------------------------------------------------
    % 8. CLIP OUTPUT TO DISPLAY RANGE
    % ---------------------------------------------------------------

    restoredImage = min(max(restoredImage, 0), 1);

    %% ---------------------------------------------------------------
    % 9. BASIC METRICS
    % ---------------------------------------------------------------

    inputEnergy  = sum(blurredImage(:).^2);
    outputEnergy = sum(restoredImage(:).^2);

    %% ---------------------------------------------------------------
    % 10. QUALITY METRICS IF ORIGINAL IS AVAILABLE
    % ---------------------------------------------------------------

    hasReference = false;
    rmse         = NaN;
    mae          = NaN;
    correlation  = NaN;
    psnrValue    = NaN;

    if ~isempty(originalImage)

        if ~isequal(size(originalImage), size(restoredImage))
            error("Original image must have the same size as the input.");
        end

        hasReference = true;

        errorImage = originalImage - restoredImage;

        rmse = sqrt(mean(errorImage(:).^2));
        mae  = mean(abs(errorImage(:)));

        referenceVector = originalImage(:);
        restoredVector  = restoredImage(:);

        if std(referenceVector) > eps && std(restoredVector) > eps
            correlation = corr(referenceVector, restoredVector);
        end

        mse = mean(errorImage(:).^2);

        if mse > eps
            psnrValue = 10 * log10(1 / mse);
        else
            psnrValue = Inf;
        end

    end

    %% ---------------------------------------------------------------
    % 11. BUILD RESULT
    % ---------------------------------------------------------------

    result = struct();

    result.input          = blurredOriginal;
    result.inputDouble    = blurredImage;
    result.processedInput = processedInput;
    result.restoredImage  = restoredImage;

    result.psf     = psf;
    result.psfSize = size(psf);
    result.method  = method;

    result.nsr        = nsr;
    result.noisePower = noisePower;
    result.edgeTaper  = edgeTaper;

    result.hasReference = hasReference;
    result.rmse         = rmse;
    result.mae          = mae;
    result.correlation  = correlation;
    result.psnr         = psnrValue;

    result.inputEnergy     = inputEnergy;
    result.outputEnergy    = outputEnergy;
    result.computationTime = computationTime;

end
