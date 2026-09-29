function result = analyzeImage(imageInput, signalClass, varargin)
% ANALYZEIMAGE
% Classification-aware DSP analysis engine for images.
%
% INPUTS:
%   imageInput  - Grayscale or RGB image
%   signalClass - ML-predicted image signal class
%
% OPTIONAL NAME-VALUE PARAMETERS:
%
%   "FFTAnalysis"
%       true/false, default true
%
%   "WaveletName"
%       Default: "db4"
%
%   "WaveletLevel"
%       Default: 3
%
%   "FilterType"
%       Default: "gaussian"
%
%   "FilterKernelSize"
%       Default: 5
%
%   "FilterSigma"
%       Default: 1
%
%   "FilterBoundary"
%       Default: "symmetric"
%
%   "FilterOperation"
%       Default: "convolution"
%
%   "FilterKernel"
%       Default: []
%
%   "ConvolutionKernel"
%       Default: []
%
%   "ConvolutionOutputMode"
%       Default: "same"
%
%   "PSF"
%       Point Spread Function for deconvolution.
%       Default: []
%
%   "DeconvolutionMethod"
%       Default: "wiener"
%
%   "NSR"
%       Wiener noise-to-signal ratio.
%       Default: 0.01
%
%   "NoisePower"
%       Regularized deconvolution noise power.
%       Default: 0.01
%
%   "EdgeTaper"
%       true/false, default true
%
% OUTPUT:
%   result - Unified image DSP analysis structure.
%
% IMPORTANT:
%   This function orchestrates the individual image DSP modules.
%   It does not replace them.

    %% ===============================================================
    % 1. INPUT VALIDATION
    % ===============================================================

    if nargin < 2
        error("analyzeImage requires imageInput and signalClass.");
    end

    if isempty(imageInput)
        error("Input image cannot be empty.");
    end

    if ~isnumeric(imageInput) && ~islogical(imageInput)
        error("Input must be numeric or logical.");
    end

    if ndims(imageInput) ~= 2 && ndims(imageInput) ~= 3
        error("Input must be a grayscale or RGB image.");
    end

    if ndims(imageInput) == 3 && size(imageInput, 3) ~= 3
        error("RGB image must have exactly 3 channels.");
    end

    if any(~isfinite(double(imageInput(:))))
        error("Input image contains NaN or Inf values.");
    end

    %% ===============================================================
    % 2. DEFAULT PARAMETERS
    % ===============================================================

    fftAnalysis = true;

    waveletName  = "db4";
    waveletLevel = 3;

    filterType      = "gaussian";
    filterKernelSize = 5;
    filterSigma     = 1;
    filterBoundary  = "symmetric";
    filterOperation = "convolution";
    filterKernel    = [];

    convolutionKernel     = [];
    convolutionOutputMode = "same";

    psf                 = [];
    deconvolutionMethod = "wiener";
    nsr                 = 0.01;
    noisePower          = 0.01;
    edgeTaper           = true;

    %% ===============================================================
    % 3. PARSE NAME-VALUE PARAMETERS
    % ===============================================================

    if mod(length(varargin), 2) ~= 0
        error("Optional parameters must be supplied as name-value pairs.");
    end

    for k = 1:2:length(varargin)

        parameter = lower(string(varargin{k}));
        value     = varargin{k+1};

        switch parameter

            case "fftanalysis"
                fftAnalysis = logical(value);

            case "waveletname"
                waveletName = string(value);

            case "waveletlevel"
                waveletLevel = value;

            case "filtertype"
                filterType = string(value);

            case "filterkernelsize"
                filterKernelSize = value;

            case "filtersigma"
                filterSigma = value;

            case "filterboundary"
                filterBoundary = string(value);

            case "filteroperation"
                filterOperation = string(value);

            case "filterkernel"
                filterKernel = value;

            case "convolutionkernel"
                convolutionKernel = value;

            case "convolutionoutputmode"
                convolutionOutputMode = string(value);

            case "psf"
                psf = value;

            case "deconvolutionmethod"
                deconvolutionMethod = string(value);

            case "nsr"
                nsr = value;

            case "noisepower"
                noisePower = value;

            case "edgetaper"
                edgeTaper = logical(value);

            otherwise
                error("Unknown parameter: %s", parameter);

        end
    end

    %% ===============================================================
    % 4. PRESERVE INPUT AND CONVERT
    % ===============================================================

    originalImage = imageInput;
    imageDouble   = im2double(imageInput);

    %% ===============================================================
    % 5. SELECT DSP PATH
    % ===============================================================
    %
    % selectDSPAnalysis() defines the generic signal-class routing.
    % analyzeImage() maps that selection to image-specific operations.
    % STFT, FIR, and IIR are audio-only — they are not applied to images.

    selection      = selectDSPAnalysis(signalClass);
    canonicalClass = selection.canonicalClass;

    %% ===============================================================
    % 6. INITIALIZE RESULT
    % ===============================================================

    result = struct();

    result.signalClass       = canonicalClass;
    result.analysisSelection = selection;
    result.input             = originalImage;
    result.analysisImage     = imageDouble;
    result.imageSize         = size(imageDouble);

    result.fft2         = [];
    result.wavelet2D    = [];
    result.filter       = [];
    result.convolution  = [];
    result.deconvolution = [];

    result.executionLog   = strings(0, 1);
    result.failedAnalyses = strings(0, 1);

    %% ===============================================================
    % 7. 2-D FFT
    % ===============================================================

    if fftAnalysis

        try

            result.fft2 = runFFT2(imageDouble);

            result.executionLog(end+1) = "2-D FFT completed.";

        catch ME

            result.failedAnalyses(end+1) = "2-D FFT";

            result.executionLog(end+1) = ...
                "2-D FFT failed: " + string(ME.message);

        end

    end

    %% ===============================================================
    % 8. 2-D WAVELET
    % ===============================================================

    try

        result.wavelet2D = runWavelet2D( ...
            imageDouble, waveletName, waveletLevel);

        result.executionLog(end+1) = ...
            "2-D wavelet analysis completed.";

    catch ME

        result.failedAnalyses(end+1) = "2-D Wavelet";

        result.executionLog(end+1) = ...
            "2-D Wavelet failed: " + string(ME.message);

    end

    %% ===============================================================
    % 9. IMAGE FILTERING
    % ===============================================================

    try

        if filterType == "custom"

            result.filter = runImageFilter( ...
                imageDouble, filterType, ...
                "Kernel",    filterKernel, ...
                "Boundary",  filterBoundary, ...
                "Operation", filterOperation);

        elseif filterType == "gaussian"

            result.filter = runImageFilter( ...
                imageDouble, filterType, ...
                "KernelSize", filterKernelSize, ...
                "Sigma",      filterSigma, ...
                "Boundary",   filterBoundary, ...
                "Operation",  filterOperation);

        else

            result.filter = runImageFilter( ...
                imageDouble, filterType, ...
                "KernelSize", filterKernelSize, ...
                "Boundary",   filterBoundary, ...
                "Operation",  filterOperation);

        end

        result.executionLog(end+1) = "Image filtering completed.";

    catch ME

        result.failedAnalyses(end+1) = "Image Filter";

        result.executionLog(end+1) = ...
            "Image filtering failed: " + string(ME.message);

    end

    %% ===============================================================
    % 10. 2-D CONVOLUTION
    % ===============================================================

    if ~isempty(convolutionKernel)

        try

            result.convolution = runImageConvolution( ...
                imageDouble, ...
                convolutionKernel, ...
                convolutionOutputMode);

            result.executionLog(end+1) = "2-D convolution completed.";

        catch ME

            result.failedAnalyses(end+1) = "2-D Convolution";

            result.executionLog(end+1) = ...
                "2-D convolution failed: " + string(ME.message);

        end

    else

        result.executionLog(end+1) = ...
            "2-D convolution skipped: no kernel supplied.";

    end

    %% ===============================================================
    % 11. IMAGE DECONVOLUTION
    % ===============================================================

    if ~isempty(psf)

        try

            if deconvolutionMethod == "wiener"

                result.deconvolution = runImageDeconvolution( ...
                    imageDouble, psf, deconvolutionMethod, ...
                    "NSR",       nsr, ...
                    "EdgeTaper", edgeTaper);

            elseif deconvolutionMethod == "regularized"

                result.deconvolution = runImageDeconvolution( ...
                    imageDouble, psf, deconvolutionMethod, ...
                    "NoisePower", noisePower, ...
                    "EdgeTaper",  edgeTaper);

            else

                result.deconvolution = runImageDeconvolution( ...
                    imageDouble, psf, deconvolutionMethod, ...
                    "EdgeTaper", edgeTaper);

            end

            result.executionLog(end+1) = ...
                "Image deconvolution completed.";

        catch ME

            result.failedAnalyses(end+1) = "Image Deconvolution";

            result.executionLog(end+1) = ...
                "Image deconvolution failed: " + string(ME.message);

        end

    else

        result.executionLog(end+1) = ...
            "Image deconvolution skipped: no PSF supplied.";

    end

    %% ===============================================================
    % 12. COMPLETED ANALYSES SUMMARY
    % ===============================================================

    result.completedAnalyses = strings(0, 1);

    if ~isempty(result.fft2)
        result.completedAnalyses(end+1) = "2-D FFT";
    end

    if ~isempty(result.wavelet2D)
        result.completedAnalyses(end+1) = "2-D Wavelet";
    end

    if ~isempty(result.filter)
        result.completedAnalyses(end+1) = "Image Filter";
    end

    if ~isempty(result.convolution)
        result.completedAnalyses(end+1) = "2-D Convolution";
    end

    if ~isempty(result.deconvolution)
        result.completedAnalyses(end+1) = "Image Deconvolution";
    end

    %% ===============================================================
    % 13. FINAL STATUS
    % ===============================================================

    if isempty(result.failedAnalyses)

        result.status = "Completed";

    elseif isempty(result.completedAnalyses)

        result.status = "Failed";

    else

        result.status = "Completed with warnings";

    end

end
