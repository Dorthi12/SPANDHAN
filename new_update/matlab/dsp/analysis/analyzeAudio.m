function result = analyzeAudio(x, Fs, signalClass, varargin)
% ANALYZEAUDIO
% Classification-aware DSP analysis engine for audio signals.
%
% INPUTS:
%   x           - Audio signal
%   Fs          - Sampling frequency
%   signalClass - ML-predicted signal class
%
% OPTIONAL NAME-VALUE PARAMETERS:
%
%   "FFTSize"
%       Default: 4096
%
%   "STFTWindow"
%       Default: 1024
%
%   "STFTOverlap"
%       Default: 512
%
%   "STFTNFFT"
%       Default: 2048
%
%   "Wavelet"
%       Default: "db4"
%
%   "WaveletLevel"
%       Default: 5
%
%   "FIROrder"
%       Default: 100
%
%   "FIRCutoff"
%       Default: 3000 Hz
%
%   "FIRType"
%       Default: "low"
%
%   "FIRWindow"
%       Default: "hamming"
%
%   "IIRPassband"
%       Default: 3000 Hz
%
%   "IIRStopband"
%       Default: 4000 Hz
%
%   "IIRPassbandRipple"
%       Default: 1 dB
%
%   "IIRStopbandAttenuation"
%       Default: 60 dB
%
%   "IIRType"
%       Default: "low"
%
%   "IIRFamily"
%       Default: "butter"
%
%   "ImpulseResponse"
%       Default: []
%
%   "DeconvolutionLambda"
%       Default: 0.01
%
% OUTPUT:
%   result - Unified DSP analysis structure.
%
% IMPORTANT:
%   This function orchestrates existing DSP modules.
%   It does not replace those modules.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 3
        error("analyzeAudio requires x, Fs and signalClass.");
    end

    if isempty(x)
        error("Audio signal cannot be empty.");
    end

    if ~isnumeric(x)
        error("Audio signal must be numeric.");
    end

    if any(~isfinite(x(:)))
        error("Audio signal contains NaN or Inf values.");
    end

    if ~isscalar(Fs) || Fs <= 0
        error("Fs must be a positive scalar.");
    end

    %% ---------------------------------------------------------------
    % 2. DEFAULT PARAMETERS
    % ---------------------------------------------------------------

    fftSize = 4096;

    stftWindow  = 1024;
    stftOverlap = 512;
    stftNFFT    = 2048;

    waveletName  = "db4";
    waveletLevel = 5;

    firOrder  = 100;
    firCutoff = 3000;
    firType   = "low";
    firWindow = "hamming";

    iirPassband             = 3000;
    iirStopband             = 4000;
    iirPassbandRipple       = 1;
    iirStopbandAttenuation  = 60;
    iirType                 = "low";
    iirFamily               = "butter";

    impulseResponse     = [];
    deconvolutionLambda = 0.01;

    %% ---------------------------------------------------------------
    % 3. PARSE OPTIONAL PARAMETERS
    % ---------------------------------------------------------------

    if mod(length(varargin), 2) ~= 0
        error("Optional parameters must be name-value pairs.");
    end

    for k = 1:2:length(varargin)

        parameter = lower(string(varargin{k}));
        value     = varargin{k+1};

        switch parameter

            case "fftsize"
                fftSize = value;

            case "stftwindow"
                stftWindow = value;

            case "stftoverlap"
                stftOverlap = value;

            case "stftnfft"
                stftNFFT = value;

            case "wavelet"
                waveletName = string(value);

            case "waveletlevel"
                waveletLevel = value;

            case "firorder"
                firOrder = value;

            case "fircutoff"
                firCutoff = value;

            case "firtype"
                firType = string(value);

            case "firwindow"
                firWindow = string(value);

            case "iirpassband"
                iirPassband = value;

            case "iirstopband"
                iirStopband = value;

            case "iirpassbandripple"
                iirPassbandRipple = value;

            case "iirstopbandattenuation"
                iirStopbandAttenuation = value;

            case "iirtype"
                iirType = string(value);

            case "iirfamily"
                iirFamily = string(value);

            case "impulseresponse"
                impulseResponse = value;

            case "deconvolutionlambda"
                deconvolutionLambda = value;

            otherwise
                error("Unknown parameter: %s", parameter);

        end
    end

    %% ---------------------------------------------------------------
    % 4. CONVERT AUDIO TO MONO COLUMN VECTOR
    % ---------------------------------------------------------------

    originalSignal = x;

    if size(x, 2) > 1
        x = mean(x, 2);
    else
        x = x(:);
    end

    %% ---------------------------------------------------------------
    % 5. SELECT DSP ANALYSIS
    % ---------------------------------------------------------------

    analysisSelection = selectDSPAnalysis(signalClass);

    %% ---------------------------------------------------------------
    % 6. INITIALIZE RESULT STRUCTURE
    % ---------------------------------------------------------------

    result = struct();

    result.signalClass       = analysisSelection.canonicalClass;
    result.analysisSelection = analysisSelection;
    result.input             = originalSignal;
    result.analysisSignal    = x;
    result.samplingFrequency = Fs;
    result.signalLength      = length(x);
    result.duration          = length(x) / Fs;

    result.fft          = [];
    result.stft         = [];
    result.wavelet      = [];
    result.fir          = [];
    result.iir          = [];
    result.convolution  = [];
    result.deconvolution = [];

    result.executionLog   = strings(0, 1);
    result.failedAnalyses = strings(0, 1);

    %% ---------------------------------------------------------------
    % 7. FFT
    % ---------------------------------------------------------------

    if analysisSelection.useFFT

        try

            result.fft = runFFT(x, Fs, fftSize);

            result.executionLog(end+1) = "FFT completed.";

        catch ME

            result.failedAnalyses(end+1) = "FFT";

            result.executionLog(end+1) = ...
                "FFT failed: " + string(ME.message);

        end
    end

    %% ---------------------------------------------------------------
    % 8. STFT
    % ---------------------------------------------------------------

    if analysisSelection.useSTFT

        try

            result.stft = runSTFT( ...
                x, Fs, stftWindow, stftOverlap, stftNFFT);

            result.executionLog(end+1) = "STFT completed.";

        catch ME

            result.failedAnalyses(end+1) = "STFT";

            result.executionLog(end+1) = ...
                "STFT failed: " + string(ME.message);

        end
    end

    %% ---------------------------------------------------------------
    % 9. WAVELET
    % ---------------------------------------------------------------

    if analysisSelection.useWavelet

        try

            result.wavelet = runWavelet( ...
                x, Fs, waveletName, waveletLevel);

            result.executionLog(end+1) = "Wavelet analysis completed.";

        catch ME

            result.failedAnalyses(end+1) = "Wavelet";

            result.executionLog(end+1) = ...
                "Wavelet failed: " + string(ME.message);

        end
    end

    %% ---------------------------------------------------------------
    % 10. FIR FILTER
    % ---------------------------------------------------------------

    if analysisSelection.useFIR

        try

            result.fir = runFIR( ...
                x, Fs, firOrder, firCutoff, firType, firWindow);

            result.executionLog(end+1) = "FIR filtering completed.";

        catch ME

            result.failedAnalyses(end+1) = "FIR";

            result.executionLog(end+1) = ...
                "FIR failed: " + string(ME.message);

        end
    end

    %% ---------------------------------------------------------------
    % 11. IIR FILTER
    % ---------------------------------------------------------------

    if analysisSelection.useIIR

        try

            result.iir = runIIR( ...
                x, Fs, ...
                iirPassband, ...
                iirStopband, ...
                iirPassbandRipple, ...
                iirStopbandAttenuation, ...
                iirType, ...
                iirFamily);

            result.executionLog(end+1) = "IIR filtering completed.";

        catch ME

            result.failedAnalyses(end+1) = "IIR";

            result.executionLog(end+1) = ...
                "IIR failed: " + string(ME.message);

        end
    end

    %% ---------------------------------------------------------------
    % 12. CONVOLUTION
    % ---------------------------------------------------------------

    if analysisSelection.useConvolution

        if isempty(impulseResponse)

            result.executionLog(end+1) = ...
                "Convolution skipped: no impulse response supplied.";

        else

            try

                result.convolution = runConvolution( ...
                    x, impulseResponse, Fs, "full", "auto");

                result.executionLog(end+1) = "Convolution completed.";

            catch ME

                result.failedAnalyses(end+1) = "Convolution";

                result.executionLog(end+1) = ...
                    "Convolution failed: " + string(ME.message);

            end
        end
    end

    %% ---------------------------------------------------------------
    % 13. DECONVOLUTION
    % ---------------------------------------------------------------

    if analysisSelection.useDeconvolution

        if isempty(impulseResponse)

            result.executionLog(end+1) = ...
                "Deconvolution skipped: no impulse response supplied.";

        else

            try

                result.deconvolution = runDeconvolution( ...
                    x, impulseResponse, Fs, ...
                    deconvolutionLambda, "regularized");

                result.executionLog(end+1) = "Deconvolution completed.";

            catch ME

                result.failedAnalyses(end+1) = "Deconvolution";

                result.executionLog(end+1) = ...
                    "Deconvolution failed: " + string(ME.message);

            end
        end
    end

    %% ---------------------------------------------------------------
    % 14. SUMMARY
    % ---------------------------------------------------------------

    result.completedAnalyses = strings(0, 1);

    if ~isempty(result.fft)
        result.completedAnalyses(end+1) = "FFT";
    end

    if ~isempty(result.stft)
        result.completedAnalyses(end+1) = "STFT";
    end

    if ~isempty(result.wavelet)
        result.completedAnalyses(end+1) = "Wavelet";
    end

    if ~isempty(result.fir)
        result.completedAnalyses(end+1) = "FIR";
    end

    if ~isempty(result.iir)
        result.completedAnalyses(end+1) = "IIR";
    end

    if ~isempty(result.convolution)
        result.completedAnalyses(end+1) = "Convolution";
    end

    if ~isempty(result.deconvolution)
        result.completedAnalyses(end+1) = "Deconvolution";
    end

    %% ---------------------------------------------------------------
    % 15. STATUS
    % ---------------------------------------------------------------

    if isempty(result.failedAnalyses)

        result.status = "Completed";

    elseif isempty(result.completedAnalyses)

        result.status = "Failed";

    else

        result.status = "Completed with warnings";

    end

end
