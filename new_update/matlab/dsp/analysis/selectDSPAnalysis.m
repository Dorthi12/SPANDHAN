function varargout = selectDSPAnalysis(varargin)
% SELECTDSPANALYSIS
% Selects relevant DSP analyses for a classified signal (Audio or Image),
% or executes DSP analysis if called with legacy (sig, Fs, signalClass, options) signature.
%
% USAGE:
%   1. Canonical Selection Mode:
%       analysis = selectDSPAnalysis(signalClass)            % Audio routing (default)
%       analysis = selectDSPAnalysis(signalClass, "image")   % Image routing
%       analysis = selectDSPAnalysis(signalClass, "audio")   % Audio routing
%
%   2. Legacy Execution Mode (backward compatibility):
%       res = selectDSPAnalysis(sig, Fs, signalClass)
%       res = selectDSPAnalysis(sig, Fs, signalClass, options)

    if nargin == 0
        error("selectDSPAnalysis requires at least 1 input.");
    end

    %% CANONICAL MODE: SELECTION ONLY
    if nargin == 1 || (nargin == 2 && (isstring(varargin{2}) || ischar(varargin{2})) && ...
            any(lower(string(varargin{2})) == ["audio", "image"]))

        signalClass = varargin{1};
        if nargin == 2
            modality = lower(string(varargin{2}));
        else
            modality = "audio";
        end

        if modality == "image"
            analysis = selectImageDSPAnalysis(signalClass);
        else
            analysis = selectAudioDSPAnalysis(signalClass);
        end

        varargout{1} = analysis;
        return;
    end

    %% LEGACY COMPATIBILITY MODE: (sig, Fs, signalClass, [options])
    sig = varargin{1};
    Fs  = varargin{2};
    if nargin >= 3
        signalClass = varargin{3};
    else
        signalClass = "sinusoidal";
    end

    options = struct();
    if nargin >= 4 && isstruct(varargin{4})
        options = varargin{4};
    end

    % Convert options struct to name-value pairs for analyzeAudio
    nvPairs = {};
    if isfield(options, 'firOrder')
        nvPairs = [nvPairs, {'FIROrder', options.firOrder}];
    end
    if isfield(options, 'firCutoff')
        nvPairs = [nvPairs, {'FIRCutoff', options.firCutoff}];
    end
    if isfield(options, 'firType')
        nvPairs = [nvPairs, {'FIRType', options.firType}];
    end
    if isfield(options, 'fftSize')
        nvPairs = [nvPairs, {'FFTSize', options.fftSize}];
    end
    if isfield(options, 'waveletName')
        nvPairs = [nvPairs, {'Wavelet', options.waveletName}];
    end
    if isfield(options, 'waveletLevel')
        nvPairs = [nvPairs, {'WaveletLevel', options.waveletLevel}];
    end
    if isfield(options, 'impulseResponse')
        nvPairs = [nvPairs, {'ImpulseResponse', options.impulseResponse}];
    end

    % Run canonical analyzeAudio
    dspRes = analyzeAudio(sig, Fs, signalClass, nvPairs{:});

    % Build pipeline list
    selection = selectAudioDSPAnalysis(signalClass);
    pipelineCell = cell(numel(selection.all), 1);
    for k = 1:numel(selection.all)
        pipelineCell{k} = lower(char(selection.all(k)));
    end

    res = struct();
    res.signalClass = lower(string(selection.canonicalClass));
    if res.signalClass == "white noise"
        res.signalClass = "white_noise";
    end
    res.pipeline    = pipelineCell;
    res.selection   = selection;
    res.status      = dspRes.status;

    % Only expose fields that were selected/executed
    if selection.useFFT && ~isempty(dspRes.fft)
        res.fft = dspRes.fft;
    end
    if selection.useSTFT && ~isempty(dspRes.stft)
        res.stft = dspRes.stft;
    end
    if selection.useWavelet && ~isempty(dspRes.wavelet)
        res.wavelet = dspRes.wavelet;
    end
    if selection.useFIR && ~isempty(dspRes.fir)
        res.fir = dspRes.fir;
        if isfield(options, 'firOrder')
            res.fir.order = options.firOrder;
        end
    end
    if selection.useIIR && ~isempty(dspRes.iir)
        res.iir = dspRes.iir;
    end
    if selection.useConvolution
        res.convolution = dspRes.convolution;
        if isempty(res.convolution)
            res.convolution = struct('status', 'skipped_no_ir');
        end
    end
    if selection.useDeconvolution
        res.deconvolution = dspRes.deconvolution;
        if isempty(res.deconvolution)
            res.deconvolution = struct('status', 'skipped_no_ir');
        end
    end

    varargout{1} = res;

end
