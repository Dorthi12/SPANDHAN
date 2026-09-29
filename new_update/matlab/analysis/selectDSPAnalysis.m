function results = selectDSPAnalysis(signal, Fs, signalClass, varargin)
%SELECTDSPANALYSIS Route a signal through its class-specific DSP pipeline.
%
%   results = selectDSPAnalysis(signal, Fs, signalClass)
%   results = selectDSPAnalysis(signal, Fs, signalClass, options)
%
%   SPANDHAN DSP ROUTING ENGINE
%
%   Dispatches the preprocessed audio signal through the set of DSP
%   operations defined for each recognised signal class.
%
%   Supported signal classes and their DSP chains
%   ──────────────────────────────────────────────
%   "impulse"     → FFT · Wavelet · FIR · Convolution · Deconvolution
%   "sinusoidal"  → FFT · FIR · IIR
%   "white_noise" → FFT · FIR · IIR
%   "step"        → FFT · Wavelet · FIR · IIR · Convolution
%   "chirp"       → FFT · STFT · Wavelet · Deconvolution
%
%   Inputs
%   ------
%   signal      : Preprocessed mono audio signal (column vector)
%   Fs          : Sampling frequency in Hz
%   signalClass : String — one of the five class names above
%   options     : (optional) struct with override parameters:
%                   .firOrder           (default 100)
%                   .firCutoff          (default Fs*0.2)
%                   .firType            (default "low")
%                   .firWindow          (default "hamming")
%                   .iirPassband        (default Fs*0.2)
%                   .iirStopband        (default Fs*0.25)
%                   .iirRipple          (default 1)
%                   .iirAttenuation     (default 60)
%                   .iirType            (default "low")
%                   .iirFamily          (default "butter")
%                   .convImpulseLength  (default 200)
%                   .deconvLambda       (default 0.01)
%                   .deconvMethod       (default "regularized")
%                   .waveletName        (default "db4")
%
%   Output
%   ------
%   results is a struct. Each field corresponds to one DSP module,
%   present only if that module was executed for the given class:
%
%       results.fft           runFFT output struct
%       results.stft          runSTFT output struct
%       results.wavelet       runWavelet output struct
%       results.fir           runFIR output struct
%       results.iir           runIIR output struct
%       results.convolution   runConvolution output struct
%       results.deconvolution runDeconvolution output struct
%       results.signalClass   String — resolved class name
%       results.pipeline      Cell array of executed module names
%
%   Example
%   -------
%       results = selectDSPAnalysis(signal, 16000, "chirp");
%       results = selectDSPAnalysis(signal, 16000, "step", opts);

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(3, 4);

    signal = signal(:);

    if isempty(signal)
        error("SPANDHAN:DSPRouter:EmptySignal", ...
            "Input signal cannot be empty.");
    end

    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    signalClass = lower(string(signalClass));

    validClasses = ["impulse", "sinusoidal", "white_noise", "step", "chirp"];

    if ~ismember(signalClass, validClasses)
        error("SPANDHAN:DSPRouter:InvalidClass", ...
            "signalClass must be one of: impulse | sinusoidal | white_noise | step | chirp. Got: '%s'.", ...
            signalClass);
    end

    %% ================================================================
    %  2. DEFAULT OPTIONS WITH OPTIONAL OVERRIDE
    %  ================================================================

    opts = struct();

    % FIR defaults
    opts.firOrder  = 100;
    opts.firCutoff = Fs * 0.2;
    opts.firType   = "low";
    opts.firWindow = "hamming";

    % IIR defaults
    opts.iirPassband    = Fs * 0.20;
    opts.iirStopband    = Fs * 0.25;
    opts.iirRipple      = 1;
    opts.iirAttenuation = 60;
    opts.iirType        = "low";
    opts.iirFamily      = "butter";

    % Convolution defaults — decaying oscillatory system kernel
    opts.convImpulseLength = 200;

    % Deconvolution defaults
    opts.deconvLambda  = 0.01;
    opts.deconvMethod  = "regularized";

    % Wavelet defaults
    opts.waveletName = "db4";

    % Merge caller-supplied overrides
    if nargin == 4 && isstruct(varargin{1})
        override = varargin{1};
        fields   = fieldnames(override);
        for k = 1:numel(fields)
            opts.(fields{k}) = override.(fields{k});
        end
    end

    %% ================================================================
    %  3. CLASS-TO-PIPELINE ROUTING TABLE
    %  ================================================================
    %
    %  Each class maps to an ordered cell array of module names.
    %  The execution loop (Section 5) iterates this list in order.

    pipelineMap = containers.Map();

    pipelineMap("impulse")    = { "fft", "wavelet", "fir", "convolution", "deconvolution" };
    pipelineMap("sinusoidal") = { "fft", "fir", "iir" };
    pipelineMap("white_noise")= { "fft", "fir", "iir" };
    pipelineMap("step")       = { "fft", "wavelet", "fir", "iir", "convolution" };
    pipelineMap("chirp")      = { "fft", "stft", "wavelet", "deconvolution" };

    pipeline = pipelineMap(char(signalClass));

    %% ================================================================
    %  4. INITIALISE RESULT STRUCT
    %  ================================================================

    results = struct();
    results.signalClass = signalClass;
    results.pipeline    = pipeline;

    %% ================================================================
    %  5. EXECUTE PIPELINE MODULES IN ORDER
    %  ================================================================

    for k = 1:numel(pipeline)

        module = pipeline{k};

        try

            switch module

                %------------------------------------------------------
                case "fft"
                %------------------------------------------------------
                    [f, Y] = runFFT(signal, Fs);
                    fftResult.frequency = f;
                    fftResult.amplitude = Y;
                    results.fft = fftResult;

                %------------------------------------------------------
                case "stft"
                %------------------------------------------------------
                    [S, F, T] = runSTFT(signal, Fs);
                    stftResult.spectrogram = S;
                    stftResult.frequency   = F;
                    stftResult.time        = T;
                    results.stft = stftResult;

                %------------------------------------------------------
                case "wavelet"
                %------------------------------------------------------
                    [c, l] = runWavelet(signal, opts.waveletName);
                    waveletResult.coefficients = c;
                    waveletResult.lengths      = l;
                    waveletResult.waveletName  = opts.waveletName;
                    results.wavelet = waveletResult;

                %------------------------------------------------------
                case "fir"
                %------------------------------------------------------
                    results.fir = runFIR( ...
                        signal, ...
                        Fs, ...
                        opts.firOrder, ...
                        opts.firCutoff, ...
                        opts.firType, ...
                        opts.firWindow);

                %------------------------------------------------------
                case "iir"
                %------------------------------------------------------
                    results.iir = runIIR( ...
                        signal, ...
                        Fs, ...
                        opts.iirPassband, ...
                        opts.iirStopband, ...
                        opts.iirRipple, ...
                        opts.iirAttenuation, ...
                        opts.iirType, ...
                        opts.iirFamily);

                %------------------------------------------------------
                case "convolution"
                %------------------------------------------------------
                    % Build a decaying oscillatory kernel for the
                    % system response.  The kernel is normalised so
                    % that the convolution output has comparable scale.
                    th = (0 : opts.convImpulseLength - 1)' / Fs;
                    h  = exp(-20 * th) .* cos(2*pi * (Fs * 0.05) * th);
                    h  = h / max(abs(h));

                    results.convolution = runConvolution( ...
                        signal, h, Fs, "full", "auto");

                %------------------------------------------------------
                case "deconvolution"
                %------------------------------------------------------
                    % Build the same kernel used in convolution so that
                    % deconvolution is self-consistent within the chain.
                    th = (0 : opts.convImpulseLength - 1)' / Fs;
                    h  = exp(-20 * th) .* cos(2*pi * (Fs * 0.05) * th);
                    h  = h / max(abs(h));

                    % Create the observed (convolved) signal to deconvolve
                    yObserved = conv(signal, h, "full");

                    results.deconvolution = runDeconvolution( ...
                        yObserved, h, Fs, ...
                        opts.deconvLambda, ...
                        opts.deconvMethod);

            end % switch

        catch ME

            % Record the failure in the result but continue the pipeline.
            % This prevents one failing module from blocking others.
            warning("SPANDHAN:DSPRouter:ModuleFailed", ...
                "Module '%s' failed for class '%s': %s", ...
                module, signalClass, ME.message);

            results.(module) = struct( ...
                "error",   ME.message, ...
                "module",  module, ...
                "class",   signalClass);

        end % try

    end % for

end
