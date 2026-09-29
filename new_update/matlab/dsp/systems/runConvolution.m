function result = runConvolution(x, h, Fs, outputMode, method)
%RUNCONVOLUTION Perform and analyze linear discrete-time convolution.
%
%   result = runConvolution(x, h, Fs, outputMode, method)
%
%   SPANDHAN DSP — System Response Analysis
%
%   Computes the linear convolution y[n] = x[n] * h[n], which models
%   the response of a discrete-time LTI system with impulse response
%   h[n] to an input signal x[n]:
%
%       y[n] = sum_{k} x[k] * h[n-k]
%
%   Two computation strategies are supported:
%
%       Direct (O(Nx*Nh))   — efficient for short signals
%       FFT    (O(N log N)) — efficient for long signals
%
%   "auto" selects the faster strategy based on operation count.
%
%   Inputs
%   ------
%   x           : Input signal vector (1-D, real or complex)
%   h           : Impulse response / system kernel vector (1-D)
%   Fs          : Sampling frequency in Hz (positive scalar)
%   outputMode  : "full"  → Nx+Nh-1 samples (complete convolution)
%                 "same"  → Nx samples (same length as input)
%                 "valid" → max(Nx,Nh)-min(Nx,Nh)+1 samples (full overlap only)
%   method      : "direct" → MATLAB conv()
%                 "fft"    → FFT-domain multiplication
%                 "auto"   → choose based on Nx*Nh vs. threshold
%
%   Output
%   ------
%   result is a struct with fields:
%
%     Signals
%       .input                  Original x (column vector)
%       .impulseResponse        Original h (column vector)
%       .output                 Convolution result y (column vector)
%
%     Time axes
%       .inputTime              Time vector for x  (s)
%       .impulseTime            Time vector for h  (s)
%       .time                   Time vector for y  (s)
%
%     Configuration
%       .samplingFrequency      Fs (Hz)
%       .outputMode             "full" | "same" | "valid"
%       .method                 "direct" | "fft" (resolved from "auto")
%
%     Lengths
%       .inputLength            Nx
%       .impulseResponseLength  Nh
%       .fullOutputLength       Nx + Nh - 1
%       .outputLength           length(y)
%
%     Energy  (sum of squared magnitudes)
%       .inputEnergy
%       .impulseResponseEnergy
%       .outputEnergy
%
%     RMS
%       .inputRMS
%       .impulseResponseRMS
%       .outputRMS
%
%     Peak amplitude and its time location
%       .inputPeak              .inputPeakTime
%       .impulseResponsePeak    .impulsePeakTime
%       .outputPeak             .outputPeakTime
%
%     Computation metadata
%       .computationTime        Wall-clock seconds for convolution step
%       .fftLength              FFT size used ([] if direct method)
%
%   Example
%   -------
%       % Sinusoid through a decaying oscillatory system
%       Fs = 16000;
%       t  = (0:1/Fs:1-1/Fs)';
%       x  = sin(2*pi*1000*t);
%       th = (0:499)' / Fs;
%       h  = exp(-20*th) .* cos(2*pi*500*th);
%       h  = h / max(abs(h));
%       result = runConvolution(x, h, Fs, "full", "auto");
%
%   Viva note
%   ---------
%   FFT convolution avoids circular aliasing by zero-padding both
%   signals to length Nfft = 2^nextpow2(Nx+Nh-1) before computing
%   the IDFT of X(k)*H(k).  Without this padding, MATLAB's fft()
%   produces circular (periodic) convolution instead of linear.

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(5, 5);

    % Force column vectors; isolate from caller's workspace
    x = x(:);
    h = h(:);

    % Non-empty check
    if isempty(x)
        error("SPANDHAN:Convolution:EmptyInput", ...
            "Input signal x cannot be empty.");
    end

    if isempty(h)
        error("SPANDHAN:Convolution:EmptyImpulseResponse", ...
            "Impulse response h cannot be empty.");
    end

    % Sampling frequency
    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    % No NaN / Inf
    if any(~isfinite(x))
        error("SPANDHAN:Convolution:InvalidInput", ...
            "Input signal x contains NaN or Inf values.");
    end

    if any(~isfinite(h))
        error("SPANDHAN:Convolution:InvalidImpulseResponse", ...
            "Impulse response h contains NaN or Inf values.");
    end

    % String arguments
    outputMode = lower(string(outputMode));
    method     = lower(string(method));

    validModes   = ["full", "same", "valid"];
    validMethods = ["auto", "direct", "fft"];

    if ~ismember(outputMode, validModes)
        error("SPANDHAN:Convolution:InvalidOutputMode", ...
            "outputMode must be 'full', 'same', or 'valid'. Got: '%s'.", ...
            outputMode);
    end

    if ~ismember(method, validMethods)
        error("SPANDHAN:Convolution:InvalidMethod", ...
            "method must be 'auto', 'direct', or 'fft'. Got: '%s'.", ...
            method);
    end

    %% ================================================================
    %  2. SIGNAL LENGTHS
    %  ================================================================

    Nx    = length(x);
    Nh    = length(h);
    Nfull = Nx + Nh - 1;       % Length of the complete linear convolution

    %% ================================================================
    %  3. AUTOMATIC METHOD SELECTION
    %  ================================================================
    %
    %  Direct  complexity: O(Nx * Nh)
    %  FFT     complexity: O(N log N) where N = 2^ceil(log2(Nx+Nh-1))
    %
    %  Crossover empirically near Nx*Nh ~ 5e6 for MATLAB.

    if method == "auto"
        if double(Nx) * double(Nh) <= 5e6
            selectedMethod = "direct";
        else
            selectedMethod = "fft";
        end
    else
        selectedMethod = method;
    end

    %% ================================================================
    %  4. CONVOLUTION — FULL OUTPUT
    %  ================================================================
    %
    %  We always compute the FULL linear convolution first.
    %  Trimming to "same" / "valid" is handled in Section 5.
    %
    %  FFT method
    %  ----------
    %  Linear convolution via DFT requires zero-padding to Nfft >= Nfull
    %  to prevent time-domain aliasing (circular wrap-around).
    %  We choose Nfft = 2^nextpow2(Nfull) for FFT efficiency.

    tic;

    switch selectedMethod

        case "direct"

            % MATLAB's conv() performs exact linear convolution
            yFull = conv(x, h, "full");

        case "fft"

            Nfft = 2^nextpow2(Nfull);   % Power-of-2 ≥ Nx+Nh-1

            X = fft(x, Nfft);
            H = fft(h, Nfft);

            % Multiply spectra → circular convolution of length Nfft
            % Since Nfft >= Nfull, no aliasing → equivalent to linear
            yFull = real(ifft(X .* H));  % real() removes floating-point imag residue

            yFull = yFull(1 : Nfull);   % Discard zero-padded tail

    end

    computationTime = toc;

    %% ================================================================
    %  5. TRIM TO REQUESTED OUTPUT MODE
    %  ================================================================
    %
    %  "full"  — All Nfull samples; start of h[0] aligns with x[0]
    %  "same"  — Central Nx samples matching the length of the input
    %  "valid" — Samples where x and h fully overlap (no boundary effects)
    %
    %  We delegate "same" and "valid" trimming to MATLAB's conv() to
    %  guarantee identical boundary conventions.

    switch outputMode

        case "full"
            y = yFull;

        case "same"
            % Use conv() to honour MATLAB's exact "same" centering rule
            y = conv(x, h, "same");

        case "valid"
            y = conv(x, h, "valid");

    end

    %% ================================================================
    %  6. TIME AXES
    %  ================================================================

    inputTime   = (0 : Nx       - 1)' / Fs;
    impulseTime = (0 : Nh       - 1)' / Fs;
    outputTime  = (0 : length(y)- 1)' / Fs;

    %% ================================================================
    %  7. ENERGY  (Parseval: E = sum |s[n]|^2)
    %  ================================================================

    inputEnergy           = sum(abs(x).^2);
    impulseResponseEnergy = sum(abs(h).^2);
    outputEnergy          = sum(abs(y).^2);

    %% ================================================================
    %  8. RMS
    %  ================================================================

    inputRMS           = sqrt(mean(abs(x).^2));
    impulseResponseRMS = sqrt(mean(abs(h).^2));
    outputRMS          = sqrt(mean(abs(y).^2));

    %% ================================================================
    %  9. PEAK AMPLITUDE AND LOCATION
    %  ================================================================

    [inputPeak,   inputPeakIdx]   = max(abs(x));
    [impulsePeak, impulsePeakIdx] = max(abs(h));
    [outputPeak,  outputPeakIdx]  = max(abs(y));

    inputPeakTime   = inputTime(inputPeakIdx);
    impulsePeakTime = impulseTime(impulsePeakIdx);
    outputPeakTime  = outputTime(outputPeakIdx);

    %% ================================================================
    %  10. FFT SIZE METADATA
    %  ================================================================

    if selectedMethod == "fft"
        fftLength = 2^nextpow2(Nfull);
    else
        fftLength = [];
    end

    %% ================================================================
    %  11. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Signals ---
    result.input           = x;
    result.impulseResponse = h;
    result.output          = y;

    % --- Time axes ---
    result.inputTime   = inputTime;
    result.impulseTime = impulseTime;
    result.time        = outputTime;

    % --- Configuration ---
    result.samplingFrequency = Fs;
    result.outputMode        = outputMode;
    result.method            = selectedMethod;

    % --- Lengths ---
    result.inputLength             = Nx;
    result.impulseResponseLength   = Nh;
    result.fullOutputLength        = Nfull;
    result.outputLength            = length(y);

    % --- Energy ---
    result.inputEnergy           = inputEnergy;
    result.impulseResponseEnergy = impulseResponseEnergy;
    result.outputEnergy          = outputEnergy;

    % --- RMS ---
    result.inputRMS           = inputRMS;
    result.impulseResponseRMS = impulseResponseRMS;
    result.outputRMS          = outputRMS;

    % --- Peak amplitudes ---
    result.inputPeak           = inputPeak;
    result.impulseResponsePeak = impulsePeak;
    result.outputPeak          = outputPeak;

    % --- Peak time locations ---
    result.inputPeakTime   = inputPeakTime;
    result.impulsePeakTime = impulsePeakTime;
    result.outputPeakTime  = outputPeakTime;

    % --- Computation metadata ---
    result.computationTime = computationTime;
    result.fftLength       = fftLength;

end
