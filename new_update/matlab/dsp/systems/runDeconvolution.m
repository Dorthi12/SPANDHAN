function result = runDeconvolution(y, h, Fs, lambda, method)
%RUNDECONVOLUTION Recover an input signal from a convolved system output.
%
%   result = runDeconvolution(y, h, Fs, lambda, method)
%
%   SPANDHAN DSP — System Inverse Analysis
%
%   Given the convolution model:
%
%       y[n] = x[n] * h[n]
%
%   estimate the original signal x̂[n] ≈ x[n] from the observed output
%   y[n] and the known/estimated impulse response h[n].
%
%   Three deconvolution strategies are provided:
%
%   ┌──────────────┬────────────────────────────────────────────────────┐
%   │ "regularized"│ Regularized Wiener-type inverse filter (default)   │
%   │              │    X̂(f) =    H*(f)      · Y(f)                    │
%   │              │           ────────────                             │
%   │              │           |H(f)|² + λ                             │
%   │              │ λ prevents amplification where |H(f)| ≈ 0         │
%   ├──────────────┼────────────────────────────────────────────────────┤
%   │ "direct"     │ Naive frequency-domain division with floor clamp   │
%   │              │    X̂(f) = Y(f) / H_safe(f)                        │
%   │              │ Sensitive to noise; use only for clean signals     │
%   ├──────────────┼────────────────────────────────────────────────────┤
%   │ "wiener"     │ Power-spectrum Wiener filter                       │
%   │              │    W(f) =   H*(f)                                  │
%   │              │           ─────────────────                        │
%   │              │           max(|H|²,ε) + λ_noise                   │
%   │              │ λ here represents the noise power level            │
%   └──────────────┴────────────────────────────────────────────────────┘
%
%   Inputs
%   ------
%   y       : Observed/output signal vector (y = x * h, possibly noisy)
%   h       : Known or estimated impulse response vector
%   Fs      : Sampling frequency in Hz
%   lambda  : Regularization parameter (≥ 0)
%             Larger λ → more stable, less exact recovery
%             Smaller λ → more aggressive, noise-amplifying recovery
%   method  : "regularized" | "direct" | "wiener"
%
%   Output
%   ------
%   result is a struct with fields:
%
%     Signals
%       .observed               Original y (column vector)
%       .impulseResponse        Original h (column vector)
%       .reconstructed          Estimated x̂ (column vector, length = Ny)
%
%     Time axes
%       .observedTime           Time vector for y  (s)
%       .impulseTime            Time vector for h  (s)
%       .reconstructedTime      Time vector for x̂ (s)
%
%     Frequency-domain data
%       .frequency              Frequency axis 0…Fs (Hz), length = Nfft
%       .observedSpectrum       Y(f)  = fft(y, Nfft)
%       .impulseSpectrum        H(f)  = fft(h, Nfft)
%       .reconstructedSpectrum  X̂(f) = fft(x̂, Nfft)
%
%     Transfer function
%       .transferFunction       H(f)  (alias for impulseSpectrum)
%       .transferMagnitude      |H(f)|
%       .transferPhase          unwrap(∠H(f))
%
%     Configuration
%       .samplingFrequency      Fs
%       .lambda                 Regularization value used
%       .method                 Resolved method string
%       .fftLength              Nfft (power of 2 ≥ Ny+Nh-1)
%
%     Lengths
%       .observedLength         Ny
%       .impulseResponseLength  Nh
%       .reconstructedLength    length(x̂)  (= Ny)
%
%     Reconstruction quality metrics
%       .RMSE                   RMS error between y and x̂ (diagnostic)
%       .MAE                    Mean absolute error
%       .correlation            Pearson correlation(y, x̂)
%
%     Energy
%       .observedEnergy         sum(|y|²)
%       .impulseResponseEnergy  sum(|h|²)
%       .reconstructedEnergy    sum(|x̂|²)
%
%     RMS
%       .observedRMS            sqrt(mean(|y|²))
%       .reconstructedRMS       sqrt(mean(|x̂|²))
%
%     Peak amplitude and time
%       .observedPeak           .observedPeakTime
%       .reconstructedPeak      .reconstructedPeakTime
%
%     Transfer function conditioning
%       .minimumTransferMagnitude   min|H(f)|  (near zero → ill-conditioned)
%       .maximumTransferMagnitude   max|H(f)|
%       .conditioningRatio          min/max ∈ [0,1]; near 0 → ill-conditioned
%
%     Computation metadata
%       .computationTime        Wall-clock seconds
%
%   Example
%   -------
%       % Build a ground-truth convolution, then recover original
%       Fs = 16000;
%       t  = (0:1/Fs:1-1/Fs)';
%       x  = sin(2*pi*1000*t);
%       th = (0:499)' / Fs;
%       h  = exp(-20*th) .* cos(2*pi*500*th);
%       h  = h / max(abs(h));
%       y  = conv(x, h, "full") + 0.005*randn(length(x)+length(h)-1, 1);
%       result = runDeconvolution(y, h, Fs, 0.01, "regularized");
%
%   Viva note — why regularization matters
%   ----------------------------------------
%   Direct inverse:  X̂ = Y/H  → diverges wherever |H(f)| ≈ 0 (null of system)
%   Regularized:     λ in the denominator bounds the gain, trading accuracy
%   for numerical stability.  This is identical in spirit to Tikhonov
%   regularization in linear algebra.

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(5, 5);

    % Coerce to column vectors, isolate from caller's workspace
    y = y(:);
    h = h(:);

    % Non-empty checks
    if isempty(y)
        error("SPANDHAN:Deconvolution:EmptyObservedSignal", ...
            "Observed signal y cannot be empty.");
    end

    if isempty(h)
        error("SPANDHAN:Deconvolution:EmptyImpulseResponse", ...
            "Impulse response h cannot be empty.");
    end

    % Sampling frequency
    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    % Regularization parameter (λ ≥ 0; λ = 0 → direct inversion)
    validateattributes(lambda, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'nonnegative'}, mfilename, 'lambda');

    % No NaN / Inf in signals
    if any(~isfinite(y))
        error("SPANDHAN:Deconvolution:InvalidObservedSignal", ...
            "Observed signal y contains NaN or Inf values.");
    end

    if any(~isfinite(h))
        error("SPANDHAN:Deconvolution:InvalidImpulseResponse", ...
            "Impulse response h contains NaN or Inf values.");
    end

    % Method string
    method = lower(string(method));

    validMethods = ["regularized", "direct", "wiener"];

    if ~ismember(method, validMethods)
        error("SPANDHAN:Deconvolution:InvalidMethod", ...
            "method must be 'regularized', 'direct', or 'wiener'. Got: '%s'.", ...
            method);
    end

    %% ================================================================
    %  2. SIGNAL LENGTHS AND FFT SIZE
    %  ================================================================

    Ny   = length(y);
    Nh   = length(h);

    % Zero-pad to prevent circular convolution artefacts.
    % Using Ny+Nh-1 ensures the DFT length exceeds the linear
    % convolution length; rounding up to the next power of 2
    % maximises FFT efficiency.
    Nfft = 2^nextpow2(Ny + Nh - 1);

    %% ================================================================
    %  3. FORWARD DFT
    %  ================================================================

    Y = fft(y, Nfft);       % Observed signal spectrum
    H = fft(h, Nfft);       % Impulse response spectrum

    % Transfer function characterisation
    Hmagnitude = abs(H);
    Hphase     = unwrap(angle(H));

    %% ================================================================
    %  4. DECONVOLUTION IN FREQUENCY DOMAIN
    %  ================================================================
    %
    %  Each method estimates X̂(f) such that X̂(f)·H(f) ≈ Y(f).
    %
    %  After IFFT, real() discards floating-point imaginary residue
    %  that should be exactly zero for real-valued input signals.

    tic;

    switch method

        %--------------------------------------------------------------
        case "regularized"
        %--------------------------------------------------------------
        %  Regularized inverse filter  (recommended default)
        %
        %            H*(f)
        %  X̂(f) = ────────────── · Y(f)
        %          |H(f)|² + λ
        %
        %  When |H(f)| is large (strong system response):
        %    denominator ≈ |H|²  →  reduces to H*/|H|² = 1/H  (exact)
        %  When |H(f)| ≈ 0 (system null):
        %    denominator ≈ λ    →  gain is bounded by 1/λ

            Xestimated = (conj(H) ./ (Hmagnitude.^2 + lambda)) .* Y;

        %--------------------------------------------------------------
        case "direct"
        %--------------------------------------------------------------
        %  Naive frequency-domain division with a magnitude floor.
        %
        %  X̂(f) = Y(f) / H_safe(f)
        %
        %  H_safe clamps values below `tolerance` to prevent
        %  division-by-zero.  Sensitive to noise; suitable only when
        %  y is noise-free and H has no deep nulls.

            tolerance         = max(Hmagnitude) * 1e-10;
            Hsafe             = H;
            Hsafe(Hmagnitude < tolerance) = tolerance;
            Xestimated        = Y ./ Hsafe;

        %--------------------------------------------------------------
        case "wiener"
        %--------------------------------------------------------------
        %  Power-spectrum Wiener filter
        %
        %            H*(f)
        %  W(f) = ──────────────────
        %         max(|H(f)|², ε) + λ_noise
        %
        %  Identical in form to "regularized" but λ is interpreted
        %  as the noise power.  max(|H|², ε) prevents a zero
        %  denominator even at system nulls.

            signalPower = max(Hmagnitude.^2, eps);
            WienerGain  = conj(H) ./ (signalPower + lambda);
            Xestimated  = WienerGain .* Y;

    end

    computationTime = toc;

    %% ================================================================
    %  5. INVERSE DFT → TIME DOMAIN
    %  ================================================================
    %
    %  Trim to Ny to match the original observed signal length.
    %  The reconstruction is never longer than the observed signal.

    xEstimatedFull = real(ifft(Xestimated));
    xEstimated     = xEstimatedFull(1 : Ny);

    %% ================================================================
    %  6. TIME AXES
    %  ================================================================

    observedTime       = (0 : Ny - 1)'             / Fs;
    impulseTime        = (0 : Nh - 1)'             / Fs;
    reconstructedTime  = (0 : length(xEstimated)-1)' / Fs;

    %% ================================================================
    %  7. SPECTRA
    %  ================================================================

    reconstructedSpectrum = fft(xEstimated, Nfft);
    frequency             = (0 : Nfft - 1)' * Fs / Nfft;

    %% ================================================================
    %  8. RECONSTRUCTION QUALITY METRICS
    %  ================================================================
    %
    %  Note: comparing y vs x̂ is a diagnostic only.
    %  y = x*h and x̂ ≈ x, so they are not theoretically identical.
    %  A meaningful ground-truth comparison requires the original x.

    compLen  = min(Ny, length(xEstimated));
    yComp    = y(1 : compLen);
    xComp    = xEstimated(1 : compLen);
    errSig   = yComp - xComp;

    RMSE = sqrt(mean(abs(errSig).^2));
    MAE  = mean(abs(errSig));

    % Pearson correlation (requires non-constant signals)
    if std(yComp) > 0 && std(xComp) > 0
        correlation = corr(yComp, xComp);
    else
        correlation = NaN;
    end

    %% ================================================================
    %  9. ENERGY
    %  ================================================================

    observedEnergy       = sum(abs(y).^2);
    impulseEnergy        = sum(abs(h).^2);
    reconstructedEnergy  = sum(abs(xEstimated).^2);

    %% ================================================================
    %  10. RMS
    %  ================================================================

    observedRMS      = sqrt(mean(abs(y).^2));
    reconstructedRMS = sqrt(mean(abs(xEstimated).^2));

    %% ================================================================
    %  11. PEAK AMPLITUDE AND LOCATION
    %  ================================================================

    [observedPeak,      obsIdx]   = max(abs(y));
    [reconstructedPeak, recIdx]   = max(abs(xEstimated));

    observedPeakTime      = observedTime(obsIdx);
    reconstructedPeakTime = reconstructedTime(recIdx);

    %% ================================================================
    %  12. TRANSFER FUNCTION CONDITIONING
    %  ================================================================
    %
    %  conditioningRatio ≈ 1 → H(f) is flat (well-conditioned)
    %  conditioningRatio ≈ 0 → H(f) has deep nulls (ill-conditioned)

    minH = min(Hmagnitude);
    maxH = max(Hmagnitude);

    if maxH > 0
        conditioningRatio = minH / maxH;
    else
        conditioningRatio = 0;
    end

    %% ================================================================
    %  13. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Signals ---
    result.observed        = y;
    result.impulseResponse = h;
    result.reconstructed   = xEstimated;

    % --- Time axes ---
    result.observedTime      = observedTime;
    result.impulseTime       = impulseTime;
    result.reconstructedTime = reconstructedTime;

    % --- Frequency-domain data ---
    result.frequency             = frequency;
    result.observedSpectrum      = Y;
    result.impulseSpectrum       = H;
    result.reconstructedSpectrum = reconstructedSpectrum;

    % --- Transfer function ---
    result.transferFunction  = H;         % alias for impulseSpectrum
    result.transferMagnitude = Hmagnitude;
    result.transferPhase     = Hphase;

    % --- Configuration ---
    result.samplingFrequency = Fs;
    result.lambda            = lambda;
    result.method            = method;
    result.fftLength         = Nfft;

    % --- Lengths ---
    result.observedLength          = Ny;
    result.impulseResponseLength   = Nh;
    result.reconstructedLength     = length(xEstimated);

    % --- Quality metrics ---
    result.RMSE        = RMSE;
    result.MAE         = MAE;
    result.correlation = correlation;

    % --- Energy ---
    result.observedEnergy       = observedEnergy;
    result.impulseResponseEnergy = impulseEnergy;
    result.reconstructedEnergy  = reconstructedEnergy;

    % --- RMS ---
    result.observedRMS      = observedRMS;
    result.reconstructedRMS = reconstructedRMS;

    % --- Peak amplitudes ---
    result.observedPeak      = observedPeak;
    result.reconstructedPeak = reconstructedPeak;

    % --- Peak time locations ---
    result.observedPeakTime      = observedPeakTime;
    result.reconstructedPeakTime = reconstructedPeakTime;

    % --- Conditioning ---
    result.minimumTransferMagnitude = minH;
    result.maximumTransferMagnitude = maxH;
    result.conditioningRatio        = conditioningRatio;

    % --- Computation ---
    result.computationTime = computationTime;

end
