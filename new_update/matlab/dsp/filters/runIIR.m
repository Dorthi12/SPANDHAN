function result = runIIR(x, Fs, passbandFrequency, stopbandFrequency, ...
                          passbandRipple, stopbandAttenuation, ...
                          filterType, filterFamily)
%RUNIIR Design and apply an IIR digital filter for the SPANDHAN framework.
%
%   result = runIIR(x, Fs, passbandFrequency, stopbandFrequency, ...
%                   passbandRipple, stopbandAttenuation, ...
%                   filterType, filterFamily)
%
%   SPANDHAN IIR DSP ANALYSIS
%
%   Inputs
%   ------
%   x                    : Input signal vector (mono audio)
%   Fs                   : Sampling frequency in Hz
%   passbandFrequency    : Passband edge frequency/frequencies in Hz
%                          Scalar  → low / high
%                          2-element vector → bandpass / stop
%   stopbandFrequency    : Stopband edge frequency/frequencies in Hz
%                          Scalar  → low / high
%                          2-element vector → bandpass / stop
%   passbandRipple       : Maximum allowable passband ripple in dB (> 0)
%   stopbandAttenuation  : Minimum required stopband attenuation in dB (> 0)
%   filterType           : "low" | "high" | "bandpass" | "stop"
%   filterFamily         : "butter" | "cheby1" | "cheby2" | "ellip"
%
%   Output
%   ------
%   result                    : Structure containing all analysis data
%     .input                  : Original input signal (column vector)
%     .output                 : Zero-phase filtered output signal
%     .b                      : Numerator coefficients
%     .a                      : Denominator coefficients
%     .sos                    : Second-order sections matrix (numerically stable)
%     .scaleFactor            : SOS scale factor g
%     .order                  : Minimum computed filter order N
%     .filterType             : Filter type string
%     .filterFamily           : Filter family string
%     .passbandFrequency      : Passband edge(s) in Hz
%     .stopbandFrequency      : Stopband edge(s) in Hz
%     .passbandRipple         : Passband ripple in dB
%     .stopbandAttenuation    : Stopband attenuation in dB
%     .frequency              : Frequency vector for response (Hz)
%     .response               : Complex frequency response H(f)
%     .magnitude              : Linear magnitude |H(f)|
%     .magnitudeDB            : Magnitude in dB (20*log10)
%     .phase                  : Unwrapped phase response (rad)
%     .groupDelay             : Group delay (samples)
%     .groupDelayFrequency    : Frequency axis for group delay (Hz)
%     .impulseResponse        : Finite-length impulse response samples
%     .impulseTime            : Time axis for impulse response (s)
%     .poles                  : Pole locations (z-domain)
%     .poleMagnitudes         : Absolute magnitude of each pole
%     .stable                 : true if all poles lie inside unit circle
%     .peakFrequency          : Frequency at peak magnitude (Hz)
%     .peakMagnitude          : Peak linear magnitude value
%     .estimated3dBCutoff     : Estimated -3 dB cutoff frequency (Hz)
%     .samplingFrequency      : Fs (Hz)
%     .nyquistFrequency       : Nyquist frequency (Hz)
%     .responsePoints         : Number of FFT points used for response
%
%   Example
%   -------
%       result = runIIR(x, 16000, 3000, 4000, 1, 60, "low", "butter");
%
%   Filter family comparison
%   ------------------------
%   "butter"  → Maximally flat passband, smooth monotonic response
%   "cheby1"  → Equiripple passband, sharper roll-off than Butterworth
%   "cheby2"  → Equiripple stopband, monotonic passband
%   "ellip"   → Equiripple in both bands, sharpest transition (highest Q)

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(8, 8);

    % --- Signal ---
    x = x(:);                               % Force column vector

    if isempty(x)
        error("SPANDHAN:IIR:EmptySignal", ...
            "Input signal x cannot be empty.");
    end

    % --- Sampling frequency ---
    validateattributes(Fs, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');

    % --- Passband / stopband frequency vectors ---
    validateattributes(passbandFrequency, {'numeric'}, ...
        {'real', 'finite', 'positive', 'vector'}, mfilename, 'passbandFrequency');

    validateattributes(stopbandFrequency, {'numeric'}, ...
        {'real', 'finite', 'positive', 'vector'}, mfilename, 'stopbandFrequency');

    % --- Ripple and attenuation ---
    validateattributes(passbandRipple, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'passbandRipple');

    validateattributes(stopbandAttenuation, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, mfilename, 'stopbandAttenuation');

    % --- String arguments ---
    filterType   = lower(string(filterType));
    filterFamily = lower(string(filterFamily));

    %% ================================================================
    %  2. VALIDATE FILTER TYPE AND FAMILY
    %  ================================================================

    validTypes    = ["low", "high", "bandpass", "stop"];
    validFamilies = ["butter", "cheby1", "cheby2", "ellip"];

    if ~ismember(filterType, validTypes)
        error("SPANDHAN:IIR:InvalidFilterType", ...
            "filterType must be: low | high | bandpass | stop. Got: '%s'.", ...
            filterType);
    end

    if ~ismember(filterFamily, validFamilies)
        error("SPANDHAN:IIR:InvalidFilterFamily", ...
            "filterFamily must be: butter | cheby1 | cheby2 | ellip. Got: '%s'.", ...
            filterFamily);
    end

    %% ================================================================
    %  3. NORMALIZE FREQUENCY VECTORS TO ROW FORM
    %  ================================================================

    passbandFrequency = passbandFrequency(:).';
    stopbandFrequency = stopbandFrequency(:).';

    Nyquist = Fs / 2;

    %% ================================================================
    %  4. VALIDATE FREQUENCY VECTOR LENGTHS PER FILTER TYPE
    %  ================================================================

    isSingleEdge = filterType == "low" || filterType == "high";
    isDualEdge   = filterType == "bandpass" || filterType == "stop";

    if isSingleEdge

        if numel(passbandFrequency) ~= 1
            error("SPANDHAN:IIR:InvalidPassband", ...
                "%s filter requires exactly one passband edge frequency.", ...
                filterType);
        end

        if numel(stopbandFrequency) ~= 1
            error("SPANDHAN:IIR:InvalidStopband", ...
                "%s filter requires exactly one stopband edge frequency.", ...
                filterType);
        end

    elseif isDualEdge

        if numel(passbandFrequency) ~= 2
            error("SPANDHAN:IIR:InvalidPassband", ...
                "%s filter requires exactly two passband edge frequencies.", ...
                filterType);
        end

        if numel(stopbandFrequency) ~= 2
            error("SPANDHAN:IIR:InvalidStopband", ...
                "%s filter requires exactly two stopband edge frequencies.", ...
                filterType);
        end

    end

    %% ================================================================
    %  5. NYQUIST BOUNDARY CHECK
    %  ================================================================

    if any(passbandFrequency >= Nyquist)
        error("SPANDHAN:IIR:PassbandAboveNyquist", ...
            "All passband frequencies must be strictly below Nyquist (%.2f Hz).", ...
            Nyquist);
    end

    if any(stopbandFrequency >= Nyquist)
        error("SPANDHAN:IIR:StopbandAboveNyquist", ...
            "All stopband frequencies must be strictly below Nyquist (%.2f Hz).", ...
            Nyquist);
    end

    %% ================================================================
    %  6. PASSBAND / STOPBAND ORDERING VALIDATION
    %  ================================================================
    %
    %  Low-pass  : Fpass < Fstop
    %  High-pass : Fstop < Fpass
    %  Bandpass  : Fstop1 < Fpass1 < Fpass2 < Fstop2
    %  Band-stop : Fpass1 < Fstop1 < Fstop2 < Fpass2

    switch filterType

        case "low"

            if stopbandFrequency(1) <= passbandFrequency(1)
                error("SPANDHAN:IIR:InvalidLowpassEdges", ...
                    "Low-pass: Fstop (%.2f) must be > Fpass (%.2f).", ...
                    stopbandFrequency(1), passbandFrequency(1));
            end

        case "high"

            if stopbandFrequency(1) >= passbandFrequency(1)
                error("SPANDHAN:IIR:InvalidHighpassEdges", ...
                    "High-pass: Fstop (%.2f) must be < Fpass (%.2f).", ...
                    stopbandFrequency(1), passbandFrequency(1));
            end

        case "bandpass"

            if ~(stopbandFrequency(1) < passbandFrequency(1) && ...
                 passbandFrequency(1) < passbandFrequency(2)  && ...
                 passbandFrequency(2) < stopbandFrequency(2))
                error("SPANDHAN:IIR:InvalidBandpassEdges", ...
                    "Bandpass: required Fstop1 < Fpass1 < Fpass2 < Fstop2.");
            end

        case "stop"

            if ~(passbandFrequency(1) < stopbandFrequency(1) && ...
                 stopbandFrequency(1) < stopbandFrequency(2)  && ...
                 stopbandFrequency(2) < passbandFrequency(2))
                error("SPANDHAN:IIR:InvalidBandstopEdges", ...
                    "Band-stop: required Fpass1 < Fstop1 < Fstop2 < Fpass2.");
            end

    end

    %% ================================================================
    %  7. NORMALIZE FREQUENCIES TO [0, 1] w.r.t. NYQUIST
    %  ================================================================

    Wp = passbandFrequency / Nyquist;
    Ws = stopbandFrequency / Nyquist;

    %% ================================================================
    %  8. AUTOMATIC FILTER ORDER DETERMINATION
    %  ================================================================
    %
    %  Each family uses its own order-estimation function, which
    %  returns the minimum integer N satisfying the specifications
    %  and the corresponding natural frequency/frequencies Wn.

    switch filterFamily

        case "butter"
            [N, Wn] = buttord(Wp, Ws, passbandRipple, stopbandAttenuation);

        case "cheby1"
            [N, Wn] = cheb1ord(Wp, Ws, passbandRipple, stopbandAttenuation);

        case "cheby2"
            [N, Wn] = cheb2ord(Wp, Ws, passbandRipple, stopbandAttenuation);

        case "ellip"
            [N, Wn] = ellipord(Wp, Ws, passbandRipple, stopbandAttenuation);

    end

    %% ================================================================
    %  9. IIR FILTER COEFFICIENT DESIGN
    %  ================================================================

    switch filterFamily

        case "butter"
            [b, a] = butter(N, Wn, char(filterType));

        case "cheby1"
            [b, a] = cheby1(N, passbandRipple, Wn, char(filterType));

        case "cheby2"
            [b, a] = cheby2(N, stopbandAttenuation, Wn, char(filterType));

        case "ellip"
            [b, a] = ellip(N, passbandRipple, stopbandAttenuation, Wn, char(filterType));

    end

    %% ================================================================
    %  10. SECOND-ORDER SECTIONS CONVERSION
    %  ================================================================
    %
    %  High-order IIR filters represented as direct-form [b, a]
    %  can suffer from numerical sensitivity due to limited floating-
    %  point precision when evaluating large-degree polynomials.
    %
    %  Converting to SOS form cascades biquad (2nd-order) sections:
    %
    %        H(z) = g * prod_k [ (b0k + b1k*z^-1 + b2k*z^-2) /
    %                             (1  + a1k*z^-1 + a2k*z^-2) ]
    %
    %  Each section is numerically well-conditioned.
    %  filtfilt() natively supports SOS input.

    try
        [sos, g] = tf2sos(b, a);
    catch
        sos = [];
        g   = 1;
    end

    %% ================================================================
    %  11. ZERO-PHASE FILTERING
    %  ================================================================
    %
    %  filtfilt() applies the filter twice:
    %    (1) Forward  pass → introduces phase shift φ
    %    (2) Backward pass → introduces phase shift −φ
    %  Net result: zero phase distortion, doubled effective order.
    %
    %  SOS form is preferred for numerical stability.

    try

        if ~isempty(sos)
            y = filtfilt(sos, g, x);
        else
            y = filtfilt(b, a, x);
        end

    catch ME
        error("SPANDHAN:IIR:FilteringFailed", ...
            "Zero-phase IIR filtering failed: %s", ME.message);
    end

    %% ================================================================
    %  12. FREQUENCY RESPONSE (4096 points)
    %  ================================================================

    responsePoints = 4096;

    [H, f] = freqz(b, a, responsePoints, Fs);

    magnitude   = abs(H);
    magnitudeDB = 20 * log10(max(magnitude, eps));
    phase       = unwrap(angle(H));

    %% ================================================================
    %  13. GROUP DELAY
    %  ================================================================
    %
    %  Group delay = -d(phase)/d(omega).
    %  For IIR filters this is frequency-dependent (non-linear phase),
    %  which is an important contrast to linear-phase FIR.

    try
        [groupDelay, groupDelayFrequency] = grpdelay(b, a, responsePoints, Fs);
    catch
        groupDelay          = zeros(responsePoints, 1);
        groupDelayFrequency = f;
    end

    %% ================================================================
    %  14. IMPULSE RESPONSE (finite observation window)
    %  ================================================================
    %
    %  IIR filters have an infinite theoretical impulse response.
    %  We compute a practical finite window:
    %    length = max(4*(N+1), 512) samples
    %
    %  SOS-based filtering is used for numerical consistency.

    impulseLength = max(4 * (N + 1), 512);
    impulseInput  = zeros(impulseLength, 1);
    impulseInput(1) = 1;

    if ~isempty(sos)
        impulseResponse = g * sosfilt(sos, impulseInput);
    else
        impulseResponse = filter(b, a, impulseInput);
    end

    impulseTime = (0 : impulseLength - 1)' / Fs;

    %% ================================================================
    %  15. STABILITY ANALYSIS
    %  ================================================================
    %
    %  An IIR filter is BIBO-stable iff all poles lie strictly
    %  inside the unit circle in the z-plane: |p_k| < 1 for all k.

    poles          = roots(a);
    poleMagnitudes = abs(poles);
    stable         = all(poleMagnitudes < 1);

    %% ================================================================
    %  16. KEY FILTER CHARACTERISTICS
    %  ================================================================

    [peakMagnitude, peakIndex] = max(magnitude);
    peakFrequency = f(peakIndex);

    % Estimated -3 dB frequency (first crossing from passband)
    maxMagnitudeDB = max(magnitudeDB);
    idx3dB = find(magnitudeDB <= maxMagnitudeDB - 3, 1, "first");

    if isempty(idx3dB)
        estimated3dBCutoff = NaN;
    else
        estimated3dBCutoff = f(idx3dB);
    end

    %% ================================================================
    %  17. ASSEMBLE RESULT STRUCTURE
    %  ================================================================

    result = struct();

    % --- Signal data ---
    result.input  = x;
    result.output = y;

    % --- Transfer function coefficients ---
    result.b = b;
    result.a = a;

    % --- Numerically stable SOS representation ---
    result.sos         = sos;
    result.scaleFactor = g;

    % --- Filter design parameters ---
    result.order               = N;
    result.filterType          = filterType;
    result.filterFamily        = filterFamily;
    result.passbandFrequency   = passbandFrequency;
    result.stopbandFrequency   = stopbandFrequency;
    result.passbandRipple      = passbandRipple;
    result.stopbandAttenuation = stopbandAttenuation;

    % --- Frequency response ---
    result.frequency   = f;
    result.response    = H;
    result.magnitude   = magnitude;
    result.magnitudeDB = magnitudeDB;
    result.phase       = phase;

    % --- Group delay ---
    result.groupDelay          = groupDelay;
    result.groupDelayFrequency = groupDelayFrequency;

    % --- Impulse response ---
    result.impulseResponse = impulseResponse;
    result.impulseTime     = impulseTime;

    % --- Stability ---
    result.poles          = poles;
    result.poleMagnitudes = poleMagnitudes;
    result.stable         = stable;

    % --- Key characteristics ---
    result.peakFrequency      = peakFrequency;
    result.peakMagnitude      = peakMagnitude;
    result.estimated3dBCutoff = estimated3dBCutoff;

    % --- Sampling metadata ---
    result.samplingFrequency = Fs;
    result.nyquistFrequency  = Nyquist;
    result.responsePoints    = responsePoints;

end
