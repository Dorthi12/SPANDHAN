function result = runFIR(x, Fs, filterOrder, cutoffFrequency, filterType, windowType)
%RUNFIR Design and apply an FIR digital filter for the SPANDHAN framework.
%
%   result = runFIR(x, Fs, filterOrder, cutoffFrequency, ...
%                   filterType, windowType)
%
%   Inputs
%   ------
%   x               : Input signal vector
%   Fs              : Sampling frequency in Hz
%   filterOrder     : FIR filter order (integer >= 1)
%   cutoffFrequency : Cutoff frequency/frequencies in Hz (scalar for low/high, 2-element vector for bandpass/stop)
%   filterType      : "low", "high", "bandpass", or "stop"
%   windowType      : "hamming", "hann", "blackman", or "rectangular"
%
%   Output
%   ------
%   result is a structure containing:
%       result.input               : Original input signal (column vector)
%       result.output              : Filtered signal output
%       result.coefficients        : Filter numerator (b) coefficients
%       result.denominator         : Filter denominator (a) coefficients (= 1)
%       result.order               : Filter order
%       result.filterType          : Type of filter ("low", "high", "bandpass", "stop")
%       result.cutoffFrequency     : Cutoff frequency/frequencies in Hz
%       result.normalizedCutoff     : Cutoff frequency normalized to Nyquist
%       result.window              : Window type used
%       result.frequency           : Frequency vector for response (Hz)
%       result.response            : Complex frequency response H(f)
%       result.magnitude           : Linear magnitude response |H(f)|
%       result.magnitudeDB         : Magnitude response in dB
%       result.phase               : Unwrapped phase response (rad)
%       result.groupDelay          : Group delay vector
%       result.groupDelayFrequency : Frequency vector for group delay (Hz)
%       result.impulseResponse     : Impulse response vector
%       result.impulseTime         : Time vector for impulse response (s)
%       result.peakFrequency       : Frequency at peak magnitude (Hz)
%       result.peakMagnitude       : Peak magnitude value
%       result.estimated3dBCutoff  : Estimated -3 dB cutoff frequency (Hz)
%       result.samplingFrequency   : Sampling frequency Fs (Hz)
%       result.nyquistFrequency    : Nyquist frequency (Hz)
%       result.responsePoints      : Number of points evaluated for frequency response
%
%   Example
%   -------
%       result = runFIR(x, 16000, 100, 3000, "low", "hamming");

    %% ================================================================
    %  1. INPUT VALIDATION
    %  ================================================================

    narginchk(6, 6);

    % Convert signal to column vector
    x = x(:);

    % Validate sampling frequency
    validateattributes(Fs, ...
        {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'}, ...
        mfilename, 'Fs');

    % Validate filter order
    validateattributes(filterOrder, ...
        {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, ...
        mfilename, 'filterOrder');

    % FIR filter order should normally be at least 1
    if filterOrder < 1
        error("SPANDHAN:FIR:InvalidOrder", ...
            "FIR filter order must be at least 1.");
    end

    % Convert string/char inputs consistently
    filterType = lower(string(filterType));
    windowType = lower(string(windowType));

    % Validate filter type
    validFilterTypes = [ ...
        "low", ...
        "high", ...
        "bandpass", ...
        "stop" ...
    ];

    if ~ismember(filterType, validFilterTypes)
        error("SPANDHAN:FIR:InvalidFilterType", ...
            "Filter type must be: low, high, bandpass, or stop.");
    end

    % Validate cutoff frequencies
    validateattributes(cutoffFrequency, ...
        {'numeric'}, ...
        {'real', 'finite', 'positive'}, ...
        mfilename, 'cutoffFrequency');

    cutoffFrequency = cutoffFrequency(:).';

    %% ================================================================
    %  2. NYQUIST FREQUENCY VALIDATION
    %  ================================================================

    Nyquist = Fs / 2;

    % Number of required cutoff frequencies
    if filterType == "low" || filterType == "high"

        if numel(cutoffFrequency) ~= 1
            error("SPANDHAN:FIR:InvalidCutoff", ...
                "Low-pass and high-pass filters require one cutoff frequency.");
        end

    elseif filterType == "bandpass" || filterType == "stop"

        if numel(cutoffFrequency) ~= 2
            error("SPANDHAN:FIR:InvalidCutoff", ...
                "Band-pass and band-stop filters require two cutoff frequencies.");
        end

        if cutoffFrequency(1) >= cutoffFrequency(2)
            error("SPANDHAN:FIR:InvalidCutoffOrder", ...
                "For band filters, the first cutoff must be smaller than the second.");
        end
    end

    % All frequencies must be below Nyquist
    if any(cutoffFrequency >= Nyquist)
        error("SPANDHAN:FIR:CutoffAboveNyquist", ...
            "Cutoff frequency must be below the Nyquist frequency (%.2f Hz).", ...
            Nyquist);
    end

    %% ================================================================
    %  3. NORMALIZE CUTOFF FREQUENCY
    %  ================================================================

    normalizedCutoff = cutoffFrequency / Nyquist;

    %% ================================================================
    %  4. SELECT WINDOW
    %  ================================================================

    switch windowType

        case "hamming"
            window = hamming(filterOrder + 1);

        case "hann"
            window = hann(filterOrder + 1);

        case "blackman"
            window = blackman(filterOrder + 1);

        case "rectangular"
            window = rectwin(filterOrder + 1);

        otherwise
            error("SPANDHAN:FIR:InvalidWindow", ...
                "Unsupported window. Use hamming, hann, blackman, or rectangular.");

    end

    %% ================================================================
    %  5. DESIGN FIR FILTER
    %  ================================================================

    b = fir1( ...
        filterOrder, ...
        normalizedCutoff, ...
        char(filterType), ...
        window, ...
        "noscale");

    % FIR denominator
    a = 1;

    %% ================================================================
    %  6. APPLY FILTER
    %  ================================================================

    % filtfilt performs forward-backward zero-phase filtering
    try
        y = filtfilt(b, a, x);
    catch ME
        error("SPANDHAN:FIR:FilteringFailed", ...
            "FIR filtering failed: %s", ME.message);
    end

    %% ================================================================
    %  7. FREQUENCY RESPONSE
    %  ================================================================

    responsePoints = 4096;

    [H, f] = freqz( ...
        b, ...
        a, ...
        responsePoints, ...
        Fs);

    magnitude = abs(H);

    magnitudeDB = 20 * log10( ...
        max(magnitude, eps));

    phase = unwrap(angle(H));

    %% ================================================================
    %  8. GROUP DELAY
    %  ================================================================

    [groupDelay, groupDelayFrequency] = grpdelay( ...
        b, ...
        a, ...
        responsePoints, ...
        Fs);

    %% ================================================================
    %  9. IMPULSE RESPONSE
    %  ================================================================

    impulseLength = filterOrder + 1;

    impulseInput = zeros(impulseLength, 1);
    impulseInput(1) = 1;

    impulseResponse = filter( ...
        b, ...
        a, ...
        impulseInput);

    impulseTime = ...
        (0:impulseLength-1)' / Fs;

    %% ================================================================
    %  10. FILTER CHARACTERISTICS
    %  ================================================================

    [peakMagnitude, peakIndex] = max(magnitude);

    peakFrequency = f(peakIndex);

    % Approximate -3 dB cutoff frequency
    passbandMagnitudeDB = magnitudeDB;

    cutoffIndex = find( ...
        passbandMagnitudeDB <= ...
        (max(passbandMagnitudeDB) - 3), ...
        1, ...
        "first");

    if isempty(cutoffIndex)
        estimatedCutoff = NaN;
    else
        estimatedCutoff = f(cutoffIndex);
    end

    %% ================================================================
    %  11. STORE RESULTS
    %  ================================================================

    result = struct();

    % Input/output
    result.input = x;
    result.output = y;

    % Filter coefficients
    result.coefficients = b;
    result.denominator = a;

    % Filter configuration
    result.order = filterOrder;
    result.filterType = filterType;
    result.cutoffFrequency = cutoffFrequency;
    result.normalizedCutoff = normalizedCutoff;
    result.window = windowType;

    % Frequency response
    result.frequency = f;
    result.response = H;
    result.magnitude = magnitude;
    result.magnitudeDB = magnitudeDB;
    result.phase = phase;

    % Group delay
    result.groupDelay = groupDelay;
    result.groupDelayFrequency = groupDelayFrequency;

    % Impulse response
    result.impulseResponse = impulseResponse;
    result.impulseTime = impulseTime;

    % Important characteristics
    result.peakFrequency = peakFrequency;
    result.peakMagnitude = peakMagnitude;
    result.estimated3dBCutoff = estimatedCutoff;

    % Sampling information
    result.samplingFrequency = Fs;
    result.nyquistFrequency = Nyquist;

    % Metadata
    result.responsePoints = responsePoints;

end
