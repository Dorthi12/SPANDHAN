function analysis = selectAudioDSPAnalysis(signalClass)
% SELECTAUDIODSPANALYSIS
% Selects the most relevant DSP analyses for a classified audio signal.
%
% INPUT:
%   signalClass - Predicted signal class from the ML model (string, char, or numeric 0..4)
%
% Supported classes:
%   "Impulse"
%   "Sinusoidal"
%   "White Noise"
%   "Step"
%   "Chirp"
%
% OUTPUT:
%   analysis - Structure containing:
%       .signalClass
%       .canonicalClass
%       .primary
%       .secondary
%       .all
%       .description
%       .reason
%       .useFFT
%       .useSTFT
%       .useWavelet
%       .useFIR
%       .useIIR
%       .useConvolution
%       .useDeconvolution
%
% IMPORTANT:
%   This function only selects audio analyses.
%   It does NOT execute any DSP algorithm.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 1 || isempty(signalClass)
        error("Signal class is required.");
    end

    if isstring(signalClass)

        if numel(signalClass) ~= 1
            error("signalClass must be a scalar string.");
        end

    elseif ischar(signalClass)

        signalClass = string(signalClass);

    elseif isnumeric(signalClass)

        if ~isscalar(signalClass)
            error("Numeric signal class must be a scalar.");
        end

        % Expected ML labels:
        % 0 = Impulse
        % 1 = Sinusoidal
        % 2 = White Noise
        % 3 = Step
        % 4 = Chirp

        classLabels = [ ...
            "Impulse", ...
            "Sinusoidal", ...
            "White Noise", ...
            "Step", ...
            "Chirp" ...
        ];

        classIndex = double(signalClass) + 1;

        if classIndex < 1 || classIndex > numel(classLabels)
            error("Unknown numeric signal class: %g", signalClass);
        end

        signalClass = classLabels(classIndex);

    else

        error("signalClass must be text or a numeric class label.");

    end

    %% ---------------------------------------------------------------
    % 2. NORMALIZE CLASS NAME
    % ---------------------------------------------------------------

    normalizedClass = lower(strtrim(signalClass));
    normalizedClass = replace(normalizedClass, "_", " ");
    normalizedClass = replace(normalizedClass, "-", " ");

    %% ---------------------------------------------------------------
    % 3. CLASSIFY INTO CANONICAL CLASS
    % ---------------------------------------------------------------

    switch normalizedClass

        case {"impulse", "dirac delta", "delta", "dirac"}

            canonicalClass = "Impulse";

        case {"sinusoidal", "sinusoid", "sine", "sine wave"}

            canonicalClass = "Sinusoidal";

        case {"white noise", "whitenoise", "noise"}

            canonicalClass = "White Noise";

        case {"step", "step signal", "unit step"}

            canonicalClass = "Step";

        case {"chirp", "swept sine", "swept sine signal", "swept-sine"}

            canonicalClass = "Chirp";

        otherwise

            error("Unsupported signal class: %s", signalClass);

    end

    %% ---------------------------------------------------------------
    % 4. SELECT AUDIO DSP PATH
    % ---------------------------------------------------------------

    switch canonicalClass

        % -----------------------------------------------------------
        % IMPULSE
        % -----------------------------------------------------------
        case "Impulse"

            primary = [ ...
                "FFT"; ...
                "Wavelet" ...
            ];

            secondary = [ ...
                "FIR"; ...
                "Convolution"; ...
                "Deconvolution" ...
            ];

            description = ...
                "Transient and broadband audio signal with highly localized energy.";

            reason = ...
                "FFT reveals broadband spectral content, while wavelets " + ...
                "localize the impulse across scales. Filtering and system " + ...
                "response analysis are useful secondary operations.";

        % -----------------------------------------------------------
        % SINUSOIDAL
        % -----------------------------------------------------------
        case "Sinusoidal"

            primary = [ ...
                "FFT" ...
            ];

            secondary = [ ...
                "FIR"; ...
                "IIR" ...
            ];

            description = ...
                "Periodic signal dominated by one or more discrete frequencies.";

            reason = ...
                "FFT directly identifies the dominant frequency and spectral " + ...
                "components. FIR and IIR filtering can demonstrate frequency " + ...
                "selectivity.";

        % -----------------------------------------------------------
        % WHITE NOISE
        % -----------------------------------------------------------
        case "White Noise"

            primary = [ ...
                "FFT" ...
            ];

            secondary = [ ...
                "FIR"; ...
                "IIR" ...
            ];

            description = ...
                "Broadband stochastic signal with approximately flat power spectrum.";

            reason = ...
                "FFT reveals broadband spectral distribution. FIR and IIR " + ...
                "filters demonstrate how frequency-selective systems shape noise.";

        % -----------------------------------------------------------
        % STEP
        % -----------------------------------------------------------
        case "Step"

            primary = [ ...
                "FFT"; ...
                "Wavelet" ...
            ];

            secondary = [ ...
                "FIR"; ...
                "IIR"; ...
                "Convolution" ...
            ];

            description = ...
                "Piecewise-constant signal containing a sharp transition.";

            reason = ...
                "Wavelets are useful for localizing the transition, while FFT " + ...
                "shows its frequency content. Filtering and convolution reveal " + ...
                "system response to a step-like input.";

        % -----------------------------------------------------------
        % CHIRP
        % -----------------------------------------------------------
        case "Chirp"

            primary = [ ...
                "FFT"; ...
                "STFT"; ...
                "Wavelet" ...
            ];

            secondary = [ ...
                "Deconvolution" ...
            ];

            description = ...
                "Signal whose instantaneous frequency changes over time.";

            reason = ...
                "FFT shows the overall frequency range, while STFT tracks the " + ...
                "frequency evolution. Wavelets provide multi-resolution analysis.";

    end

    %% ---------------------------------------------------------------
    % 5. COMBINE ANALYSIS LIST
    % ---------------------------------------------------------------

    allAnalyses = [
        primary;
        secondary
    ];

    %% ---------------------------------------------------------------
    % 6. BUILD OUTPUT STRUCTURE
    % ---------------------------------------------------------------

    analysis = struct();

    analysis.signalClass    = signalClass;
    analysis.canonicalClass = canonicalClass;
    analysis.primary        = primary;
    analysis.secondary      = secondary;
    analysis.all            = allAnalyses;
    analysis.description    = description;
    analysis.reason         = reason;

    %% ---------------------------------------------------------------
    % 7. ANALYSIS FLAGS
    % ---------------------------------------------------------------

    analysis.useFFT           = any(allAnalyses == "FFT");
    analysis.useSTFT          = any(allAnalyses == "STFT");
    analysis.useWavelet       = any(allAnalyses == "Wavelet");
    analysis.useFIR           = any(allAnalyses == "FIR");
    analysis.useIIR           = any(allAnalyses == "IIR");
    analysis.useConvolution   = any(allAnalyses == "Convolution");
    analysis.useDeconvolution = any(allAnalyses == "Deconvolution");

end
