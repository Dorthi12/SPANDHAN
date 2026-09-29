function analysis = selectImageDSPAnalysis(signalClass)
% SELECTIMAGEDSPANALYSIS
% Selects the most relevant DSP analyses for a classified 2-D image signal.
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
%       .useFFT2
%       .useWavelet2D
%       .useFilter
%       .useConvolution
%       .useDeconvolution
%
% IMPORTANT:
%   This function only selects image DSP analyses.
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

        case {"impulse", "dirac delta", "delta", "dirac", "point"}

            canonicalClass = "Impulse";

        case {"sinusoidal", "sinusoid", "sine", "sine wave", "fringe"}

            canonicalClass = "Sinusoidal";

        case {"white noise", "whitenoise", "noise"}

            canonicalClass = "White Noise";

        case {"step", "step signal", "unit step", "edge"}

            canonicalClass = "Step";

        case {"chirp", "swept sine", "swept sine signal", "swept-sine"}

            canonicalClass = "Chirp";

        otherwise

            error("Unsupported image signal class: %s", signalClass);

    end

    %% ---------------------------------------------------------------
    % 4. SELECT IMAGE DSP PATH
    % ---------------------------------------------------------------

    switch canonicalClass

        % -----------------------------------------------------------
        % IMPULSE
        % -----------------------------------------------------------
        case "Impulse"

            primary = [ ...
                "2D FFT"; ...
                "2D Wavelet"; ...
                "Image Filter" ...
            ];

            secondary = [ ...
                "2D Convolution"; ...
                "2D Deconvolution" ...
            ];

            description = ...
                "Point-source impulse image with concentrated spatial energy.";

            reason = ...
                "2D FFT reveals broadband spatial frequency content; 2D wavelet isolates " + ...
                "localized spatial singularity; filtering, convolution, and deconvolution " + ...
                "evaluate PSF response.";

        % -----------------------------------------------------------
        % SINUSOIDAL
        % -----------------------------------------------------------
        case "Sinusoidal"

            primary = [ ...
                "2D FFT"; ...
                "Image Filter" ...
            ];

            secondary = [ ...
                "2D Convolution" ...
            ];

            description = ...
                "Periodic spatial pattern dominated by discrete spatial frequencies.";

            reason = ...
                "2D FFT isolates harmonic peaks in the spatial frequency domain; 2D filtering " + ...
                "and convolution demonstrate spatial frequency selectivity.";

        % -----------------------------------------------------------
        % WHITE NOISE
        % -----------------------------------------------------------
        case "White Noise"

            primary = [ ...
                "2D FFT"; ...
                "Image Filter" ...
            ];

            secondary = strings(0, 1);

            description = ...
                "Broadband stochastic 2D image with approximately uniform spatial power spectrum.";

            reason = ...
                "2D FFT reveals spatial noise distribution; image filtering illustrates " + ...
                "noise smoothing and suppression.";

        % -----------------------------------------------------------
        % STEP
        % -----------------------------------------------------------
        case "Step"

            primary = [ ...
                "2D FFT"; ...
                "2D Wavelet"; ...
                "Image Filter" ...
            ];

            secondary = strings(0, 1);

            description = ...
                "Piecewise-constant spatial pattern with sharp edge transitions.";

            reason = ...
                "2D Wavelet analysis localizes sharp spatial boundaries across multi-scale " + ...
                "directional subbands; 2D FFT shows high-frequency edge spectra; filtering " + ...
                "illustrates edge response.";

        % -----------------------------------------------------------
        % CHIRP
        % -----------------------------------------------------------
        case "Chirp"

            primary = [ ...
                "2D FFT"; ...
                "2D Wavelet"; ...
                "Image Filter" ...
            ];

            secondary = [ ...
                "2D Deconvolution" ...
            ];

            description = ...
                "Non-stationary spatial pattern with position-dependent spatial frequencies.";

            reason = ...
                "2D FFT characterizes total spatial bandwidth; 2D wavelets capture " + ...
                "spatial-frequency variation across scales; deconvolution demonstrates " + ...
                "spatial resolution recovery.";

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

    analysis.useFFT2          = any(allAnalyses == "2D FFT");
    analysis.useWavelet2D     = any(allAnalyses == "2D Wavelet");
    analysis.useFilter        = any(allAnalyses == "Image Filter");
    analysis.useConvolution   = any(allAnalyses == "2D Convolution");
    analysis.useDeconvolution = any(allAnalyses == "2D Deconvolution");

end
