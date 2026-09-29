function features = calculateFeatures(analysisResult, modality)
% CALCULATEFEATURES
% Extracts standardized numerical features from completed DSP analysis.
%
% INPUTS:
%   analysisResult - Output structure from analyzeAudio() or
%                    analyzeImage()
%
%   modality       - "audio" or "image"
%
% OUTPUT:
%   features - Structure containing:
%       .modality
%       .signalClass
%       .timeDomain
%       .frequencyDomain
%       .timeFrequency
%       .wavelet
%       .image
%       .filter
%       .system
%       .vector
%       .names
%       .count
%
% IMPORTANT:
%   This function does NOT rerun DSP algorithms.
%   It extracts features from already-computed results.

    %% ===============================================================
    % 1. INPUT VALIDATION
    % ===============================================================

    if nargin < 2
        error("calculateFeatures requires analysisResult and modality.");
    end

    if ~isstruct(analysisResult)
        error("analysisResult must be a structure.");
    end

    modality = lower(string(modality));

    if ~any(modality == ["audio", "image"])
        error("Modality must be audio or image.");
    end

    %% ===============================================================
    % 2. INITIALIZE OUTPUT
    % ===============================================================

    features = struct();

    if isfield(analysisResult, "signalClass")
        features.signalClass = analysisResult.signalClass;
    else
        features.signalClass = "Unknown";
    end

    features.modality        = modality;
    features.timeDomain      = struct();
    features.frequencyDomain = struct();
    features.timeFrequency   = struct();
    features.wavelet         = struct();
    features.image           = struct();
    features.filter          = struct();
    features.system          = struct();

    featureValues = [];
    featureNames  = strings(0, 1);

    %% ===============================================================
    % 3. AUDIO FEATURE EXTRACTION
    % ===============================================================

    if modality == "audio"

        if ~isfield(analysisResult, "analysisSignal")
            error("Audio analysis result lacks analysisSignal.");
        end

        x = double(analysisResult.analysisSignal(:));

        %% -----------------------------------------------------------
        % TIME-DOMAIN FEATURES
        % -----------------------------------------------------------

        meanValue     = mean(x);
        stdValue      = std(x);
        rmsValue      = sqrt(mean(x.^2));
        peakValue     = max(abs(x));
        energyValue   = sum(x.^2);

        if rmsValue > eps
            peakToRMS = peakValue / rmsValue;
        else
            peakToRMS = 0;
        end

        zeroCrossings    = sum(x(1:end-1) .* x(2:end) < 0);
        zeroCrossingRate = zeroCrossings / max(length(x) - 1, 1);

        features.timeDomain.mean               = meanValue;
        features.timeDomain.standardDeviation  = stdValue;
        features.timeDomain.rms                = rmsValue;
        features.timeDomain.peak               = peakValue;
        features.timeDomain.energy             = energyValue;
        features.timeDomain.peakToRMS          = peakToRMS;
        features.timeDomain.zeroCrossingRate   = zeroCrossingRate;

        featureValues = [featureValues; meanValue; stdValue; rmsValue;
                         peakValue; energyValue; peakToRMS; zeroCrossingRate];

        featureNames = [featureNames;
            "time_mean";
            "time_std";
            "time_rms";
            "time_peak";
            "time_energy";
            "time_peak_to_rms";
            "time_zero_crossing_rate"];

        %% -----------------------------------------------------------
        % FFT FEATURES
        % -----------------------------------------------------------

        if isfield(analysisResult, "fft") && ~isempty(analysisResult.fft)

            fftResult = analysisResult.fft;

            fftFieldMap = {
                "peakFrequency",    "dominantFrequency",    "fft_dominant_frequency";
                "spectralCentroid", "spectralCentroid",     "fft_spectral_centroid";
                "spectralBandwidth","spectralBandwidth",    "fft_spectral_bandwidth";
                "spectralFlatness", "spectralFlatness",     "fft_spectral_flatness";
                "bandwidth3dB",     "bandwidth3dB",         "fft_3db_bandwidth"
            };

            for k = 1:size(fftFieldMap, 1)

                srcField  = fftFieldMap{k, 1};
                destField = fftFieldMap{k, 2};
                featName  = fftFieldMap{k, 3};

                if isfield(fftResult, srcField)
                    value = fftResult.(srcField);
                    if isscalar(value) && isfinite(value)
                        features.frequencyDomain.(destField) = value;
                        featureValues(end+1) = value;
                        featureNames(end+1)  = featName;
                    end
                end

            end

        end

        %% -----------------------------------------------------------
        % STFT FEATURES
        % -----------------------------------------------------------

        if isfield(analysisResult, "stft") && ~isempty(analysisResult.stft)

            stftResult = analysisResult.stft;

            if isfield(stftResult, "dominantFrequency")

                vals  = stftResult.dominantFrequency(:);
                valid = isfinite(vals);
                vals  = vals(valid);

                if ~isempty(vals)
                    meanFreq  = mean(vals);
                    freqVar   = std(vals);

                    features.timeFrequency.meanDominantFrequency = meanFreq;
                    features.timeFrequency.frequencyVariation    = freqVar;

                    featureValues(end+1) = meanFreq;
                    featureValues(end+1) = freqVar;
                    featureNames(end+1)  = "stft_mean_dominant_frequency";
                    featureNames(end+1)  = "stft_frequency_variation";
                end
            end

            stftFieldMap = {
                "spectralCentroid",  "meanSpectralCentroid", "stft_mean_spectral_centroid";
                "spectralBandwidth", "meanBandwidth",        "stft_mean_bandwidth";
                "totalEnergy",       "energy",               "stft_energy"
            };

            for k = 1:size(stftFieldMap, 1)

                srcField  = stftFieldMap{k, 1};
                destField = stftFieldMap{k, 2};
                featName  = stftFieldMap{k, 3};

                if isfield(stftResult, srcField)

                    vals = stftResult.(srcField)(:);
                    vals = vals(isfinite(vals));

                    if ~isempty(vals)
                        value = mean(vals);
                        features.timeFrequency.(destField) = value;
                        featureValues(end+1) = value;
                        featureNames(end+1)  = featName;
                    end

                end

            end

        end

        %% -----------------------------------------------------------
        % WAVELET FEATURES (1-D)
        % -----------------------------------------------------------

        if isfield(analysisResult, "wavelet") && ...
                ~isempty(analysisResult.wavelet)

            waveletResult = analysisResult.wavelet;

            if isfield(waveletResult, "approximationEnergy")
                value = waveletResult.approximationEnergy;
                if isscalar(value) && isfinite(value)
                    features.wavelet.approximationEnergy = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet_approximation_energy";
                end
            end

            if isfield(waveletResult, "detailEnergy")
                detailEnergy = waveletResult.detailEnergy(:);
                for k = 1:length(detailEnergy)
                    if isfinite(detailEnergy(k))
                        features.wavelet.("detailEnergy" + string(k)) = detailEnergy(k);
                        featureValues(end+1) = detailEnergy(k);
                        featureNames(end+1)  = "wavelet_detail_energy_" + string(k);
                    end
                end
            end

            if isfield(waveletResult, "detailEnergyRatio")
                ratios = waveletResult.detailEnergyRatio(:);
                for k = 1:length(ratios)
                    if isfinite(ratios(k))
                        featureValues(end+1) = ratios(k);
                        featureNames(end+1)  = "wavelet_detail_ratio_" + string(k);
                    end
                end
            end

            if isfield(waveletResult, "waveletEntropy")
                value = waveletResult.waveletEntropy;
                if isscalar(value) && isfinite(value)
                    features.wavelet.entropy = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet_entropy";
                end
            end

            if isfield(waveletResult, "dominantLevel")
                value = waveletResult.dominantLevel;
                if isscalar(value) && isfinite(value)
                    features.wavelet.dominantLevel = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet_dominant_level";
                end
            end

        end

    %% ===============================================================
    % 4. IMAGE FEATURE EXTRACTION
    % ===============================================================

    else

        if ~isfield(analysisResult, "analysisImage")
            error("Image analysis result lacks analysisImage.");
        end

        I = double(analysisResult.analysisImage);

        %% -----------------------------------------------------------
        % GRAYSCALE CONVERSION FOR GLOBAL FEATURES
        % -----------------------------------------------------------

        if ndims(I) == 3
            Igray = 0.2989 * I(:,:,1) + ...
                    0.5870 * I(:,:,2) + ...
                    0.1140 * I(:,:,3);
        else
            Igray = I;
        end

        %% -----------------------------------------------------------
        % SPATIAL FEATURES
        % -----------------------------------------------------------

        meanValue    = mean(Igray(:));
        stdValue     = std(Igray(:));
        rmsValue     = sqrt(mean(Igray(:).^2));
        peakValue    = max(abs(Igray(:)));
        minValue     = min(Igray(:));
        maxValue     = max(Igray(:));
        dynamicRange = maxValue - minValue;
        energyValue  = sum(Igray(:).^2);

        features.image.meanIntensity   = meanValue;
        features.image.standardDeviation = stdValue;
        features.image.rms             = rmsValue;
        features.image.peak            = peakValue;
        features.image.minimum         = minValue;
        features.image.maximum         = maxValue;
        features.image.dynamicRange    = dynamicRange;
        features.image.energy          = energyValue;

        featureValues = [featureValues; meanValue; stdValue; rmsValue;
                         peakValue; dynamicRange; energyValue];

        featureNames = [featureNames;
            "image_mean_intensity";
            "image_std";
            "image_rms";
            "image_peak";
            "image_dynamic_range";
            "image_energy"];

        %% -----------------------------------------------------------
        % 2-D FFT FEATURES
        % -----------------------------------------------------------

        if isfield(analysisResult, "fft2") && ~isempty(analysisResult.fft2)

            fftResult = analysisResult.fft2;

            fft2FieldMap = {
                "dominantSpatialFrequency", "dominantSpatialFrequency", "fft2_dominant_spatial_frequency";
                "spectralCentroidX",        "spectralCentroidX",        "fft2_spectral_centroid_x";
                "spectralCentroidY",        "spectralCentroidY",        "fft2_spectral_centroid_y";
                "spectralBandwidth",        "spectralBandwidth",        "fft2_spectral_bandwidth";
                "spectralFlatness",         "spectralFlatness",         "fft2_spectral_flatness";
                "totalEnergy",              "totalEnergy",              "fft2_total_energy"
            };

            for k = 1:size(fft2FieldMap, 1)
                srcField  = fft2FieldMap{k, 1};
                destField = fft2FieldMap{k, 2};
                featName  = fft2FieldMap{k, 3};
                if isfield(fftResult, srcField)
                    value = fftResult.(srcField);
                    if isscalar(value) && isfinite(value)
                        features.frequencyDomain.(destField) = value;
                        featureValues(end+1) = value;
                        featureNames(end+1)  = featName;
                    end
                end
            end

        end

        %% -----------------------------------------------------------
        % 2-D WAVELET FEATURES
        % -----------------------------------------------------------

        if isfield(analysisResult, "wavelet2D") && ...
                ~isempty(analysisResult.wavelet2D)

            waveletResult = analysisResult.wavelet2D;

            % Approximation energy
            if isfield(waveletResult, "approximationEnergy")
                value = waveletResult.approximationEnergy;
                if isscalar(value) && isfinite(value)
                    features.wavelet.approximationEnergy = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet2d_approximation_energy";
                end
            end

            % Per-level sub-band energies
            subbandFieldMap = {
                "horizontalEnergy", "wavelet2d_horizontal_energy";
                "verticalEnergy",   "wavelet2d_vertical_energy";
                "diagonalEnergy",   "wavelet2d_diagonal_energy"
            };

            for r = 1:size(subbandFieldMap, 1)
                srcField = subbandFieldMap{r, 1};
                prefix   = subbandFieldMap{r, 2};
                if isfield(waveletResult, srcField)
                    vals = waveletResult.(srcField)(:);
                    for k = 1:length(vals)
                        if isfinite(vals(k))
                            featureValues(end+1) = vals(k);
                            featureNames(end+1)  = prefix + "_" + string(k);
                        end
                    end
                end
            end

            % Per-level energy ratios
            ratioFieldMap = {
                "horizontalEnergyRatio", "wavelet2d_horizontal_ratio";
                "verticalEnergyRatio",   "wavelet2d_vertical_ratio";
                "diagonalEnergyRatio",   "wavelet2d_diagonal_ratio"
            };

            for r = 1:size(ratioFieldMap, 1)
                srcField = ratioFieldMap{r, 1};
                prefix   = ratioFieldMap{r, 2};
                if isfield(waveletResult, srcField)
                    vals = waveletResult.(srcField)(:);
                    for k = 1:length(vals)
                        if isfinite(vals(k))
                            featureValues(end+1) = vals(k);
                            featureNames(end+1)  = prefix + "_" + string(k);
                        end
                    end
                end
            end

            % Entropy
            if isfield(waveletResult, "waveletEntropy")
                value = waveletResult.waveletEntropy;
                if isscalar(value) && isfinite(value)
                    features.wavelet.entropy = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet2d_entropy";
                end
            end

            % Dominant level
            if isfield(waveletResult, "dominantLevel")
                value = waveletResult.dominantLevel;
                if isscalar(value) && isfinite(value)
                    features.wavelet.dominantLevel = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = "wavelet2d_dominant_level";
                end
            end

        end

    end

    %% ===============================================================
    % 5. FILTER FEATURES (AUDIO + IMAGE)
    % ===============================================================

    filterResultField = "";

    if modality == "audio" && isfield(analysisResult, "fir") && ...
            ~isempty(analysisResult.fir)
        filterResultField = "fir";
    elseif modality == "image" && isfield(analysisResult, "filter") && ...
            ~isempty(analysisResult.filter)
        filterResultField = "filter";
    end

    if filterResultField ~= ""

        filterResult = analysisResult.(filterResultField);

        filterFieldMap = {
            "inputEnergy",             "filter_input_energy";
            "outputEnergy",            "filter_output_energy";
            "differenceEnergy",        "filter_difference_energy";
            "outputStandardDeviation", "filter_output_std"
        };

        for k = 1:size(filterFieldMap, 1)
            srcField = filterFieldMap{k, 1};
            featName = filterFieldMap{k, 2};
            if isfield(filterResult, srcField)
                value = filterResult.(srcField);
                if isscalar(value) && isfinite(value)
                    features.filter.(srcField) = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = featName;
                end
            end
        end

    end

    %% ===============================================================
    % 6. SYSTEM / DECONVOLUTION FEATURES (AUDIO + IMAGE)
    % ===============================================================

    deconvField = "";

    if isfield(analysisResult, "deconvolution") && ...
            ~isempty(analysisResult.deconvolution)
        deconvField = "deconvolution";
    end

    if deconvField ~= ""

        deconvResult = analysisResult.(deconvField);

        deconvFieldMap = {
            "rmse",        "deconvolution_rmse";
            "mae",         "deconvolution_mae";
            "correlation", "deconvolution_correlation";
            "psnr",        "deconvolution_psnr"
        };

        for k = 1:size(deconvFieldMap, 1)
            srcField = deconvFieldMap{k, 1};
            featName = deconvFieldMap{k, 2};
            if isfield(deconvResult, srcField)
                value = deconvResult.(srcField);
                if isscalar(value) && isfinite(value)
                    features.system.(srcField) = value;
                    featureValues(end+1) = value;
                    featureNames(end+1)  = featName;
                end
            end
        end

    end

    %% ===============================================================
    % 7. REMOVE NON-FINITE FEATURES
    % ===============================================================

    featureValues = double(featureValues(:));
    valid         = isfinite(featureValues);
    featureValues = featureValues(valid);
    featureNames  = featureNames(valid);

    %% ===============================================================
    % 8. FINAL FEATURE VECTOR
    % ===============================================================

    features.vector = featureValues;
    features.names  = featureNames;
    features.count  = length(featureValues);

end
