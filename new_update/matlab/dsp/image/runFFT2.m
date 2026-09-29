function result = runFFT2(imageInput)
% RUNFFT2
% Two-dimensional Fourier Transform analysis for images.
%
% INPUT:
%   imageInput - 2-D grayscale image or RGB image
%
% OUTPUT:
%   result - Structure containing:
%       .input
%       .grayImage
%       .fft
%       .fftShifted
%       .magnitude
%       .magnitudeDB
%       .phase
%       .powerSpectrum
%       .normalizedMagnitude
%       .frequencyX
%       .frequencyY
%       .frequencyRadius
%       .spectralCentroidX
%       .spectralCentroidY
%       .spectralBandwidth
%       .totalEnergy
%       .dominantFrequencyX
%       .dominantFrequencyY
%       .dominantSpatialFrequency
%
% NOTE:
%   For image analysis, spatial frequency is expressed in
%   cycles/pixel because no physical pixel spacing is assumed.

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 1
        error("runFFT2 requires an image input.");
    end

    if isempty(imageInput)
        error("Input image cannot be empty.");
    end

    if ~isnumeric(imageInput) && ~islogical(imageInput)
        error("Input must be a numeric or logical image.");
    end

    if ndims(imageInput) ~= 2 && ndims(imageInput) ~= 3
        error("Input must be a 2-D grayscale or RGB image.");
    end

    if any(~isfinite(double(imageInput(:))))
        error("Input image contains NaN or Inf values.");
    end

    %% ---------------------------------------------------------------
    % 2. PRESERVE ORIGINAL IMAGE
    % ---------------------------------------------------------------

    originalImage = imageInput;

    %% ---------------------------------------------------------------
    % 3. CONVERT RGB TO GRAYSCALE
    % ---------------------------------------------------------------

    if ndims(imageInput) == 3

        if size(imageInput, 3) ~= 3
            error("RGB image must have exactly 3 channels.");
        end

        imageGray = im2double(rgb2gray(imageInput));

    else

        imageGray = im2double(imageInput);

    end

    %% ---------------------------------------------------------------
    % 4. REMOVE DC COMPONENT
    % ---------------------------------------------------------------

    % Mean intensity corresponds to the zero-frequency/DC component.

    imageAnalysis = imageGray - mean(imageGray(:));

    %% ---------------------------------------------------------------
    % 5. IMAGE DIMENSIONS
    % ---------------------------------------------------------------

    [rows, cols] = size(imageAnalysis);

    %% ---------------------------------------------------------------
    % 6. 2-D FOURIER TRANSFORM
    % ---------------------------------------------------------------

    F = fft2(imageAnalysis);

    %% ---------------------------------------------------------------
    % 7. SHIFT ZERO FREQUENCY TO CENTER
    % ---------------------------------------------------------------

    Fshift = fftshift(F);

    %% ---------------------------------------------------------------
    % 8. MAGNITUDE
    % ---------------------------------------------------------------

    magnitude = abs(Fshift);

    % Normalize magnitude for visualization.

    if max(magnitude(:)) > 0

        normalizedMagnitude = ...
            magnitude / max(magnitude(:));

    else

        normalizedMagnitude = zeros(size(magnitude));

    end

    %% ---------------------------------------------------------------
    % 9. LOG MAGNITUDE SPECTRUM
    % ---------------------------------------------------------------

    % Log scaling makes weak frequency components visible.

    magnitudeDB = 20 * log10( ...
        normalizedMagnitude + eps);

    %% ---------------------------------------------------------------
    % 10. PHASE
    % ---------------------------------------------------------------

    phase = angle(Fshift);

    %% ---------------------------------------------------------------
    % 11. POWER SPECTRUM
    % ---------------------------------------------------------------

    powerSpectrum = abs(Fshift).^2;

    %% ---------------------------------------------------------------
    % 12. SPATIAL FREQUENCY AXES
    % ---------------------------------------------------------------

    % Frequency units:
    % cycles/pixel

    fx = (-floor(cols/2):ceil(cols/2)-1) / cols;

    fy = (-floor(rows/2):ceil(rows/2)-1) / rows;

    [frequencyX, frequencyY] = meshgrid(fx, fy);

    %% ---------------------------------------------------------------
    % 13. RADIAL SPATIAL FREQUENCY
    % ---------------------------------------------------------------

    frequencyRadius = sqrt( ...
        frequencyX.^2 + frequencyY.^2);

    %% ---------------------------------------------------------------
    % 14. TOTAL SPECTRAL ENERGY
    % ---------------------------------------------------------------

    totalEnergy = sum(powerSpectrum(:));

    %% ---------------------------------------------------------------
    % 15. SPECTRAL CENTROID
    % ---------------------------------------------------------------

    if totalEnergy > eps

        spectralCentroidX = ...
            sum(abs(frequencyX(:)) .* powerSpectrum(:)) ...
            / totalEnergy;

        spectralCentroidY = ...
            sum(abs(frequencyY(:)) .* powerSpectrum(:)) ...
            / totalEnergy;

        spectralCentroidRadial = ...
            sum(frequencyRadius(:) .* powerSpectrum(:)) ...
            / totalEnergy;

    else

        spectralCentroidX = 0;
        spectralCentroidY = 0;
        spectralCentroidRadial = 0;

    end

    %% ---------------------------------------------------------------
    % 16. SPECTRAL BANDWIDTH
    % ---------------------------------------------------------------

    if totalEnergy > eps

        deviationX = ...
            frequencyX - spectralCentroidX;

        deviationY = ...
            frequencyY - spectralCentroidY;

        spectralBandwidth = sqrt( ...
            sum( ...
                (deviationX(:).^2 + deviationY(:).^2) ...
                .* powerSpectrum(:) ...
            ) / totalEnergy ...
        );

    else

        spectralBandwidth = 0;

    end

    %% ---------------------------------------------------------------
    % 17. FIND DOMINANT SPATIAL FREQUENCY
    % ---------------------------------------------------------------

    % Ignore the center/DC component.

    centerRow = floor(rows/2) + 1;
    centerCol = floor(cols/2) + 1;

    magnitudeForPeak = magnitude;

    magnitudeForPeak(centerRow, centerCol) = 0;

    [~, peakIndex] = max(magnitudeForPeak(:));

    [peakRow, peakCol] = ...
        ind2sub(size(magnitudeForPeak), peakIndex);

    dominantFrequencyX = frequencyX(peakRow, peakCol);

    dominantFrequencyY = frequencyY(peakRow, peakCol);

    dominantSpatialFrequency = ...
        sqrt( ...
            dominantFrequencyX^2 + ...
            dominantFrequencyY^2);

    %% ---------------------------------------------------------------
    % 18. SPECTRAL FLATNESS
    % ---------------------------------------------------------------

    positivePower = powerSpectrum(:);
    positivePower = positivePower(positivePower > 0);

    if ~isempty(positivePower)

        geometricMean = exp( ...
            mean(log(positivePower)));

        arithmeticMean = mean(positivePower);

        spectralFlatness = ...
            geometricMean / max(arithmeticMean, eps);

    else

        spectralFlatness = 0;

    end

    %% ---------------------------------------------------------------
    % 19. BUILD OUTPUT STRUCTURE
    % ---------------------------------------------------------------

    result = struct();

    result.input = originalImage;

    result.grayImage = imageGray;

    result.analysisImage = imageAnalysis;

    result.rows = rows;
    result.columns = cols;

    result.fft = F;

    result.fftShifted = Fshift;

    result.magnitude = magnitude;

    result.normalizedMagnitude = normalizedMagnitude;

    result.magnitudeDB = magnitudeDB;

    result.phase = phase;

    result.powerSpectrum = powerSpectrum;

    result.frequencyX = frequencyX;

    result.frequencyY = frequencyY;

    result.frequencyRadius = frequencyRadius;

    result.spectralCentroidX = spectralCentroidX;

    result.spectralCentroidY = spectralCentroidY;

    result.spectralCentroidRadial = ...
        spectralCentroidRadial;

    result.spectralBandwidth = ...
        spectralBandwidth;

    result.spectralFlatness = ...
        spectralFlatness;

    result.dominantFrequencyX = ...
        dominantFrequencyX;

    result.dominantFrequencyY = ...
        dominantFrequencyY;

    result.dominantSpatialFrequency = ...
        dominantSpatialFrequency;

    result.totalEnergy = totalEnergy;

end
