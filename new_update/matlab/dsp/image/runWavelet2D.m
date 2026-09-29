function result = runWavelet2D(imageInput, waveletName, decompositionLevel)
% RUNWAVELET2D
% Two-dimensional discrete wavelet analysis for images.
%
% INPUTS:
%   imageInput         - 2-D grayscale image or RGB image
%   waveletName        - Wavelet name, e.g. "db4"
%   decompositionLevel - Number of decomposition levels
%
% OUTPUT:
%   result - Structure containing:
%       .input
%       .grayImage
%       .analysisImage
%       .waveletName
%       .decompositionLevel
%       .approximation
%       .horizontalDetails
%       .verticalDetails
%       .diagonalDetails
%       .detailEnergy
%       .energyRatio
%       .waveletEntropy
%       .reconstructedSignal
%       .reconstructionRMSE
%
% REQUIREMENT:
%   MATLAB Wavelet Toolbox

    %% ---------------------------------------------------------------
    % 1. INPUT VALIDATION
    % ---------------------------------------------------------------

    if nargin < 1
        error("runWavelet2D requires an image input.");
    end

    if isempty(imageInput)
        error("Input image cannot be empty.");
    end

    if nargin < 2 || isempty(waveletName)
        waveletName = "db4";
    end

    if nargin < 3 || isempty(decompositionLevel)
        decompositionLevel = 3;
    end

    if ~isnumeric(imageInput) && ~islogical(imageInput)
        error("Input must be a numeric or logical image.");
    end

    if ndims(imageInput) ~= 2 && ndims(imageInput) ~= 3
        error("Input must be a grayscale or RGB image.");
    end

    if any(~isfinite(double(imageInput(:))))
        error("Input image contains NaN or Inf values.");
    end

    if ~isscalar(decompositionLevel) || ...
            decompositionLevel < 1 || ...
            decompositionLevel ~= floor(decompositionLevel)

        error("Decomposition level must be a positive integer.");
    end

    %% ---------------------------------------------------------------
    % 2. PRESERVE ORIGINAL
    % ---------------------------------------------------------------

    originalImage = imageInput;

    %% ---------------------------------------------------------------
    % 3. CONVERT TO GRAYSCALE
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
    % 4. REMOVE DC / MEAN INTENSITY
    % ---------------------------------------------------------------

    imageAnalysis = ...
        imageGray - mean(imageGray(:));

    %% ---------------------------------------------------------------
    % 5. IMAGE SIZE
    % ---------------------------------------------------------------

    [rows, cols] = size(imageAnalysis);

    %% ---------------------------------------------------------------
    % 6. DETERMINE MAXIMUM DECOMPOSITION LEVEL
    % ---------------------------------------------------------------

    try

        maxLevel = wmaxlev( ...
            [rows cols], ...
            waveletName);

    catch ME

        error( ...
            "Unable to determine maximum wavelet level. " + ...
            "Check that Wavelet Toolbox is installed.\n%s", ...
            ME.message);

    end

    if maxLevel < 1
        error("Image is too small for wavelet decomposition.");
    end

    if decompositionLevel > maxLevel

        warning( ...
            "Requested level %d exceeds maximum level %d. " + ...
            "Using level %d.", ...
            decompositionLevel, ...
            maxLevel, ...
            maxLevel);

        decompositionLevel = maxLevel;

    end

    %% ---------------------------------------------------------------
    % 7. 2-D WAVELET DECOMPOSITION
    % ---------------------------------------------------------------

    [C, S] = wavedec2( ...
        imageAnalysis, ...
        decompositionLevel, ...
        waveletName);

    %% ---------------------------------------------------------------
    % 8. EXTRACT APPROXIMATION
    % ---------------------------------------------------------------

    approximation = appcoef2( ...
        C, ...
        S, ...
        waveletName, ...
        decompositionLevel);

    %% ---------------------------------------------------------------
    % 9. INITIALIZE DETAIL STORAGE
    % ---------------------------------------------------------------

    horizontalDetails = ...
        cell(1, decompositionLevel);

    verticalDetails = ...
        cell(1, decompositionLevel);

    diagonalDetails = ...
        cell(1, decompositionLevel);

    %% ---------------------------------------------------------------
    % 10. EXTRACT HORIZONTAL, VERTICAL AND DIAGONAL DETAILS
    % ---------------------------------------------------------------

    for k = 1:decompositionLevel

        [horizontalDetails{k}, ...
         verticalDetails{k}, ...
         diagonalDetails{k}] = ...
            detcoef2( ...
                "all", ...
                C, ...
                S, ...
                k);

    end

    %% ---------------------------------------------------------------
    % 11. CALCULATE ENERGY
    % ---------------------------------------------------------------

    approximationEnergy = ...
        sum(approximation(:).^2);

    horizontalEnergy = zeros(1, decompositionLevel);
    verticalEnergy = zeros(1, decompositionLevel);
    diagonalEnergy = zeros(1, decompositionLevel);

    for k = 1:decompositionLevel

        horizontalEnergy(k) = ...
            sum(horizontalDetails{k}(:).^2);

        verticalEnergy(k) = ...
            sum(verticalDetails{k}(:).^2);

        diagonalEnergy(k) = ...
            sum(diagonalDetails{k}(:).^2);

    end

    %% ---------------------------------------------------------------
    % 12. TOTAL ENERGY
    % ---------------------------------------------------------------

    totalEnergy = ...
        approximationEnergy + ...
        sum(horizontalEnergy) + ...
        sum(verticalEnergy) + ...
        sum(diagonalEnergy);

    %% ---------------------------------------------------------------
    % 13. ENERGY RATIOS
    % ---------------------------------------------------------------

    if totalEnergy > eps

        approximationEnergyRatio = ...
            approximationEnergy / totalEnergy;

        horizontalEnergyRatio = ...
            horizontalEnergy / totalEnergy;

        verticalEnergyRatio = ...
            verticalEnergy / totalEnergy;

        diagonalEnergyRatio = ...
            diagonalEnergy / totalEnergy;

    else

        approximationEnergyRatio = 0;

        horizontalEnergyRatio = ...
            zeros(1, decompositionLevel);

        verticalEnergyRatio = ...
            zeros(1, decompositionLevel);

        diagonalEnergyRatio = ...
            zeros(1, decompositionLevel);

    end

    %% ---------------------------------------------------------------
    % 14. COMBINED DETAIL ENERGY
    % ---------------------------------------------------------------

    detailEnergy = ...
        horizontalEnergy + ...
        verticalEnergy + ...
        diagonalEnergy;

    %% ---------------------------------------------------------------
    % 15. DOMINANT SCALE
    % ---------------------------------------------------------------

    if any(detailEnergy > 0)

        [~, dominantLevel] = ...
            max(detailEnergy);

    else

        dominantLevel = NaN;

    end

    %% ---------------------------------------------------------------
    % 16. DOMINANT ORIENTATION
    % ---------------------------------------------------------------

    totalHorizontal = sum(horizontalEnergy);
    totalVertical = sum(verticalEnergy);
    totalDiagonal = sum(diagonalEnergy);

    orientationEnergy = ...
        [totalHorizontal, ...
         totalVertical, ...
         totalDiagonal];

    [~, dominantOrientationIndex] = ...
        max(orientationEnergy);

    orientationNames = ...
        ["Horizontal", "Vertical", "Diagonal"];

    dominantOrientation = ...
        orientationNames(dominantOrientationIndex);

    %% ---------------------------------------------------------------
    % 17. WAVELET ENTROPY
    % ---------------------------------------------------------------

    energyDistribution = [ ...
        approximationEnergy, ...
        horizontalEnergy, ...
        verticalEnergy, ...
        diagonalEnergy ...
    ];

    energySum = sum(energyDistribution);

    if energySum > eps

        probability = ...
            energyDistribution / energySum;

        probability = ...
            probability(probability > 0);

        waveletEntropy = ...
            -sum(probability .* log2(probability));

    else

        waveletEntropy = 0;

    end

    %% ---------------------------------------------------------------
    % 18. RECONSTRUCT APPROXIMATION
    % ---------------------------------------------------------------

    reconstructedApproximation = ...
        wrcoef2( ...
            "a", ...
            C, ...
            S, ...
            waveletName, ...
            decompositionLevel);

    %% ---------------------------------------------------------------
    % 19. RECONSTRUCT DETAIL COMPONENTS
    % ---------------------------------------------------------------

    reconstructedHorizontal = ...
        cell(1, decompositionLevel);

    reconstructedVertical = ...
        cell(1, decompositionLevel);

    reconstructedDiagonal = ...
        cell(1, decompositionLevel);

    for k = 1:decompositionLevel

        reconstructedHorizontal{k} = ...
            wrcoef2( ...
                "h", ...
                C, ...
                S, ...
                waveletName, ...
                k);

        reconstructedVertical{k} = ...
            wrcoef2( ...
                "v", ...
                C, ...
                S, ...
                waveletName, ...
                k);

        reconstructedDiagonal{k} = ...
            wrcoef2( ...
                "d", ...
                C, ...
                S, ...
                waveletName, ...
                k);

    end

    %% ---------------------------------------------------------------
    % 20. RECONSTRUCT COMPLETE IMAGE
    % ---------------------------------------------------------------

    reconstructedImage = ...
        reconstructedApproximation;

    for k = 1:decompositionLevel

        reconstructedImage = ...
            reconstructedImage + ...
            reconstructedHorizontal{k} + ...
            reconstructedVertical{k} + ...
            reconstructedDiagonal{k};

    end

    %% ---------------------------------------------------------------
    % 21. RECONSTRUCTION ERROR
    % ---------------------------------------------------------------

    reconstructionError = ...
        imageAnalysis - reconstructedImage;

    reconstructionRMSE = ...
        sqrt(mean(reconstructionError(:).^2));

    reconstructionMAE = ...
        mean(abs(reconstructionError(:)));

    %% ---------------------------------------------------------------
    % 22. BUILD RESULT STRUCTURE
    % ---------------------------------------------------------------

    result = struct();

    result.input = originalImage;

    result.grayImage = imageGray;

    result.analysisImage = imageAnalysis;

    result.rows = rows;

    result.columns = cols;

    result.waveletName = waveletName;

    result.decompositionLevel = ...
        decompositionLevel;

    result.maximumLevel = maxLevel;

    result.coefficients = C;

    result.bookkeeping = S;

    result.approximation = approximation;

    result.horizontalDetails = ...
        horizontalDetails;

    result.verticalDetails = ...
        verticalDetails;

    result.diagonalDetails = ...
        diagonalDetails;

    result.approximationEnergy = ...
        approximationEnergy;

    result.horizontalEnergy = ...
        horizontalEnergy;

    result.verticalEnergy = ...
        verticalEnergy;

    result.diagonalEnergy = ...
        diagonalEnergy;

    result.detailEnergy = ...
        detailEnergy;

    result.totalEnergy = totalEnergy;

    result.approximationEnergyRatio = ...
        approximationEnergyRatio;

    result.horizontalEnergyRatio = ...
        horizontalEnergyRatio;

    result.verticalEnergyRatio = ...
        verticalEnergyRatio;

    result.diagonalEnergyRatio = ...
        diagonalEnergyRatio;

    result.dominantLevel = ...
        dominantLevel;

    result.dominantOrientation = ...
        dominantOrientation;

    result.waveletEntropy = ...
        waveletEntropy;

    result.reconstructedApproximation = ...
        reconstructedApproximation;

    result.reconstructedHorizontal = ...
        reconstructedHorizontal;

    result.reconstructedVertical = ...
        reconstructedVertical;

    result.reconstructedDiagonal = ...
        reconstructedDiagonal;

    result.reconstructedImage = ...
        reconstructedImage;

    result.reconstructionRMSE = ...
        reconstructionRMSE;

    result.reconstructionMAE = ...
        reconstructionMAE;

end
