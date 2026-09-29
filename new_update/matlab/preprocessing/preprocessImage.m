function [processed, info] = preprocessImage(img)
    % PREPROCESSIMAGE Converts to grayscale, resizes to 128x128, and normalizes to [0, 1].
    % Inputs:
    %   img       - Raw input image (any size, RGB or grayscale)
    % Outputs:
    %   processed - Preprocessed 128x128 grayscale image matrix in [0, 1]
    %   info      - Struct containing preprocessing metadata

    origSize = size(img);
    origClass = class(img);

    % 1. Convert to grayscale if 3 channels (RGB) or 4 channels (RGBA)
    wasRgb = false;
    if size(img, 3) == 3
        img = rgb2gray(img);
        wasRgb = true;
    elseif size(img, 3) == 4
        img = rgb2gray(img(:, :, 1:3));
        wasRgb = true;
    end

    % 2. Resize to canonical 128x128 if dimensions differ
    wasResized = false;
    if size(img, 1) ~= 128 || size(img, 2) ~= 128
        img = imresize(img, [128, 128]);
        wasResized = true;
    end

    % 3. Normalize intensity range to [0, 1]
    processed = normalizeImage(double(img));

    % 4. Build info struct
    info = struct();
    info.originalSize = origSize;
    info.originalClass = origClass;
    info.wasGrayscaleConverted = wasRgb;
    info.wasResized = wasResized;
    info.processedSize = size(processed);
    info.processedHeight = size(processed, 1);
    info.processedWidth = size(processed, 2);
    info.processedMin = min(processed(:));
    info.processedMax = max(processed(:));
end
