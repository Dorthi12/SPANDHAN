function processed = preprocessImage(img)
    % PREPROCESSIMAGE Converts to grayscale, resizes to 128x128, and normalizes to [0, 1].
    % Inputs:
    %   img       - Raw input image (any size, RGB or grayscale)
    % Outputs:
    %   processed - Preprocessed 128x128 grayscale image matrix in [0, 1]

    % 1. Convert to grayscale if 3 channels (RGB) or 4 channels (RGBA)
    if size(img, 3) == 3
        img = rgb2gray(img);
    elseif size(img, 3) == 4
        img = rgb2gray(img(:, :, 1:3));
    end

    % 2. Resize to canonical 128x128 if dimensions differ
    if size(img, 1) ~= 128 || size(img, 2) ~= 128
        img = imresize(img, [128, 128]);
    end

    % 3. Normalize intensity range to [0, 1]
    processed = normalizeImage(double(img));
end
