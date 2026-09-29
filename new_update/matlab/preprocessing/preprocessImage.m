function processed = preprocessImage(img)
    % PREPROCESSIMAGE Resizes, converts to grayscale if needed, and normalizes image.
    % Inputs:
    %   img       - Raw input image
    % Outputs:
    %   processed - Preprocessed image matrix

    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    processed = normalizeImage(double(img));
end
