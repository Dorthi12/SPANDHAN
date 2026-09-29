function img = loadImage(filepath)
    % LOADIMAGE Loads an image file into MATLAB array.
    % Inputs:
    %   filepath - Path to image file (.png, .jpg, etc.)
    % Outputs:
    %   img      - Image array

    if ~exist(filepath, 'file')
        error('File not found: %s', filepath);
    end
    img = imread(filepath);
end
