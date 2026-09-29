function result = classifyImage(processedImage)
%CLASSIFYIMAGE Classifies preprocessed image using persistent Python ML model.
%
%   Inputs:
%       processedImage - 128x128 grayscale normalized image matrix in [0, 1]
%
%   Outputs:
%       result - Struct with fields (.class, .class_id, .confidence, .probabilities)

    % Get persistent image predictor
    predictor = initializePythonML('image');

    if isempty(predictor)
        error('SPANDHAN:ImageModelNotReady', ...
            'Image classifier model is not yet trained or loaded.');
    end

    % Convert MATLAB 2D image matrix directly to NumPy array
    pyImage = py.numpy.array(double(processedImage));

    % Call prediction method on MATLAB-preprocessed 128x128 image
    pyResult = predictor.predict_processed_image(pyImage);

    % Convert Python dictionary to standard MATLAB struct
    result = convertPredictionResult(pyResult);
end
