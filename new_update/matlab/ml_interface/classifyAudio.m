function result = classifyAudio(processedSignal, fs)
%CLASSIFYAUDIO Classifies preprocessed audio signal using persistent Python ML model.
%
%   Inputs:
%       processedSignal - Preprocessed mono audio vector (e.g., 32000x1 from preprocessAudio)
%       fs              - Sampling rate in Hz (default: 16000)
%
%   Outputs:
%       result - Struct with fields:
%                  .class         : detected signal class string ("impulse", "sinusoidal", etc.)
%                  .class_id      : 0 to 4 integer ID
%                  .confidence    : confidence score [0, 1]
%                  .probabilities : struct of per-class probabilities

    if nargin < 2
        fs = 16000;
    end

    % Ensure 1D row vector for numpy array conversion
    flatSignal = double(processedSignal(:))';

    % Get persistent audio predictor (loads .pkl once)
    predictor = initializePythonML('audio');

    % Convert MATLAB vector directly to NumPy array without temporary WAV
    pySignal = py.numpy.array(flatSignal);

    % Call predict_signal on the loaded predictor
    pyResult = predictor.predict_signal(pySignal, int32(fs));

    % Convert Python dictionary to standard MATLAB struct
    result = convertPredictionResult(pyResult);
end
