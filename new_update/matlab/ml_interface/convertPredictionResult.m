function result = convertPredictionResult(pyResult)
%CONVERTPREDICTIONRESULT Convert Python prediction dictionary to MATLAB struct.
%
%   Standardizes the output across audio and image models:
%     result.class         : string (e.g., "chirp")
%     result.class_id      : double (e.g., 4)
%     result.confidence    : double (e.g., 0.964)
%     result.probabilities : struct with fields (impulse, sinusoidal, white_noise, step, chirp)

    result = struct();

    % Class name (string)
    try
        result.class = string(char(pyResult{'class'}));
    catch
        result.class = "unknown";
    end

    % Class ID (numeric)
    try
        result.class_id = double(pyResult{'class_id'});
    catch
        result.class_id = -1;
    end

    % Confidence score (numeric)
    try
        result.confidence = double(pyResult{'confidence'});
    catch
        result.confidence = 0.0;
    end

    % Per-class probabilities struct
    probStruct = struct();
    try
        pyProbs = pyResult{'probabilities'};
        keys = cell(py.list(pyProbs.keys()));
        for k = 1:length(keys)
            keyStr = char(keys{k});
            probVal = double(pyProbs{keyStr});
            probStruct.(keyStr) = probVal;
        end
    catch
    end
    result.probabilities = probStruct;
end
