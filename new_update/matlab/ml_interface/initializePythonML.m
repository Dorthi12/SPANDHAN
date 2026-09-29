function [audioPredictor, imagePredictor] = initializePythonML(modelType)
%INITIALIZEPYTHONML Initialize persistent Python ML predictors in MATLAB.
%
%   Initializes Python environment and loads the trained SPANDHAN models
%   once for repeated, fast inference without restarting Python.
%
%   Usage:
%       audioPredictor = initializePythonML('audio');
%       imagePredictor = initializePythonML('image');
%       [audioPred, imgPred] = initializePythonML();

    persistent persistentAudioPredictor persistentImagePredictor

    if nargin < 1
        modelType = 'all';
    end

    % Locate project root from this file
    mfilePath = mfilename('fullpath');
    mlInterfaceDir = fileparts(mfilePath);
    matlabDir = fileparts(mlInterfaceDir);
    projectRoot = fileparts(matlabDir);

    % Configure Python sys.path so 'ml' and 'python' packages are importable
    pySys = py.importlib.import_module('sys');
    pythonPath = pySys.path;
    projectRootPy = py.str(projectRoot);
    if ~pythonPath.count(projectRootPy)
        pythonPath.insert(int32(0), projectRootPy);
    end

    % Initialize AudioPredictor if requested
    if ismember(lower(modelType), {'audio', 'all'})
        if isempty(persistentAudioPredictor)
            audioModelPath = fullfile(projectRoot, 'models', 'audio', 'audio_signal_classifier.pkl');
            if ~exist(audioModelPath, 'file')
                error('SPANDHAN:ModelNotFound', ...
                    'Trained audio model not found at: %s\nPlease run training first.', audioModelPath);
            end
            pyPredictModule = py.importlib.import_module('ml.audio.inference.predict');
            persistentAudioPredictor = pyPredictModule.AudioPredictor(audioModelPath);
        end
        audioPredictor = persistentAudioPredictor;
    else
        audioPredictor = [];
    end

    % Initialize ImagePredictor if requested
    if ismember(lower(modelType), {'image', 'all'})
        if isempty(persistentImagePredictor)
            imageModelPath = fullfile(projectRoot, 'models', 'image', 'image_signal_classifier.pkl');
            if ~exist(imageModelPath, 'file')
                altPath = fullfile(projectRoot, 'models', 'image_signal_classifier.pkl');
                if exist(altPath, 'file')
                    imageModelPath = altPath;
                end
            end
            if exist(imageModelPath, 'file')
                pyPredictModule = py.importlib.import_module('ml.image.inference.predict');
                persistentImagePredictor = pyPredictModule.get_image_predictor(imageModelPath);
            else
                persistentImagePredictor = [];
            end
        end
        imagePredictor = persistentImagePredictor;
    else
        imagePredictor = [];
    end
end
