function startSPANDHAN()
%STARTSPANDHAN  Master application launcher for the SPANDHAN system.
%
%   Bootstraps the entire SPANDHAN application from a single entry point:
%
%       1. Determines project root dynamically (does not depend on pwd)
%       2. Configures MATLAB paths from the existing project layout
%       3. Verifies the MATLAB-embedded Python interpreter
%       4. Verifies required Python runtime packages
%       5. Verifies trained model files
%       6. Initialises the MATLAB<->Python ML interface (initializePythonML)
%       7. Launches the PySide6 UI as a separate OS process
%
%   USAGE:
%       startSPANDHAN          % from any MATLAB working directory
%
%   All initialisation logic delegates to existing SPANDHAN modules.
%   This file is intentionally free of DSP, ML training, and GUI widget code.
%
%   See also: initializePythonML, runAudioPipeline, runImagePipeline

    % ------------------------------------------------------------------ %
    % 0.  Startup banner                                                   %
    % ------------------------------------------------------------------ %
    printStartupBanner();

    % ------------------------------------------------------------------ %
    % 1.  MATLAB environment                                               %
    % ------------------------------------------------------------------ %
    fprintf('[1/6] Initializing MATLAB environment...\n');

    projectRoot = getProjectRoot();
    fprintf('      Project root      : %s\n', projectRoot);

    setupMatlabPaths(projectRoot);

    % ------------------------------------------------------------------ %
    % 2.  Python interpreter                                               %
    % ------------------------------------------------------------------ %
    fprintf('\n[2/6] Checking Python environment...\n');
    pyExe = checkPython();

    % ------------------------------------------------------------------ %
    % 3.  Python package dependencies                                      %
    % ------------------------------------------------------------------ %
    fprintf('\n[3/6] Checking dependencies...\n');
    checkDependencies(pyExe);

    % ------------------------------------------------------------------ %
    % 4.  Trained model files                                              %
    % ------------------------------------------------------------------ %
    fprintf('\n[4/6] Checking trained models...\n');
    checkModels(projectRoot);

    % ------------------------------------------------------------------ %
    % 5.  MATLAB <-> Python ML interface                                   %
    % ------------------------------------------------------------------ %
    fprintf('\n[5/6] Initializing ML interface...\n');
    initializeML(projectRoot);

    % ------------------------------------------------------------------ %
    % 6.  PySide6 UI                                                       %
    % ------------------------------------------------------------------ %
    fprintf('\n[6/6] Starting SPANDHAN UI...\n');
    launchUI(projectRoot, pyExe);

    % ------------------------------------------------------------------ %
    % Complete                                                             %
    % ------------------------------------------------------------------ %
    printCompleteBanner();

end


% ======================================================================= %
%  LOCAL HELPER FUNCTIONS                                                  %
% ======================================================================= %

% ----------------------------------------------------------------------- %
function printStartupBanner()
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('                     SPANDHAN\n');
    fprintf('        SIGNAL PROCESSING & ANALYSIS SYSTEM\n');
    fprintf('============================================================\n');
    fprintf('\n');
end

% ----------------------------------------------------------------------- %
function printCompleteBanner()
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('          SPANDHAN INITIALIZATION COMPLETE\n');
    fprintf('============================================================\n');
    fprintf('\nLaunching graphical interface...\n\n');
end

% ----------------------------------------------------------------------- %
function root = getProjectRoot()
%GETPROJECTROOT  Returns the absolute path of new_update/ dynamically.
%
%   Uses the location of THIS file (startSPANDHAN.m), which lives at:
%       <projectRoot>/startSPANDHAN.m
%   so projectRoot = fileparts of this file's full path.

    thisFile = mfilename('fullpath');   % e.g. C:\...\new_update\startSPANDHAN
    root     = fileparts(thisFile);     % strips filename → new_update\
end

% ----------------------------------------------------------------------- %
function setupMatlabPaths(projectRoot)
%SETUPMATLABPATHS  Adds SPANDHAN MATLAB source directories to the path.
%
%   The canonical MATLAB source tree lives under:
%       <projectRoot>/matlab/
%
%   The subdirectory  matlab/analysis/  is known to potentially shadow
%   functions in  matlab/dsp/analysis/ .  This function checks for it
%   and warns instead of silently including it.

    matlabRoot = fullfile(projectRoot, 'matlab');

    if ~isfolder(matlabRoot)
        startupError( ...
            'MATLAB source directory not found.', ...
            sprintf('Expected: %s', matlabRoot), ...
            'Verify the project layout has not been restructured.');
    end

    % ---- Check for the known legacy conflict directory ----------------
    legacyAnalysisDir = fullfile(matlabRoot, 'analysis');
    if isfolder(legacyAnalysisDir)
        fprintf('      [WARNING] Legacy directory detected:\n');
        fprintf('                %s\n', legacyAnalysisDir);
        fprintf('      This may conflict with matlab/dsp/analysis/.\n');
        fprintf('      Adding MATLAB paths WITHOUT the legacy directory.\n');
        addMatlabPathsSelectively(matlabRoot, legacyAnalysisDir);
    else
        % No conflict — safe to use genpath
        addpath(genpath(matlabRoot));
    end

    % Confirm key sub-modules are resolvable on the path
    requiredFunctions = { ...
        'config', ...
        'initializePythonML', ...
        'classifyAudio', ...
        'classifyImage', ...
        'runAudioPipeline', ...
        'runImagePipeline' };

    allReady = true;
    for k = 1:numel(requiredFunctions)
        fn = requiredFunctions{k};
        if isempty(which(fn))
            fprintf('      [WARNING] Function not found on path: %s\n', fn);
            allReady = false;
        end
    end

    if allReady
        fprintf('      MATLAB modules    : READY\n');
        fprintf('      DSP modules       : READY\n');
        fprintf('      Preprocessing     : READY\n');
    else
        fprintf('      [WARNING] Some MATLAB functions could not be located.\n');
        fprintf('                Check that matlab/ sub-directories are intact.\n');
    end
end

% ----------------------------------------------------------------------- %
function addMatlabPathsSelectively(matlabRoot, excludeDir)
%ADDMATLABPATHSSELECTIVELY  Adds sub-directories of matlabRoot one by one,
%   skipping excludeDir and its descendants to avoid function shadowing.

    excludeDir = strtrim(excludeDir);

    % Walk immediate sub-directories (max 1 level deep via genpath per dir)
    items = dir(matlabRoot);
    for k = 1:numel(items)
        if ~items(k).isdir || strcmp(items(k).name, '.') || strcmp(items(k).name, '..')
            continue;
        end
        absItem = fullfile(matlabRoot, items(k).name);
        % Skip if this is (or is inside) the excluded directory
        if startsWith(absItem, excludeDir)
            continue;
        end
        addpath(genpath(absItem));
    end
    % Also add the root matlab/ directory itself (for main.m, config.m, etc.)
    addpath(matlabRoot);
end

% ----------------------------------------------------------------------- %
function pyExe = checkPython()
%CHECKPYTHON  Interrogates the Python interpreter embedded in MATLAB.
%
%   Uses pyenv (R2019b+) when available, otherwise falls back to the
%   MATLAB 'python' command.  Reports the executable path and version.
%   Stops startup if Python is unavailable.

    pyExe = '';

    % ---- Detect via pyenv (preferred) ----------------------------------
    try
        pe = pyenv();
        if strlength(pe.Executable) > 0
            pyExe    = char(pe.Executable);
            pyVer    = char(pe.Version);
            pyStatus = char(pe.Status);

            fprintf('      Python executable : %s\n', pyExe);
            fprintf('      Python version    : %s\n', pyVer);
            fprintf('      MATLAB py.Status  : %s\n', pyStatus);

            if ~strcmp(pyStatus, 'Loaded') && ~strcmp(pyStatus, 'NotLoaded')
                % Unexpected status
                fprintf('      [WARNING] Unexpected Python status: %s\n', pyStatus);
            end

            fprintf('      Python runtime    : READY\n');
            return;
        end
    catch ME
        % pyenv not available on this MATLAB release; fall through
        if ~strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
            fprintf('      [WARNING] pyenv query error: %s\n', ME.message);
        end
    end

    % ---- Fallback: try py.sys to confirm Python is reachable ------------
    try
        pySys = py.importlib.import_module('sys');
        pyExe = char(pySys.executable);
        pyVer = char(pySys.version);
        fprintf('      Python executable : %s\n', pyExe);
        fprintf('      Python version    : %s\n', pyVer);
        fprintf('      Python runtime    : READY\n');
        return;
    catch ME2
        % Nothing worked
    end

    % ---- Hard failure ---------------------------------------------------
    startupError( ...
        'Python could not be initialized.', ...
        'MATLAB could not locate or start the Python interpreter.', ...
        sprintf('Required action:\n  Configure Python for MATLAB using pyenv(''Version'', ''<path>'')\n  See: https://www.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html'));
end

% ----------------------------------------------------------------------- %
function checkDependencies(pyExe)
%CHECKDEPENDENCIES  Verifies that critical Python packages are importable.
%
%   Package list is derived from requirements.txt plus PySide6 (UI).
%   Does NOT install anything automatically.

    % Packages to verify: { import_name, display_name }
    % import_name is what Python uses in 'import X'
    packages = { ...
        'numpy',      'NumPy';      ...
        'scipy',      'SciPy';      ...
        'torch',      'PyTorch';    ...
        'sklearn',    'scikit-learn'; ...
        'joblib',     'joblib';     ...
        'PIL',        'Pillow';     ...
        'cv2',        'OpenCV';     ...
        'librosa',    'librosa';    ...
        'matplotlib', 'matplotlib'; ...
        'PySide6',    'PySide6'     ...
    };

    allOk = true;
    for k = 1:size(packages, 1)
        importName  = packages{k, 1};
        displayName = packages{k, 2};

        ok = checkSinglePackage(importName);

        if ok
            fprintf('      %-16s : READY\n', displayName);
        else
            fprintf('      %-16s : MISSING\n', displayName);
            allOk = false;
        end
    end

    if ~allOk
        fprintf('\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('SPANDHAN STARTUP FAILED\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('\nReason:\n  One or more required Python packages are missing.\n\n');
        fprintf('Install missing packages using:\n');
        if ~isempty(pyExe)
            fprintf('  "%s" -m pip install <package>\n', pyExe);
        else
            fprintf('  python -m pip install <package>\n');
        end
        fprintf('\nOr install all requirements at once:\n');
        fprintf('  "%s" -m pip install -r requirements.txt\n', pyExe);
        fprintf('  "%s" -m pip install PySide6\n', pyExe);
        fprintf('------------------------------------------------------------\n\n');
        error('SPANDHAN:MissingDependency', ...
            'One or more required Python packages are unavailable. See messages above.');
    end
end

% ----------------------------------------------------------------------- %
function ok = checkSinglePackage(importName)
%CHECKSINGPLEPACKAGE  Returns true if the Python package can be imported.

    ok = false;
    try
        py.importlib.import_module(importName);
        ok = true;
    catch
        ok = false;
    end
end

% ----------------------------------------------------------------------- %
function checkModels(projectRoot)
%CHECKMODELS  Verifies that all required trained model files exist on disk.
%
%   Expected paths are taken from initializePythonML.m (source of truth):
%       models/audio/audio_signal_classifier.pkl
%       models/audio/audio_scaler.pkl
%       models/image/image_signal_classifier.pkl
%       models/image/image_scaler.pkl
%
%   A stray root-level models/image_signal_classifier.pkl also exists;
%   initializePythonML falls back to it for the image model, so we report
%   it but do not treat its absence as fatal if the canonical path exists.

    modelsRoot = fullfile(projectRoot, 'models');

    % Required model files (must exist for startup to proceed)
    required = { ...
        fullfile(modelsRoot, 'audio', 'audio_signal_classifier.pkl'), 'Audio classifier'; ...
        fullfile(modelsRoot, 'audio', 'audio_scaler.pkl'),            'Audio scaler';     ...
        fullfile(modelsRoot, 'image', 'image_signal_classifier.pkl'), 'Image classifier'; ...
        fullfile(modelsRoot, 'image', 'image_scaler.pkl'),            'Image scaler'      ...
    };

    fprintf('      [MODEL CHECK]\n');
    allOk = true;
    for k = 1:size(required, 1)
        modelPath = required{k, 1};
        label     = required{k, 2};
        if isfile(modelPath)
            fprintf('      %-20s : READY\n', label);
        else
            fprintf('      %-20s : MISSING\n', label);
            fprintf('        Expected at: %s\n', modelPath);
            allOk = false;
        end
    end

    if ~allOk
        fprintf('\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('SPANDHAN STARTUP FAILED\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('\nReason:\n  One or more trained model files are missing.\n\n');
        fprintf('Required action:\n');
        fprintf('  Run the SPANDHAN training pipeline to generate missing models,\n');
        fprintf('  or restore them from a backup.\n');
        fprintf('  Do NOT run training from this launcher.\n');
        fprintf('------------------------------------------------------------\n\n');
        error('SPANDHAN:ModelNotFound', ...
            'One or more trained model files are missing. See messages above.');
    end
end

% ----------------------------------------------------------------------- %
function initializeML(projectRoot) %#ok<INUSD>
%INITIALIZEML  Initialises the persistent MATLAB<->Python ML predictors.
%
%   Delegates entirely to the existing initializePythonML function.
%   Does NOT reimplement model loading.

    try
        [audioPredictor, imagePredictor] = initializePythonML();

        if ~isempty(audioPredictor)
            fprintf('      Audio predictor   : READY\n');
        else
            fprintf('      Audio predictor   : NOT LOADED\n');
        end

        if ~isempty(imagePredictor)
            fprintf('      Image predictor   : READY\n');
        else
            fprintf('      Image predictor   : NOT LOADED\n');
        end

    catch ME
        fprintf('\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('SPANDHAN STARTUP FAILED\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('\nReason:\n  ML interface initialization failed.\n\n');
        fprintf('MATLAB error:\n  %s\n\n', ME.message);
        fprintf('Identifier : %s\n', ME.identifier);
        if ~isempty(ME.stack)
            fprintf('Location   : %s  (line %d)\n', ...
                ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\nLikely causes:\n');
        fprintf('  - Python interpreter not configured for MATLAB\n');
        fprintf('  - Required Python packages not installed\n');
        fprintf('  - Trained model files not found\n');
        fprintf('  - ml.audio.inference.predict or ml.image.inference.predict not importable\n');
        fprintf('------------------------------------------------------------\n\n');
        rethrow(ME);
    end
end

% ----------------------------------------------------------------------- %
function launchUI(projectRoot, pyExe)
%LAUNCHUI  Launches the PySide6 UI as a separate OS process.
%
%   Entry point: <projectRoot>/ui/main.py   (confirmed to exist)
%   The UI process runs independently; MATLAB is not blocked.
%   The process handle is returned to the base workspace for inspection.

    uiEntryPoint = fullfile(projectRoot, 'ui', 'main.py');

    if ~isfile(uiEntryPoint)
        startupError( ...
            'UI entry point not found.', ...
            sprintf('Expected: %s', uiEntryPoint), ...
            'Verify that ui/main.py exists in the project.');
    end

    % Determine the Python executable to use for the subprocess.
    % Prefer the interpreter that MATLAB's Python engine is using
    % (already validated above); fall back to 'python' only as last resort.
    if isempty(pyExe)
        pyExe = 'python';
    end

    % Build the system command.
    % Quote both the executable and the script path to handle spaces safely.
    % Set the working directory to projectRoot so that relative imports
    % inside ui/main.py resolve correctly.
    cmdStr = sprintf('"%s" "%s"', pyExe, uiEntryPoint);

    % On Windows, use 'start /B' to launch as a detached background process
    % so MATLAB's command window is not blocked.
    launchCmd = sprintf('start /B "" %s', cmdStr);

    fprintf('      UI entry point    : %s\n', uiEntryPoint);
    fprintf('      Launch command    : %s\n', cmdStr);
    fprintf('      Working directory : %s\n', projectRoot);

    % Save current directory and switch to projectRoot for the launch
    oldDir = pwd;
    try
        cd(projectRoot);
        [status, cmdOut] = system(launchCmd);
        cd(oldDir);
    catch ME
        cd(oldDir);
        rethrow(ME);
    end

    if status ~= 0
        fprintf('\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('SPANDHAN STARTUP FAILED\n');
        fprintf('------------------------------------------------------------\n');
        fprintf('\nReason:\n  UI process could not be started.\n\n');
        fprintf('Command:\n  %s\n\n', cmdStr);
        fprintf('System output:\n  %s\n', cmdOut);
        fprintf('\nRequired action:\n');
        fprintf('  Verify Python executable: %s\n', pyExe);
        fprintf('  Verify PySide6 is installed in that environment.\n');
        fprintf('------------------------------------------------------------\n\n');
        error('SPANDHAN:UILaunchFailed', ...
            'Failed to launch PySide6 UI process. See messages above.');
    end

    fprintf('      UI process        : STARTED\n');
end

% ----------------------------------------------------------------------- %
function startupError(reason, detail, action)
%STARTUPERROR  Prints a structured error block and stops execution.

    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('SPANDHAN STARTUP FAILED\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('\nReason:\n  %s\n', reason);
    if nargin >= 2 && ~isempty(detail)
        fprintf('\nDetail:\n  %s\n', detail);
    end
    if nargin >= 3 && ~isempty(action)
        fprintf('\nRequired action:\n  %s\n', action);
    end
    fprintf('------------------------------------------------------------\n\n');
    error('SPANDHAN:StartupFailed', '%s', reason);
end
