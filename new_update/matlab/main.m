function outResult = main(mode, filepath)
% MAIN - SPANDHAN Signal Analyzer Main Execution Pipeline
% End-to-end integration: Loading -> Preprocessing -> ML Classifier -> Class-Aware DSP -> Visualization
%
% USAGE:
%   main()                     % Interactive prompt for modality and file path
%   result = main("audio", filepath)
%   result = main("image", filepath)

    matlabRoot = fileparts(mfilename("fullpath"));
    addpath(genpath(matlabRoot));

    fprintf("\n============================================================\n");
    fprintf("              SPANDHAN SIGNAL ANALYZER\n");
    fprintf("============================================================\n\n");

    % Load project configuration
    cfg = config();

    % 1. Determine modality
    if nargin < 1 || isempty(mode)
        mode = input("Enter modality (audio/image): ", "s");
    end
    mode = lower(strtrim(string(mode)));

    % 2. Determine file path
    if nargin < 2 || isempty(filepath)
        filepath = input("Enter input file path: ", "s");
    end
    filepath = strtrim(string(filepath));

    % Remove wrapping quotes if entered by user
    filepath = regexprep(filepath, '^["'']|["'']$', '');

    % 3. Route to dedicated pipeline
    switch mode
        case "audio"
            result = runAudioPipeline(filepath);

        case "image"
            result = runImagePipeline(filepath);

        otherwise
            error("Unsupported modality '%s'. Please choose 'audio' or 'image'.", mode);
    end

    fprintf("SPANDHAN pipeline execution completed successfully.\n\n");

    if nargout > 0
        outResult = result;
    end

end
