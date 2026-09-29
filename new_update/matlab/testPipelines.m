% testPipelines.m - End-to-end 10-case verification test for SPANDHAN
%
% Tests runAudioPipeline and runImagePipeline across the 5 canonical classes:
%   1. Impulse
%   2. Sinusoidal
%   3. White Noise
%   4. Step
%   5. Chirp

clear; clc;
fprintf("========================================================================\n");
fprintf("          SPANDHAN END-TO-END PIPELINE VERIFICATION (10 CASES)\n");
fprintf("========================================================================\n\n");

matlabDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(matlabDir);
addpath(genpath(matlabDir));

classes = ["impulse", "sinusoidal", "white_noise", "step", "chirp"];
results = struct();

%% ------------------------------------------------------------------------
% 1. Audio Pipeline Tests (5 Cases)
% ------------------------------------------------------------------------
fprintf(">>> SECTION 1: AUDIO PIPELINE (5 CASES)\n");
fprintf("------------------------------------------------------------------------\n");

for i = 1:numel(classes)
    c = classes(i);
    audioFile = fullfile(projectRoot, "datasets", "audio", c, c + "_0001.wav");
    
    fprintf("\n--- Audio Test %d/5: Class '%s' ---\n", i, c);
    if ~isfile(audioFile)
        warning("Audio file not found: %s", audioFile);
        continue;
    end
    
    try
        res = runAudioPipeline(audioFile);
        results.audio.(c) = res;
        
        fprintf("Result Summary:\n");
        fprintf("  File:               %s\n", res.input.file);
        fprintf("  Original Fs:        %d Hz (%d samples)\n", res.input.originalFs, res.input.originalSamples);
        fprintf("  Processed Fs:       %d Hz (%d samples)\n", res.preprocessing.processedFs, res.preprocessing.processedSamples);
        fprintf("  Predicted Class:    %s (Confidence: %.2f%%)\n", res.ml.class, res.ml.confidence * 100);
        fprintf("  Completed DSP:      [%s]\n", strjoin(res.dsp.completedAnalyses, ", "));
        fprintf("  DSP Status:         %s\n", res.dsp.status);
    catch ME
        fprintf(2, "Audio pipeline failed for class '%s': %s\n", c, ME.message);
        results.audio.(c).error = ME.message;
    end
end

%% ------------------------------------------------------------------------
% 2. Image Pipeline Tests (5 Cases)
% ------------------------------------------------------------------------
fprintf("\n\n>>> SECTION 2: IMAGE PIPELINE (5 CASES)\n");
fprintf("------------------------------------------------------------------------\n");

for i = 1:numel(classes)
    c = classes(i);
    imgFile = fullfile(projectRoot, "datasets", "image", c, c + "_0001.png");
    
    fprintf("\n--- Image Test %d/5: Class '%s' ---\n", i, c);
    if ~isfile(imgFile)
        warning("Image file not found: %s", imgFile);
        continue;
    end
    
    try
        res = runImagePipeline(imgFile);
        results.image.(c) = res;
        
        fprintf("Result Summary:\n");
        fprintf("  File:               %s\n", res.input.file);
        fprintf("  Original Size:      %s\n", mat2str(res.input.originalSize));
        fprintf("  Processed Size:     %s (%s)\n", mat2str(res.preprocessing.processedSize), res.preprocessing.dataType);
        fprintf("  Predicted Class:    %s (Confidence: %.2f%%)\n", res.ml.class, res.ml.confidence * 100);
        fprintf("  Completed DSP:      [%s]\n", strjoin(res.dsp.completedAnalyses, ", "));
        fprintf("  DSP Status:         %s\n", res.dsp.status);
    catch ME
        fprintf(2, "Image pipeline failed for class '%s': %s\n", c, ME.message);
        results.image.(c).error = ME.message;
    end
end

%% ------------------------------------------------------------------------
% 3. Summary Table
% ------------------------------------------------------------------------
fprintf("\n\n========================================================================\n");
fprintf("                        10-CASE VERIFICATION MATRIX\n");
fprintf("========================================================================\n");
fprintf("%-8s | %-12s | %-14s | %-10s | %-25s\n", ...
    "Modality", "Input Class", "Predicted", "Confidence", "Completed DSP");
fprintf("------------------------------------------------------------------------\n");

for i = 1:numel(classes)
    c = classes(i);
    if isfield(results, "audio") && isfield(results.audio, c) && isfield(results.audio.(c), "ml")
        r = results.audio.(c);
        dspStr = strjoin(r.dsp.completedAnalyses, ", ");
        fprintf("%-8s | %-12s | %-14s | %8.2f%% | %-25s\n", ...
            "Audio", c, string(r.ml.class), r.ml.confidence * 100, dspStr);
    else
        fprintf("%-8s | %-12s | %-14s | %10s | %-25s\n", ...
            "Audio", c, "FAILED", "-", "-");
    end
end

fprintf("------------------------------------------------------------------------\n");

for i = 1:numel(classes)
    c = classes(i);
    if isfield(results, "image") && isfield(results.image, c) && isfield(results.image.(c), "ml")
        r = results.image.(c);
        dspStr = strjoin(r.dsp.completedAnalyses, ", ");
        fprintf("%-8s | %-12s | %-14s | %8.2f%% | %-25s\n", ...
            "Image", c, string(r.ml.class), r.ml.confidence * 100, dspStr);
    else
        fprintf("%-8s | %-12s | %-14s | %10s | %-25s\n", ...
            "Image", c, "FAILED", "-", "-");
    end
end

fprintf("========================================================================\n");
