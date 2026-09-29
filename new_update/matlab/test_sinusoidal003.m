function test_sinusoidal003()
% TEST_SINUSOIDAL003
% End-to-end verification script for audio and image sinusoidal_0003 samples.

    scriptDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(scriptDir);
    addpath(genpath(scriptDir));

    logFile = fullfile(projectRoot, 'test_sinusoidal003_result.txt');
    fid = fopen(logFile, 'w');

    logPrint(fid, '============================================================\n');
    logPrint(fid, '  SPANDHAN VERIFICATION: SINUSOIDAL_0003 TEST\n');
    logPrint(fid, '============================================================\n\n');

    %% 1. AUDIO TEST
    audioFile = fullfile(projectRoot, 'datasets', 'audio', 'sinusoidal', 'sinusoidal_0003.wav');
    logPrint(fid, '>>> [1/2] RUNNING AUDIO PIPELINE: %s\n', audioFile);

    try
        audioRes = runAudioPipeline(audioFile, 'Visualize', false);
        logPrint(fid, '\nAUDIO PIPELINE RESULT:\n');
        logPrint(fid, '  Modality:             %s\n', audioRes.modality);
        logPrint(fid, '  Input File:           %s\n', audioRes.input.file);
        logPrint(fid, '  Original Audio:       %d samples @ %d Hz\n', ...
            audioRes.input.originalSamples, audioRes.input.originalFs);
        logPrint(fid, '  Preprocessed Audio:   %d samples @ %d Hz\n', ...
            audioRes.preprocessing.processedSamples, audioRes.preprocessing.processedFs);
        logPrint(fid, '  Predicted Class:      %s\n', upper(audioRes.ml.class));
        logPrint(fid, '  ML Confidence:        %.2f%%\n', audioRes.ml.confidence * 100);
        logPrint(fid, '  Completed DSP:        %s\n', strjoin(audioRes.dsp.completedAnalyses, ', '));
        logPrint(fid, '  DSP Status:           %s\n', audioRes.dsp.status);
        if isfield(audioRes.dsp, 'fft') && ~isempty(audioRes.dsp.fft)
            logPrint(fid, '  FFT Peak Frequency:   %.2f Hz (Magnitude: %.4f)\n', ...
                audioRes.dsp.fft.peakFrequency, audioRes.dsp.fft.peakMagnitude);
        end
        if isfield(audioRes.dsp, 'fir') && ~isempty(audioRes.dsp.fir)
            logPrint(fid, '  FIR Filter:           Order %d, Cutoff %.0f Hz\n', ...
                audioRes.dsp.fir.order, audioRes.dsp.fir.cutoff);
        end
        if isfield(audioRes.dsp, 'iir') && ~isempty(audioRes.dsp.iir)
            logPrint(fid, '  IIR Filter:           Order %d (%s)\n', ...
                audioRes.dsp.iir.order, audioRes.dsp.iir.filterFamily);
        end
        logPrint(fid, '  Audio Pipeline:       SUCCESS\n\n');
    catch ME
        logPrint(fid, '  Audio Pipeline FAILED: %s\n\n', ME.message);
    end

    %% 2. IMAGE TEST
    imageFile = fullfile(projectRoot, 'datasets', 'image', 'sinusoidal', 'sinusoidal_0003.png');
    logPrint(fid, '>>> [2/2] RUNNING IMAGE PIPELINE: %s\n', imageFile);

    try
        imageRes = runImagePipeline(imageFile, 'Visualize', false);
        logPrint(fid, '\nIMAGE PIPELINE RESULT:\n');
        logPrint(fid, '  Modality:             %s\n', imageRes.modality);
        logPrint(fid, '  Input File:           %s\n', imageRes.input.file);
        logPrint(fid, '  Original Size:        %s\n', mat2str(imageRes.input.originalSize));
        logPrint(fid, '  Preprocessed Size:    %s (%s)\n', ...
            mat2str(imageRes.preprocessing.processedSize), imageRes.preprocessing.dataType);
        logPrint(fid, '  Predicted Class:      %s\n', upper(imageRes.ml.class));
        logPrint(fid, '  ML Confidence:        %.2f%%\n', imageRes.ml.confidence * 100);
        logPrint(fid, '  Completed DSP:        %s\n', strjoin(imageRes.dsp.completedAnalyses, ', '));
        logPrint(fid, '  DSP Status:           %s\n', imageRes.dsp.status);
        if isfield(imageRes.dsp, 'fft2') && ~isempty(imageRes.dsp.fft2)
            logPrint(fid, '  2-D FFT:              Completed (%dx%d spectrum)\n', ...
                size(imageRes.dsp.fft2.spectrum2D, 1), size(imageRes.dsp.fft2.spectrum2D, 2));
        end
        if isfield(imageRes.dsp, 'filter') && ~isempty(imageRes.dsp.filter)
            logPrint(fid, '  Image Filter:         %s (Kernel Size: %d)\n', ...
                imageRes.dsp.filter.filterType, imageRes.dsp.filter.kernelSize);
        end
        logPrint(fid, '  Image Pipeline:       SUCCESS\n\n');
    catch ME
        logPrint(fid, '  Image Pipeline FAILED: %s\n\n', ME.message);
    end

    logPrint(fid, '============================================================\n');
    logPrint(fid, '  TEST COMPLETED\n');
    logPrint(fid, '============================================================\n');

    if fid > 0
        fclose(fid);
    end

end

function logPrint(fid, formatStr, varargin)
    fprintf(formatStr, varargin{:});
    if fid > 0
        fprintf(fid, formatStr, varargin{:});
    end
end
