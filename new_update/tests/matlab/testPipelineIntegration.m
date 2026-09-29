function tests = testPipelineIntegration
% TESTPIPELINEINTEGRATION Integration tests for SPANDHAN end-to-end pipelines
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    % Ensure all matlab directories are on path
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    matlabDir = fullfile(projectRoot, 'matlab');
    addpath(genpath(matlabDir));
end

%% =========================================================================
%  SELECT AUDIO ROUTING TESTS
% =========================================================================

function testAudioRoutingImpulse(testCase)
    sel = selectAudioDSPAnalysis("impulse");
    testCase.verifyEqual(sel.canonicalClass, "Impulse");
    testCase.verifyTrue(sel.useFFT);
    testCase.verifyTrue(sel.useWavelet);
    testCase.verifyTrue(sel.useFIR);
    testCase.verifyTrue(sel.useConvolution);
    testCase.verifyTrue(sel.useDeconvolution);
    testCase.verifyFalse(sel.useSTFT);
    testCase.verifyFalse(sel.useIIR);
end

function testAudioRoutingSinusoidal(testCase)
    sel = selectAudioDSPAnalysis("sinusoidal");
    testCase.verifyEqual(sel.canonicalClass, "Sinusoidal");
    testCase.verifyTrue(sel.useFFT);
    testCase.verifyTrue(sel.useFIR);
    testCase.verifyTrue(sel.useIIR);
    testCase.verifyFalse(sel.useSTFT);
    testCase.verifyFalse(sel.useWavelet);
    testCase.verifyFalse(sel.useConvolution);
    testCase.verifyFalse(sel.useDeconvolution);
end

function testAudioRoutingWhiteNoise(testCase)
    sel = selectAudioDSPAnalysis("white_noise");
    testCase.verifyEqual(sel.canonicalClass, "White Noise");
    testCase.verifyTrue(sel.useFFT);
    testCase.verifyTrue(sel.useFIR);
    testCase.verifyTrue(sel.useIIR);
    testCase.verifyFalse(sel.useSTFT);
    testCase.verifyFalse(sel.useWavelet);
end

function testAudioRoutingStep(testCase)
    sel = selectAudioDSPAnalysis("step");
    testCase.verifyEqual(sel.canonicalClass, "Step");
    testCase.verifyTrue(sel.useFFT);
    testCase.verifyTrue(sel.useWavelet);
    testCase.verifyTrue(sel.useFIR);
    testCase.verifyTrue(sel.useIIR);
    testCase.verifyTrue(sel.useConvolution);
    testCase.verifyFalse(sel.useSTFT);
    testCase.verifyFalse(sel.useDeconvolution);
end

function testAudioRoutingChirp(testCase)
    sel = selectAudioDSPAnalysis("chirp");
    testCase.verifyEqual(sel.canonicalClass, "Chirp");
    testCase.verifyTrue(sel.useFFT);
    testCase.verifyTrue(sel.useSTFT);
    testCase.verifyTrue(sel.useWavelet);
    testCase.verifyTrue(sel.useDeconvolution);
    testCase.verifyFalse(sel.useFIR);
    testCase.verifyFalse(sel.useIIR);
    testCase.verifyFalse(sel.useConvolution);
end

%% =========================================================================
%  SELECT IMAGE ROUTING TESTS
% =========================================================================

function testImageRoutingImpulse(testCase)
    sel = selectImageDSPAnalysis("impulse");
    testCase.verifyEqual(sel.canonicalClass, "Impulse");
    testCase.verifyTrue(sel.useFFT2);
    testCase.verifyTrue(sel.useWavelet2D);
    testCase.verifyTrue(sel.useFilter);
    testCase.verifyTrue(sel.useConvolution);
    testCase.verifyTrue(sel.useDeconvolution);
end

function testImageRoutingSinusoidal(testCase)
    sel = selectImageDSPAnalysis("sinusoidal");
    testCase.verifyEqual(sel.canonicalClass, "Sinusoidal");
    testCase.verifyTrue(sel.useFFT2);
    testCase.verifyFalse(sel.useWavelet2D);
    testCase.verifyTrue(sel.useFilter);
    testCase.verifyTrue(sel.useConvolution);
    testCase.verifyFalse(sel.useDeconvolution);
end

function testImageRoutingWhiteNoise(testCase)
    sel = selectImageDSPAnalysis("white_noise");
    testCase.verifyEqual(sel.canonicalClass, "White Noise");
    testCase.verifyTrue(sel.useFFT2);
    testCase.verifyFalse(sel.useWavelet2D);
    testCase.verifyTrue(sel.useFilter);
    testCase.verifyFalse(sel.useConvolution);
    testCase.verifyFalse(sel.useDeconvolution);
end

function testImageRoutingStep(testCase)
    sel = selectImageDSPAnalysis("step");
    testCase.verifyEqual(sel.canonicalClass, "Step");
    testCase.verifyTrue(sel.useFFT2);
    testCase.verifyTrue(sel.useWavelet2D);
    testCase.verifyTrue(sel.useFilter);
    testCase.verifyFalse(sel.useConvolution);
    testCase.verifyFalse(sel.useDeconvolution);
end

function testImageRoutingChirp(testCase)
    sel = selectImageDSPAnalysis("chirp");
    testCase.verifyEqual(sel.canonicalClass, "Chirp");
    testCase.verifyTrue(sel.useFFT2);
    testCase.verifyTrue(sel.useWavelet2D);
    testCase.verifyTrue(sel.useFilter);
    testCase.verifyFalse(sel.useConvolution);
    testCase.verifyTrue(sel.useDeconvolution);
end

%% =========================================================================
%  END-TO-END AUDIO PIPELINE TEST
% =========================================================================

function testEndToEndAudioPipeline(testCase)
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    sampleFile = fullfile(projectRoot, 'datasets', 'audio', 'chirp', 'chirp_0001.wav');

    if ~exist(sampleFile, 'file')
        % Fall back to synthetic audio file if dataset not found
        sampleFile = fullfile(tempdir, 'test_chirp.wav');
        Fs = 16000;
        t = (0:1/Fs:1-1/Fs)';
        y = chirp(t, 100, 1, 4000);
        audiowrite(sampleFile, y, Fs);
    end

    result = runAudioPipeline(sampleFile, "Visualize", false);

    % Verify complete contract
    testCase.verifyEqual(result.modality, "audio");
    testCase.verifyEqual(result.input.file, sampleFile);
    testCase.verifyEqual(length(result.preprocessing.signal), 32000);
    testCase.verifyEqual(result.preprocessing.processedFs, 16000);
    testCase.verifyTrue(isfield(result.ml, 'class'));
    testCase.verifyTrue(isfield(result.ml, 'confidence'));
    testCase.verifyTrue(isfield(result.ml, 'probabilities'));
    testCase.verifyNotEmpty(result.dsp.completedAnalyses);
    testCase.verifyTrue(isfield(result.features, 'modality'));
end

%% =========================================================================
%  END-TO-END IMAGE PIPELINE TEST
% =========================================================================

function testEndToEndImagePipeline(testCase)
    testDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(testDir);
    sampleFile = fullfile(projectRoot, 'datasets', 'image', 'chirp', 'chirp_0001.png');

    if ~exist(sampleFile, 'file')
        % Fall back to synthetic image file if dataset not found
        sampleFile = fullfile(tempdir, 'test_sin.png');
        [X, ~] = meshgrid(1:128, 1:128);
        img = uint8(255 * (0.5 + 0.5 * sin(2 * pi * 0.05 * X)));
        imwrite(img, sampleFile);
    end

    result = runImagePipeline(sampleFile, "Visualize", false);

    % Verify complete contract
    testCase.verifyEqual(result.modality, "image");
    testCase.verifyEqual(result.input.file, sampleFile);
    testCase.verifyEqual(size(result.preprocessing.image), [128, 128]);
    testCase.verifyTrue(isfield(result.ml, 'class'));
    testCase.verifyTrue(isfield(result.ml, 'confidence'));
    testCase.verifyTrue(isfield(result.ml, 'probabilities'));
    testCase.verifyNotEmpty(result.dsp.completedAnalyses);
    testCase.verifyTrue(isfield(result.features, 'modality'));
end
