function tests = testPreprocessing
%TESTPREPROCESSING Unit tests for the SPANDHAN preprocessing + DSP routing layer.
    tests = functiontests(localfunctions);
end

% =========================================================================
%  PREPROCESSING TESTS
% =========================================================================

function testPreprocessOutputLength(testCase)
%Preprocessed signal must have the same length as the raw input.
    Fs     = 16000;
    signal = sin(2*pi*440*(0:1/Fs:1)');
    proc   = preprocessAudio(signal, Fs);
    testCase.verifyEqual(length(proc), length(signal));
end

% -------------------------------------------------------------------------

function testPreprocessIsColumnVector(testCase)
    Fs     = 16000;
    signal = (sin(2*pi*440*(0:1/Fs:0.5)))';   % already column
    proc   = preprocessAudio(signal, Fs);
    testCase.verifyEqual(size(proc, 2), 1);
end

% -------------------------------------------------------------------------

function testNormalizeAudioRange(testCase)
%normalizeAudio must scale peak amplitude to exactly 1.
    signal = [0.1, -0.5, 0.3, 0.8, -0.2]';
    normed = normalizeAudio(signal);
    testCase.verifyEqual(max(abs(normed)), 1, "AbsTol", 1e-12);
end

% -------------------------------------------------------------------------

function testNormalizeAudioZeroSignal(testCase)
%Zero signal must be returned unchanged (avoids div-by-zero).
    signal = zeros(100, 1);
    normed = normalizeAudio(signal);
    testCase.verifyEqual(normed, signal);
end

% -------------------------------------------------------------------------

function testNormalizeImageRange(testCase)
%normalizeImage must scale pixel values to [0, 1].
    img    = [10, 20; 30, 40];
    normed = normalizeImage(double(img));
    testCase.verifyEqual(min(normed(:)), 0, "AbsTol", 1e-12);
    testCase.verifyEqual(max(normed(:)), 1, "AbsTol", 1e-12);
end

% =========================================================================
%  DSP ROUTING — PIPELINE STRUCTURE TESTS
% =========================================================================

function testImpulsePipelineFields(testCase)
%Impulse class must produce: fft, wavelet, fir, convolution, deconvolution.
    [sig, Fs] = makeSignal();
    r = selectDSPAnalysis(sig, Fs, "impulse");
    testCase.verifyTrue(isfield(r, 'fft'),           'impulse: missing fft');
    testCase.verifyTrue(isfield(r, 'wavelet'),       'impulse: missing wavelet');
    testCase.verifyTrue(isfield(r, 'fir'),           'impulse: missing fir');
    testCase.verifyTrue(isfield(r, 'convolution'),   'impulse: missing convolution');
    testCase.verifyTrue(isfield(r, 'deconvolution'), 'impulse: missing deconvolution');
    testCase.verifyFalse(isfield(r, 'stft'),         'impulse: should not have stft');
    testCase.verifyFalse(isfield(r, 'iir'),          'impulse: should not have iir');
end

% -------------------------------------------------------------------------

function testSinusoidalPipelineFields(testCase)
%Sinusoidal class must produce: fft, fir, iir.
    [sig, Fs] = makeSignal();
    r = selectDSPAnalysis(sig, Fs, "sinusoidal");
    testCase.verifyTrue(isfield(r, 'fft'),  'sinusoidal: missing fft');
    testCase.verifyTrue(isfield(r, 'fir'),  'sinusoidal: missing fir');
    testCase.verifyTrue(isfield(r, 'iir'),  'sinusoidal: missing iir');
    testCase.verifyFalse(isfield(r, 'stft'),         'sinusoidal: should not have stft');
    testCase.verifyFalse(isfield(r, 'wavelet'),      'sinusoidal: should not have wavelet');
    testCase.verifyFalse(isfield(r, 'convolution'),  'sinusoidal: should not have convolution');
    testCase.verifyFalse(isfield(r, 'deconvolution'),'sinusoidal: should not have deconvolution');
end

% -------------------------------------------------------------------------

function testWhiteNoisePipelineFields(testCase)
%White noise class must produce: fft, fir, iir.
    Fs  = 16000;
    sig = randn(Fs, 1);            % 1 second of Gaussian noise
    r   = selectDSPAnalysis(sig, Fs, "white_noise");
    testCase.verifyTrue(isfield(r, 'fft'), 'white_noise: missing fft');
    testCase.verifyTrue(isfield(r, 'fir'), 'white_noise: missing fir');
    testCase.verifyTrue(isfield(r, 'iir'), 'white_noise: missing iir');
end

% -------------------------------------------------------------------------

function testStepPipelineFields(testCase)
%Step class must produce: fft, wavelet, fir, iir, convolution.
    Fs  = 16000;
    sig = [zeros(Fs/2, 1); ones(Fs/2, 1)];   % step at t = 0.5 s
    r   = selectDSPAnalysis(sig, Fs, "step");
    testCase.verifyTrue(isfield(r, 'fft'),         'step: missing fft');
    testCase.verifyTrue(isfield(r, 'wavelet'),     'step: missing wavelet');
    testCase.verifyTrue(isfield(r, 'fir'),         'step: missing fir');
    testCase.verifyTrue(isfield(r, 'iir'),         'step: missing iir');
    testCase.verifyTrue(isfield(r, 'convolution'), 'step: missing convolution');
    testCase.verifyFalse(isfield(r, 'stft'),          'step: should not have stft');
    testCase.verifyFalse(isfield(r, 'deconvolution'), 'step: should not have deconvolution');
end

% -------------------------------------------------------------------------

function testChirpPipelineFields(testCase)
%Chirp class must produce: fft, stft, wavelet, deconvolution.
    Fs  = 16000;
    t   = (0:1/Fs:1-1/Fs)';
    sig = chirp(t, 100, 1, 4000);          % 100 Hz → 4 kHz linear chirp
    r   = selectDSPAnalysis(sig, Fs, "chirp");
    testCase.verifyTrue(isfield(r, 'fft'),           'chirp: missing fft');
    testCase.verifyTrue(isfield(r, 'stft'),          'chirp: missing stft');
    testCase.verifyTrue(isfield(r, 'wavelet'),       'chirp: missing wavelet');
    testCase.verifyTrue(isfield(r, 'deconvolution'), 'chirp: missing deconvolution');
    testCase.verifyFalse(isfield(r, 'fir'),          'chirp: should not have fir');
    testCase.verifyFalse(isfield(r, 'iir'),          'chirp: should not have iir');
    testCase.verifyFalse(isfield(r, 'convolution'),  'chirp: should not have convolution');
end

% =========================================================================
%  PIPELINE METADATA TESTS
% =========================================================================

function testSignalClassStoredCorrectly(testCase)
    [sig, Fs] = makeSignal();
    r = selectDSPAnalysis(sig, Fs, "chirp");
    testCase.verifyEqual(r.signalClass, "chirp");
end

% -------------------------------------------------------------------------

function testPipelineCellArrayReturned(testCase)
%result.pipeline must be a non-empty cell array.
    [sig, Fs] = makeSignal();
    r = selectDSPAnalysis(sig, Fs, "step");
    testCase.verifyClass(r.pipeline, 'cell');
    testCase.verifyGreaterThan(numel(r.pipeline), 0);
end

% =========================================================================
%  PARAMETER OVERRIDE TEST
% =========================================================================

function testOptionsOverrideApplied(testCase)
%Custom firOrder passed via options must reach the FIR module.
    [sig, Fs] = makeSignal();
    opts           = struct();
    opts.firOrder  = 50;
    opts.firCutoff = 2000;
    r = selectDSPAnalysis(sig, Fs, "sinusoidal", opts);
    testCase.verifyEqual(r.fir.order, 50, ...
        "Custom firOrder override must be applied to FIR module.");
end

% =========================================================================
%  FFT RESULT SPOT-CHECK
% =========================================================================

function testFFTResultHasFrequencyAndMagnitude(testCase)
    [sig, Fs] = makeSignal();
    r = selectDSPAnalysis(sig, Fs, "sinusoidal");
    testCase.verifyTrue(isfield(r.fft, 'frequency'),  'fft result missing frequency');
    testCase.verifyTrue(isfield(r.fft, 'magnitude'),  'fft result missing magnitude');
    testCase.verifyTrue(isfield(r.fft, 'peakFrequency'), 'fft result missing peakFrequency');
    testCase.verifyGreaterThan(numel(r.fft.frequency), 0);
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [sig, Fs] = makeSignal()
%MAKESIGNAL Standard 1-second 1 kHz sinusoidal test signal at 16 kHz.
    Fs  = 16000;
    t   = (0:1/Fs:1-1/Fs)';
    sig = sin(2*pi*1000*t) + 0.01*randn(size(t));
end
