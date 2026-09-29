function tests = testFFT
%TESTFFT Unit tests for runFFT (SPANDHAN DSP module).
    tests = functiontests(localfunctions);
end

% =========================================================================
%  OUTPUT LENGTH TESTS
% =========================================================================

function testOneSidedFrequencyLength(testCase)
%One-sided frequency vector must have nfft/2+1 bins (even nfft).
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyEqual(numel(r.frequency), 4096/2 + 1);
end

% -------------------------------------------------------------------------

function testTwoSidedFrequencyLength(testCase)
%Two-sided frequency vector must have exactly nfft bins.
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyEqual(numel(r.frequencyTwoSided), 4096);
end

% -------------------------------------------------------------------------

function testOneSidedMagnitudeLength(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 512);
    testCase.verifyEqual(numel(r.magnitude), numel(r.frequency));
end

% =========================================================================
%  PEAK FREQUENCY DETECTION
% =========================================================================

function testPeakFrequencyDetectedSingleTone(testCase)
%For a pure 1 kHz sine, peak frequency must be within 10 Hz of 1000 Hz.
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyEqual(r.peakFrequency, 1000, "AbsTol", 10, ...
        "Peak frequency must be within 10 Hz of the true tone.");
end

% -------------------------------------------------------------------------

function testPeakFrequencyIgnoresDC(testCase)
%Peak detection must not report DC (0 Hz) as the peak.
    Fs = 16000;
    t  = (0:1/Fs:1-1/Fs)';
    x  = 5 + sin(2*pi*1000*t);     % large DC offset + 1 kHz tone
    r  = runFFT(x, Fs, 4096);
    testCase.verifyGreaterThan(r.peakFrequency, 0, ...
        "Peak must not be at DC even with a large DC offset.");
end

% =========================================================================
%  AMPLITUDE SCALING
% =========================================================================

function testMagnitudeScalingPureSine(testCase)
%For sin(2*pi*f*t) of amplitude A, peak magnitude must ≈ A.
    Fs  = 16000;
    t   = (0:1/Fs:2-1/Fs)';
    A   = 0.8;
    x   = A * sin(2*pi*2000*t);
    r   = runFFT(x, Fs, 2^nextpow2(length(x)));
    testCase.verifyEqual(r.peakMagnitude, A, "AbsTol", 0.02, ...
        "Peak magnitude must equal the sine amplitude.");
end

% =========================================================================
%  DC REMOVAL
% =========================================================================

function testDCValueStoredCorrectly(testCase)
    Fs = 16000;
    t  = (0:1/Fs:1-1/Fs)';
    x  = 3.7 + sin(2*pi*500*t);
    r  = runFFT(x, Fs, 4096);
    testCase.verifyEqual(r.dcValue, mean(x), "AbsTol", 1e-10, ...
        "dcValue must equal mean(x).");
end

% -------------------------------------------------------------------------

function testInputPreservedAfterDCRemoval(testCase)
%.input must be the original x (unchanged by DC removal).
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyEqual(r.input, x);
end

% -------------------------------------------------------------------------

function testAnalysisSignalIsMeanFree(testCase)
%analysisSignal must have mean ≈ 0.
    Fs = 16000;
    t  = (0:1/Fs:1-1/Fs)';
    x  = 10 + sin(2*pi*500*t);
    r  = runFFT(x, Fs, 4096);
    testCase.verifyEqual(mean(r.analysisSignal), 0, "AbsTol", 1e-10);
end

% =========================================================================
%  AUTOMATIC NFFT
% =========================================================================

function testAutoNFFTIsPowerOfTwo(testCase)
%When nfft=[], result.nfft must be a power of 2 ≥ length(x).
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, []);
    testCase.verifyEqual(r.nfft, 2^nextpow2(length(x)));
end

% =========================================================================
%  ALL STRUCT FIELDS PRESENT
% =========================================================================

function testAllFieldsPresent(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);

    requiredFields = { ...
        'input', 'analysisSignal', 'time', 'dcValue', ...
        'nfft', 'samplingFrequency', ...
        'frequency', 'spectrum', 'magnitude', 'magnitudeDB', ...
        'phase', 'power', ...
        'frequencyTwoSided', 'spectrumTwoSided', 'magnitudeTwoSided', ...
        'phaseTwoSided', 'powerTwoSided', ...
        'dcMagnitude', 'peakFrequency', 'peakMagnitude', ...
        'spectralCentroid', 'spectralBandwidth', ...
        'bandwidth3dB', 'lower3dBFrequency', 'upper3dBFrequency', ...
        'spectralFlatness', 'energy', 'rms' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(r, requiredFields{k}), ...
            sprintf("Missing field: %s", requiredFields{k}));
    end
end

% =========================================================================
%  SPECTRAL CHARACTERISTICS
% =========================================================================

function testSpectralCentroidPositive(testCase)
    [x, Fs] = makeSine(2000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyGreaterThan(r.spectralCentroid, 0);
end

% -------------------------------------------------------------------------

function testSpectralBandwidthNonNegative(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyGreaterThanOrEqual(r.spectralBandwidth, 0);
end

% -------------------------------------------------------------------------

function testBandwidth3dBNonNegative(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    if ~isnan(r.bandwidth3dB)
        testCase.verifyGreaterThanOrEqual(r.bandwidth3dB, 0);
    end
end

% -------------------------------------------------------------------------

function testSpectralFlatnessRange(testCase)
%spectralFlatness must lie in [0, 1].
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyGreaterThanOrEqual(r.spectralFlatness, 0);
    testCase.verifyLessThanOrEqual(r.spectralFlatness,    1);
end

% -------------------------------------------------------------------------

function testWhiteNoiseHigherFlatness(testCase)
%White noise must have higher spectral flatness than a pure sine.
    Fs      = 16000;
    [xSine, ~] = makeSine(1000);
    xNoise  = randn(size(xSine));

    rSine  = runFFT(xSine,  Fs, 4096);
    rNoise = runFFT(xNoise, Fs, 4096);

    testCase.verifyGreaterThan(rNoise.spectralFlatness, ...
                               rSine.spectralFlatness, ...
        "White noise must be more spectrally flat than a pure sine.");
end

% =========================================================================
%  ENERGY AND RMS
% =========================================================================

function testEnergyNonNegative(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyGreaterThanOrEqual(r.energy, 0);
end

% -------------------------------------------------------------------------

function testRMSMatchesMATLAB(testCase)
%result.rms must equal rms(x - mean(x)).
    [x, Fs] = makeSine(500);
    r = runFFT(x, Fs, 4096);
    expected = sqrt(mean((x - mean(x)).^2));
    testCase.verifyEqual(r.rms, expected, "AbsTol", 1e-12);
end

% =========================================================================
%  MAGNITUDE DB
% =========================================================================

function testMagnitudeDBIsReal(testCase)
    [x, Fs] = makeSine(1000);
    r = runFFT(x, Fs, 4096);
    testCase.verifyTrue(isreal(r.magnitudeDB));
    testCase.verifyTrue(all(isfinite(r.magnitudeDB)));
end

% =========================================================================
%  ROW-VECTOR INPUT
% =========================================================================

function testRowVectorInputAccepted(testCase)
    Fs = 16000;
    x  = sin(2*pi*1000*(0:Fs-1)/Fs);    % row vector
    r  = runFFT(x, Fs, 4096);
    testCase.verifyEqual(size(r.input, 2), 1, "input must be stored as column.");
end

% =========================================================================
%  TWO-TONE SIGNAL — DUAL PEAK
% =========================================================================

function testTwoTonesHaveCorrectPeaks(testCase)
%For x = sin(2pi*1000*t) + 0.5*sin(2pi*3000*t):
%  - peak frequency must be 1000 Hz (stronger component)
%  - 1000 Hz magnitude must be ≈ 2× 3000 Hz magnitude.
    Fs = 16000;
    t  = (0:1/Fs:2-1/Fs)';
    x  = sin(2*pi*1000*t) + 0.5*sin(2*pi*3000*t);
    r  = runFFT(x, Fs, 2^nextpow2(length(x)));

    testCase.verifyEqual(r.peakFrequency, 1000, "AbsTol", 10, ...
        "Strongest component must be at 1000 Hz.");

    % Find bins closest to 1000 Hz and 3000 Hz
    [~, idx1] = min(abs(r.frequency - 1000));
    [~, idx3] = min(abs(r.frequency - 3000));

    ratio = r.magnitude(idx1) / r.magnitude(idx3);
    testCase.verifyEqual(ratio, 2, "AbsTol", 0.1, ...
        "1000 Hz magnitude must be ~2x the 3000 Hz magnitude.");
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [x, Fs] = makeSine(f0)
%MAKESINE 2-second pure sine of frequency f0 at Fs = 16000 Hz.
    Fs = 16000;
    t  = (0:1/Fs:2-1/Fs)';
    x  = sin(2*pi*f0*t);
end
