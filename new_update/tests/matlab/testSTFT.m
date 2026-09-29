function tests = testSTFT
%TESTSTFT Unit tests for runSTFT (SPANDHAN DSP module).
    tests = functiontests(localfunctions);
end

% =========================================================================
%  OUTPUT DIMENSION TESTS
% =========================================================================

function testSpectrumRowsEqualFrequencyBins(testCase)
%Spectrum must have nfft/2+1 rows (one-sided, even nfft).
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(size(r.spectrum, 1), nfft/2 + 1);
end

% -------------------------------------------------------------------------

function testFrequencyVectorLength(testCase)
%frequency vector must have nfft/2+1 elements.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.frequency), nfft/2 + 1);
end

% -------------------------------------------------------------------------

function testTimeVectorMatchesSpectrumColumns(testCase)
%Number of time frames must equal number of spectrum columns.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.time), size(r.spectrum, 2));
end

% -------------------------------------------------------------------------

function testMagnitudeSameShapeAsSpectrum(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(size(r.magnitude), size(r.spectrum));
end

% -------------------------------------------------------------------------

function testMagnitudeDBSameShapeAsSpectrum(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(size(r.magnitudeDB), size(r.spectrum));
end

% =========================================================================
%  PER-FRAME VECTOR DIMENSIONS
% =========================================================================

function testDominantFrequencyIsColumnVector(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(size(r.dominantFrequency, 2), 1);
    testCase.verifyEqual(numel(r.dominantFrequency), numel(r.time));
end

% -------------------------------------------------------------------------

function testSpectralCentroidLength(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.spectralCentroid), numel(r.time));
end

% -------------------------------------------------------------------------

function testSpectralBandwidthLength(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.spectralBandwidth), numel(r.time));
end

% -------------------------------------------------------------------------

function testFrameEnergyLength(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.frameEnergy), numel(r.time));
end

% =========================================================================
%  HOP SIZE AND RESOLUTION
% =========================================================================

function testHopSizeCalculation(testCase)
%hopSize must equal windowLength - overlap.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.hopSize, wL - ov);
end

% -------------------------------------------------------------------------

function testTimeResolution(testCase)
%timeResolution must equal windowLength / Fs.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.timeResolution, wL / Fs, "AbsTol", 1e-12);
end

% -------------------------------------------------------------------------

function testFrequencyResolution(testCase)
%frequencyResolution must equal Fs / nfft.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.frequencyResolution, Fs / nfft, "AbsTol", 1e-12);
end

% =========================================================================
%  ALL STRUCT FIELDS PRESENT
% =========================================================================

function testAllFieldsPresent(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);

    requiredFields = { ...
        'input', 'analysisSignal', 'dcValue', ...
        'spectrum', 'frequency', 'time', ...
        'magnitude', 'magnitudeDB', 'phase', ...
        'window', 'windowLength', 'overlap', 'hopSize', 'nfft', ...
        'samplingFrequency', ...
        'dominantFrequency', 'dominantMagnitude', ...
        'spectralCentroid', 'spectralBandwidth', 'frameEnergy', ...
        'totalEnergy', 'timeResolution', 'frequencyResolution' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(r, requiredFields{k}), ...
            sprintf("Missing field: %s", requiredFields{k}));
    end
end

% =========================================================================
%  DC REMOVAL
% =========================================================================

function testInputPreservedAfterDCRemoval(testCase)
%.input must be the original x (unchanged).
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.input, x(:));
end

% -------------------------------------------------------------------------

function testAnalysisSignalMeanFree(testCase)
    Fs  = 16000;
    t   = (0:1/Fs:2-1/Fs)';
    x   = 4.2 + chirp(t, 200, 2, 3000);   % large DC offset
    r   = runSTFT(x, Fs, 512, 256, 1024);
    testCase.verifyEqual(mean(r.analysisSignal), 0, "AbsTol", 1e-10);
end

% -------------------------------------------------------------------------

function testDCValueStoredCorrectly(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.dcValue, mean(x), "AbsTol", 1e-10);
end

% =========================================================================
%  CONFIGURATION FIELDS
% =========================================================================

function testWindowLengthStored(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.windowLength, wL);
end

% -------------------------------------------------------------------------

function testOverlapStored(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.overlap, ov);
end

% -------------------------------------------------------------------------

function testNFFTStored(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.nfft, nfft);
end

% -------------------------------------------------------------------------

function testWindowVectorLength(testCase)
%.window must have windowLength elements.
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(numel(r.window), wL);
end

% =========================================================================
%  ENERGY TESTS
% =========================================================================

function testTotalEnergyIsNonNegative(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyGreaterThanOrEqual(r.totalEnergy, 0);
end

% -------------------------------------------------------------------------

function testFrameEnergyNonNegative(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyTrue(all(r.frameEnergy >= 0), ...
        "All per-frame energies must be non-negative.");
end

% -------------------------------------------------------------------------

function testTotalEnergyEqualsFrameEnergySum(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyEqual(r.totalEnergy, sum(r.frameEnergy), "AbsTol", 1e-9);
end

% =========================================================================
%  SPECTRAL CENTROID SANITY
% =========================================================================

function testSpectralCentroidPositive(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyTrue(all(r.spectralCentroid >= 0), ...
        "Spectral centroid must be non-negative in all frames.");
end

% -------------------------------------------------------------------------

function testSpectralBandwidthNonNegative(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyTrue(all(r.spectralBandwidth >= 0), ...
        "Spectral bandwidth must be non-negative in all frames.");
end

% =========================================================================
%  CHIRP DOMINANT FREQUENCY TRACKING  (key SPANDHAN validation)
% =========================================================================

function testChirpFrequencyIncreases(testCase)
%A linear chirp sweeping low→high must have an increasing dominant
%frequency track.  We check that the median of the second half is
%higher than the median of the first half.
    Fs     = 16000;
    dur    = 3;
    t      = (0:1/Fs:dur-1/Fs)';
    fStart = 500;
    fEnd   = 5000;
    x      = chirp(t, fStart, dur, fEnd, "linear");

    r = runSTFT(x, Fs, 1024, 512, 2048);

    nFrames = numel(r.dominantFrequency);
    half    = floor(nFrames / 2);

    medFirst  = median(r.dominantFrequency(1   : half));
    medSecond = median(r.dominantFrequency(half+1 : end));

    testCase.verifyGreaterThan(medSecond, medFirst, ...
        "Dominant frequency must increase over time for a rising chirp.");
end

% -------------------------------------------------------------------------

function testChirpStartFrequencyApproximate(testCase)
%First frames must be near fStart (500 Hz), within 300 Hz.
    Fs  = 16000;
    dur = 3;
    t   = (0:1/Fs:dur-1/Fs)';
    x   = chirp(t, 500, dur, 5000, "linear");

    r = runSTFT(x, Fs, 1024, 512, 2048);

    startEstimate = median(r.dominantFrequency(1:5));
    testCase.verifyEqual(startEstimate, 500, "AbsTol", 300, ...
        "First frames must be near 500 Hz start frequency.");
end

% -------------------------------------------------------------------------

function testChirpEndFrequencyApproximate(testCase)
%Last frames must be near fEnd (5000 Hz), within 600 Hz.
    Fs  = 16000;
    dur = 3;
    t   = (0:1/Fs:dur-1/Fs)';
    x   = chirp(t, 500, dur, 5000, "linear");

    r = runSTFT(x, Fs, 1024, 512, 2048);

    nF          = numel(r.dominantFrequency);
    endEstimate = median(r.dominantFrequency(end-4 : end));
    testCase.verifyEqual(endEstimate, 5000, "AbsTol", 600, ...
        "Last frames must be near 5000 Hz end frequency.");
end

% =========================================================================
%  DOMINANT FREQUENCY SUPPRESSES DC
% =========================================================================

function testDominantFrequencyNotDC(testCase)
%dominantFrequency must never report 0 Hz (DC is suppressed).
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyTrue(all(r.dominantFrequency > 0), ...
        "Dominant frequency must not be 0 Hz (DC suppression required).");
end

% =========================================================================
%  MAGNITUDE DB
% =========================================================================

function testMagnitudeDBIsReal(testCase)
    [x, Fs, wL, ov, nfft] = makeChirp();
    r = runSTFT(x, Fs, wL, ov, nfft);
    testCase.verifyTrue(isreal(r.magnitudeDB));
    testCase.verifyTrue(all(isfinite(r.magnitudeDB(:))));
end

% =========================================================================
%  ROW-VECTOR INPUT
% =========================================================================

function testRowVectorInputAccepted(testCase)
    Fs = 16000;
    t  = (0:1/Fs:2-1/Fs);      % row vector
    x  = chirp(t, 100, 2, 3000);
    r  = runSTFT(x, Fs, 512, 256, 1024);
    testCase.verifyEqual(size(r.input, 2), 1, "input must be stored as column.");
end

% =========================================================================
%  PURE SINE — SINGLE-BIN CONCENTRATION
% =========================================================================

function testSineConcentratedAtCorrectFrequency(testCase)
%For a pure sine at f0, the median dominant frequency must be near f0.
    Fs = 16000;
    f0 = 2000;
    t  = (0:1/Fs:2-1/Fs)';
    x  = sin(2*pi*f0*t);

    r = runSTFT(x, Fs, 1024, 512, 2048);

    testCase.verifyEqual(median(r.dominantFrequency), f0, "AbsTol", 50, ...
        "Dominant frequency for a pure sine must be near f0.");
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [x, Fs, windowLength, overlap, nfft] = makeChirp()
%MAKECHIRP Standard 3-second linear chirp test signal.
    Fs           = 16000;
    windowLength = 1024;
    overlap      = 512;
    nfft         = 2048;
    dur          = 3;
    t            = (0:1/Fs:dur-1/Fs)';
    x            = chirp(t, 500, dur, 5000, "linear") + ...
                   0.01 * randn(size(t));
end
