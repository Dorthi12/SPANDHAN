function tests = testDeconvolution
%TESTDECONVOLUTION Unit tests for runDeconvolution (SPANDHAN DSP module).
    tests = functiontests(localfunctions);
end

% =========================================================================
%  OUTPUT LENGTH TESTS
% =========================================================================

function testReconstructedLengthEqualsObserved(testCase)
%Reconstructed signal must have the same length as the observed signal y.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(r.reconstructedLength, r.observedLength, ...
        "Reconstructed length must equal observed signal length.");
end

% -------------------------------------------------------------------------

function testOutputIsColumnVector(testCase)
%.reconstructed must be stored as a column vector.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(size(r.reconstructed, 2), 1, ...
        "Reconstructed signal must be a column vector.");
end

% =========================================================================
%  ALL THREE METHODS RUN WITHOUT ERROR
% =========================================================================

function testRegularizedMethodRuns(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(r.method, "regularized");
    testCase.verifyFalse(isempty(r.reconstructed));
end

% -------------------------------------------------------------------------

function testDirectMethodRuns(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "direct");
    testCase.verifyEqual(r.method, "direct");
    testCase.verifyFalse(isempty(r.reconstructed));
end

% -------------------------------------------------------------------------

function testWienerMethodRuns(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "wiener");
    testCase.verifyEqual(r.method, "wiener");
    testCase.verifyFalse(isempty(r.reconstructed));
end

% =========================================================================
%  IMPULSE IDENTITY TEST  (core mathematical sanity check)
% =========================================================================

function testImpulseIdentity(testCase)
%If h[n] = delta[n], then deconvolution of y should recover y exactly.
%
%  y = x * delta  =>  y = x  =>  deconv(y, delta) = x = y
    Fs    = 16000;
    Ny    = 500;
    y     = sin(2*pi*1000*(0:Ny-1)' / Fs);
    delta = zeros(50, 1);
    delta(1) = 1;                           % unit impulse → identity system

    r = runDeconvolution(y, delta, Fs, 1e-12, "regularized");

    testCase.verifyEqual(r.reconstructed, y, "AbsTol", 1e-6, ...
        "Deconvolution with delta impulse response should recover y.");
end

% =========================================================================
%  ROUNDTRIP TEST
% =========================================================================

function testRoundtripCorrelation(testCase)
%Build y = conv(x, h) cleanly (no noise), then deconvolve.
%Correlation between x and reconstructed x̂ should be high (> 0.9).
    Fs     = 16000;
    t      = (0:1/Fs:0.5-1/Fs)';
    x      = sin(2*pi*1000*t);

    th     = (0:99)' / Fs;
    h      = exp(-30*th) .* cos(2*pi*300*th);
    h      = h / max(abs(h));

    y      = conv(x, h, "full");            % clean convolution

    result = runDeconvolution(y, h, Fs, 1e-6, "regularized");

    % Compare reconstructed x̂ against original x (not y)
    Ncomp  = min(length(x), length(result.reconstructed));
    c      = corr(x(1:Ncomp), result.reconstructed(1:Ncomp));

    testCase.verifyGreaterThan(c, 0.9, ...
        "Clean roundtrip: correlation between x and x̂ must exceed 0.9.");
end

% =========================================================================
%  RESULT STRUCT FIELD TESTS
% =========================================================================

function testAllFieldsPresent(testCase)
%Every documented output field must exist in the result struct.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");

    requiredFields = { ...
        'observed', 'impulseResponse', 'reconstructed', ...
        'observedTime', 'impulseTime', 'reconstructedTime', ...
        'frequency', 'observedSpectrum', 'impulseSpectrum', ...
        'reconstructedSpectrum', 'transferFunction', ...
        'transferMagnitude', 'transferPhase', ...
        'samplingFrequency', 'lambda', 'method', 'fftLength', ...
        'observedLength', 'impulseResponseLength', 'reconstructedLength', ...
        'RMSE', 'MAE', 'correlation', ...
        'observedEnergy', 'impulseResponseEnergy', 'reconstructedEnergy', ...
        'observedRMS', 'reconstructedRMS', ...
        'observedPeak', 'reconstructedPeak', ...
        'observedPeakTime', 'reconstructedPeakTime', ...
        'minimumTransferMagnitude', 'maximumTransferMagnitude', ...
        'conditioningRatio', 'computationTime' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(r, requiredFields{k}), ...
            sprintf('Missing result field: %s', requiredFields{k}));
    end
end

% =========================================================================
%  SIGNAL PRESERVATION TESTS
% =========================================================================

function testObservedPreserved(testCase)
%.observed must be the original y as a column vector.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(r.observed, y(:));
end

% -------------------------------------------------------------------------

function testImpulseResponsePreserved(testCase)
%.impulseResponse must be the original h as a column vector.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(r.impulseResponse, h(:));
end

% =========================================================================
%  FFT / SPECTRUM TESTS
% =========================================================================

function testFFTLengthIsPowerOfTwo(testCase)
%fftLength must be 2^nextpow2(Ny + Nh - 1).
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    expected = 2^nextpow2(r.observedLength + r.impulseResponseLength - 1);
    testCase.verifyEqual(r.fftLength, expected);
end

% -------------------------------------------------------------------------

function testFrequencyVectorLength(testCase)
%frequency vector must have fftLength elements.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(numel(r.frequency), r.fftLength);
end

% -------------------------------------------------------------------------

function testSpectraLengths(testCase)
%All spectral arrays must have fftLength elements.
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(numel(r.observedSpectrum),      r.fftLength);
    testCase.verifyEqual(numel(r.impulseSpectrum),       r.fftLength);
    testCase.verifyEqual(numel(r.reconstructedSpectrum), r.fftLength);
end

% =========================================================================
%  ENERGY AND RMS TESTS
% =========================================================================

function testEnergyNonNegative(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyGreaterThanOrEqual(r.observedEnergy,       0);
    testCase.verifyGreaterThanOrEqual(r.impulseResponseEnergy, 0);
    testCase.verifyGreaterThanOrEqual(r.reconstructedEnergy,  0);
end

% -------------------------------------------------------------------------

function testRMSNonNegative(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyGreaterThanOrEqual(r.observedRMS,      0);
    testCase.verifyGreaterThanOrEqual(r.reconstructedRMS, 0);
end

% =========================================================================
%  PEAK TESTS
% =========================================================================

function testPeakMatchesMax(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(r.observedPeak,      max(abs(y)),                "AbsTol", 1e-14);
    testCase.verifyEqual(r.reconstructedPeak, max(abs(r.reconstructed)),  "AbsTol", 1e-14);
end

% =========================================================================
%  CONDITIONING TESTS
% =========================================================================

function testConditioningRatioInRange(testCase)
%conditioningRatio must lie in [0, 1].
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyGreaterThanOrEqual(r.conditioningRatio, 0);
    testCase.verifyLessThanOrEqual(r.conditioningRatio,    1);
end

% -------------------------------------------------------------------------

function testTransferMagnitudeNonNegative(testCase)
    [y, h, Fs] = makeSignals();
    r = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyTrue(all(r.transferMagnitude >= 0), ...
        "|H(f)| must be non-negative everywhere.");
end

% =========================================================================
%  LAMBDA SENSITIVITY TEST
% =========================================================================

function testLambdaStoredCorrectly(testCase)
%result.lambda must match the lambda argument supplied.
    [y, h, Fs] = makeSignals();
    lambda = 0.123;
    r = runDeconvolution(y, h, Fs, lambda, "regularized");
    testCase.verifyEqual(r.lambda, lambda);
end

% -------------------------------------------------------------------------

function testLargerLambdaGivesMoreSmoothing(testCase)
%Larger λ should produce lower output energy (more aggressive suppression).
    [y, h, Fs] = makeSignals();
    rLow  = runDeconvolution(y, h, Fs, 1e-6, "regularized");
    rHigh = runDeconvolution(y, h, Fs, 10,   "regularized");
    testCase.verifyGreaterThan(rLow.reconstructedEnergy, ...
                               rHigh.reconstructedEnergy, ...
        "Larger λ should yield lower reconstructed energy (more smoothing).");
end

% =========================================================================
%  ROW-VECTOR TOLERANCE TEST
% =========================================================================

function testRowVectorInputsAccepted(testCase)
%Row-vector inputs for y and h must be silently coerced to column vectors.
    Fs = 16000;
    y  = randn(1, 500);       % row vector
    h  = exp(-0.01*(0:99));   % row vector
    r  = runDeconvolution(y, h, Fs, 0.01, "regularized");
    testCase.verifyEqual(size(r.observed,      2), 1);
    testCase.verifyEqual(size(r.reconstructed, 2), 1);
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [y, h, Fs] = makeSignals()
%MAKESIGNALS Standard test: sinusoidal input convolved through a
%decaying oscillatory system, plus 0.5 % Gaussian noise.
    Fs     = 16000;
    t      = (0 : 1/Fs : 0.5 - 1/Fs)';
    x      = sin(2*pi*1000*t);

    th     = (0 : 199)' / Fs;
    h      = exp(-20*th) .* cos(2*pi*500*th);
    h      = h / max(abs(h));

    yClean = conv(x, h, "full");
    y      = yClean + 0.005 * randn(size(yClean));
end
