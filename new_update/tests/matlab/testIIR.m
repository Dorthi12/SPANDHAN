function tests = testIIR
%TESTIIR Unit tests for runIIR (SPANDHAN IIR filter module).
    tests = functiontests(localfunctions);
end

% -------------------------------------------------------------------------

function testLowpassOutputLength(testCase)
%Filtered output must have the same length as the input.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    testCase.verifyEqual(numel(result.output), numel(x));
end

% -------------------------------------------------------------------------

function testHighpassOutputLength(testCase)
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 4000, 3000, 1, 60, "high", "butter");
    testCase.verifyEqual(numel(result.output), numel(x));
end

% -------------------------------------------------------------------------

function testBandpassOutputLength(testCase)
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, [1000, 3000], [500, 4000], 1, 60, "bandpass", "butter");
    testCase.verifyEqual(numel(result.output), numel(x));
end

% -------------------------------------------------------------------------

function testBandstopOutputLength(testCase)
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, [1000, 5000], [2000, 4000], 1, 60, "stop", "butter");
    testCase.verifyEqual(numel(result.output), numel(x));
end

% -------------------------------------------------------------------------

function testAllFilterFamilies(testCase)
%All four filter families must run without error and produce correct length output.
    [x, Fs] = makeSignal();
    families = ["butter", "cheby1", "cheby2", "ellip"];
    for k = 1:numel(families)
        result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", families(k));
        testCase.verifyEqual(numel(result.output), numel(x), ...
            sprintf('Length mismatch for family: %s', families(k)));
    end
end

% -------------------------------------------------------------------------

function testResultStructureFields(testCase)
%All required fields must be present in the result struct.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");

    requiredFields = { ...
        'input', 'output', 'b', 'a', 'sos', 'scaleFactor', ...
        'order', 'filterType', 'filterFamily', ...
        'passbandFrequency', 'stopbandFrequency', ...
        'passbandRipple', 'stopbandAttenuation', ...
        'frequency', 'response', 'magnitude', 'magnitudeDB', 'phase', ...
        'groupDelay', 'groupDelayFrequency', ...
        'impulseResponse', 'impulseTime', ...
        'poles', 'poleMagnitudes', 'stable', ...
        'peakFrequency', 'peakMagnitude', 'estimated3dBCutoff', ...
        'samplingFrequency', 'nyquistFrequency', 'responsePoints' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(result, requiredFields{k}), ...
            sprintf('Missing result field: %s', requiredFields{k}));
    end
end

% -------------------------------------------------------------------------

function testStabilityFlag(testCase)
%Butterworth filter from buttord must always be stable.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    testCase.verifyTrue(result.stable, ...
        'Butterworth filter should be stable (all poles inside unit circle).');
end

% -------------------------------------------------------------------------

function testInputPreserved(testCase)
%The .input field must be a column vector identical to x.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    testCase.verifyEqual(result.input, x(:));
end

% -------------------------------------------------------------------------

function testOrderIsPositiveInteger(testCase)
%Filter order must be a positive integer.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    testCase.verifyGreaterThan(result.order, 0);
    testCase.verifyEqual(result.order, floor(result.order));
end

% -------------------------------------------------------------------------

function testFrequencyResponseLength(testCase)
%Frequency response vectors must all have length == responsePoints.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    n = result.responsePoints;
    testCase.verifyEqual(numel(result.frequency),   n);
    testCase.verifyEqual(numel(result.magnitude),   n);
    testCase.verifyEqual(numel(result.magnitudeDB), n);
    testCase.verifyEqual(numel(result.phase),       n);
end

% -------------------------------------------------------------------------

function testMagnitudeDBIsReal(testCase)
%magnitudeDB must contain only real, finite values.
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    testCase.verifyTrue(isreal(result.magnitudeDB));
    testCase.verifyTrue(all(isfinite(result.magnitudeDB)));
end

% -------------------------------------------------------------------------

function testImpulseResponseLength(testCase)
%Impulse response length must be max(4*(N+1), 512).
    [x, Fs] = makeSignal();
    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");
    expectedLength = max(4 * (result.order + 1), 512);
    testCase.verifyEqual(numel(result.impulseResponse), expectedLength);
end

% -------------------------------------------------------------------------

function testLowpassAttenuation(testCase)
%A 1 kHz tone should be retained and a 5 kHz tone should be attenuated
%by a low-pass filter with a 3 kHz passband.
    Fs       = 16000;
    duration = 2;
    t        = (0 : 1/Fs : duration - 1/Fs)';

    x = sin(2*pi*1000*t) + sin(2*pi*5000*t);

    result = runIIR(x, Fs, 3000, 4000, 1, 60, "low", "butter");

    y = result.output;

    % Energy in 1 kHz band (pass) and 5 kHz band (stop)
    pass = bandEnergy(y, Fs, 800, 1200);
    stop = bandEnergy(y, Fs, 4500, 5500);

    testCase.verifyGreaterThan(pass, stop * 10, ...
        '1 kHz component should be >> 5 kHz component after low-pass.');
end

% =========================================================================
%  LOCAL HELPERS
% =========================================================================

function [x, Fs] = makeSignal()
%MAKESIGNAL Return a 1-second multi-frequency test signal at 16 kHz.
    Fs = 16000;
    t  = (0 : 1/Fs : 1 - 1/Fs)';
    x  = sin(2*pi*1000*t) + 0.5*sin(2*pi*5000*t) + 0.01*randn(size(t));
end

% -------------------------------------------------------------------------

function e = bandEnergy(signal, Fs, fLow, fHigh)
%BANDENERGY Compute energy of signal within [fLow, fHigh] Hz.
    N     = length(signal);
    f     = (0 : N-1)' * Fs / N;
    Y     = abs(fft(signal)).^2;
    mask  = (f >= fLow) & (f <= fHigh);
    e     = sum(Y(mask));
end
