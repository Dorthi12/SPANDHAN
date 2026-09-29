function tests = testConvolution
%TESTCONVOLUTION Unit tests for runConvolution (SPANDHAN DSP module).
    tests = functiontests(localfunctions);
end

% =========================================================================
%  OUTPUT LENGTH TESTS
% =========================================================================

function testFullOutputLength(testCase)
%Full convolution must produce Nx+Nh-1 samples.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(r.outputLength, length(x) + length(h) - 1);
end

% -------------------------------------------------------------------------

function testSameOutputLength(testCase)
%"same" output must have the same length as x.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "same", "direct");
    testCase.verifyEqual(r.outputLength, length(x));
end

% -------------------------------------------------------------------------

function testValidOutputLength(testCase)
%"valid" output length = max(Nx,Nh) - min(Nx,Nh) + 1.
    Fs = 16000;
    x  = randn(1000, 1);
    h  = randn(200,  1);
    r  = runConvolution(x, h, Fs, "valid", "direct");
    expected = max(length(x), length(h)) - min(length(x), length(h)) + 1;
    testCase.verifyEqual(r.outputLength, expected);
end

% =========================================================================
%  METHOD EQUIVALENCE TESTS
% =========================================================================

function testDirectAndFFTAgreement(testCase)
%Direct and FFT methods must produce numerically identical results (full mode).
    [x, h, Fs] = makeSignals();
    rDirect = runConvolution(x, h, Fs, "full", "direct");
    rFFT    = runConvolution(x, h, Fs, "full", "fft");
    testCase.verifyEqual(rDirect.output, rFFT.output, "AbsTol", 1e-9, ...
        "Direct and FFT convolution results must match to within 1e-9.");
end

% -------------------------------------------------------------------------

function testAutoMethodRuns(testCase)
%"auto" must resolve to either "direct" or "fft" without error.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "auto");
    testCase.verifyTrue( ...
        r.method == "direct" || r.method == "fft", ...
        '"auto" must resolve to "direct" or "fft".');
end

% =========================================================================
%  IMPULSE RESPONSE IDENTITY TEST  (key SPANDHAN sanity check)
% =========================================================================

function testImpulseIdentity(testCase)
%delta[n] * h[n] = h[n].
%Convolving any h with a unit impulse at index 1 must return h exactly.
    Fs = 16000;
    h  = exp(-0.01 * (0:200)');
    x  = zeros(500, 1);
    x(1) = 1;                           % unit impulse

    r = runConvolution(x, h, Fs, "full", "direct");

    % The first Nh samples of the output must equal h
    testCase.verifyEqual(r.output(1:length(h)), h, "AbsTol", 1e-12, ...
        "delta * h must equal h (identity property of convolution).");
end

% =========================================================================
%  RESULT STRUCT FIELD TESTS
% =========================================================================

function testAllFieldsPresent(testCase)
%Every documented output field must be present in the result struct.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "auto");

    requiredFields = { ...
        'input', 'impulseResponse', 'output', ...
        'inputTime', 'impulseTime', 'time', ...
        'samplingFrequency', 'outputMode', 'method', ...
        'inputLength', 'impulseResponseLength', ...
        'fullOutputLength', 'outputLength', ...
        'inputEnergy', 'impulseResponseEnergy', 'outputEnergy', ...
        'inputRMS', 'impulseResponseRMS', 'outputRMS', ...
        'inputPeak', 'impulseResponsePeak', 'outputPeak', ...
        'inputPeakTime', 'impulsePeakTime', 'outputPeakTime', ...
        'computationTime', 'fftLength' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(r, requiredFields{k}), ...
            sprintf('Missing field: %s', requiredFields{k}));
    end
end

% =========================================================================
%  SIGNAL PRESERVATION TESTS
% =========================================================================

function testInputPreserved(testCase)
%.input must be the original x as a column vector.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(r.input, x(:));
end

% -------------------------------------------------------------------------

function testImpulseResponsePreserved(testCase)
%.impulseResponse must be the original h as a column vector.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(r.impulseResponse, h(:));
end

% =========================================================================
%  TIME AXIS TESTS
% =========================================================================

function testInputTimeLength(testCase)
%inputTime must have Nx elements.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(numel(r.inputTime), r.inputLength);
end

% -------------------------------------------------------------------------

function testOutputTimeLength(testCase)
%time must have the same number of elements as the output.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(numel(r.time), r.outputLength);
end

% =========================================================================
%  ENERGY AND RMS TESTS
% =========================================================================

function testEnergyIsNonNegative(testCase)
%All energy values must be non-negative.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyGreaterThanOrEqual(r.inputEnergy,           0);
    testCase.verifyGreaterThanOrEqual(r.impulseResponseEnergy, 0);
    testCase.verifyGreaterThanOrEqual(r.outputEnergy,          0);
end

% -------------------------------------------------------------------------

function testRMSIsNonNegative(testCase)
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyGreaterThanOrEqual(r.inputRMS,           0);
    testCase.verifyGreaterThanOrEqual(r.impulseResponseRMS, 0);
    testCase.verifyGreaterThanOrEqual(r.outputRMS,          0);
end

% =========================================================================
%  PEAK TESTS
% =========================================================================

function testPeakMatchesMax(testCase)
%Peak values must equal max(abs(signal)).
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(r.inputPeak,  max(abs(x)), "AbsTol", 1e-14);
    testCase.verifyEqual(r.outputPeak, max(abs(r.output)), "AbsTol", 1e-14);
end

% =========================================================================
%  FFT METADATA TEST
% =========================================================================

function testFFTLengthPowerOfTwo(testCase)
%When using the FFT method, fftLength must be a power of 2.
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "fft");
    testCase.verifyEqual(r.fftLength, 2^nextpow2(r.fullOutputLength));
end

% -------------------------------------------------------------------------

function testDirectMethodHasEmptyFFTLength(testCase)
%Direct method must return fftLength = [].
    [x, h, Fs] = makeSignals();
    r = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEmpty(r.fftLength);
end

% =========================================================================
%  ROW-VECTOR INPUT TOLERANCE
% =========================================================================

function testRowVectorInputsAccepted(testCase)
%The function must accept row vectors and internally convert them.
    Fs = 16000;
    x  = sin(2*pi*1000*(0:999)/Fs);    % row vector
    h  = exp(-0.01*(0:99));            % row vector
    r  = runConvolution(x, h, Fs, "full", "direct");
    testCase.verifyEqual(size(r.input,   2), 1, 'x must be stored as column.');
    testCase.verifyEqual(size(r.output,  2), 1, 'y must be stored as column.');
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [x, h, Fs] = makeSignals()
%MAKESIGNALS Standard 1-second test signal and 200-sample system response.
    Fs = 16000;
    t  = (0 : 1/Fs : 1 - 1/Fs)';
    x  = sin(2*pi*1000*t);

    th = (0:199)' / Fs;
    h  = exp(-20*th) .* cos(2*pi*500*th);
    h  = h / max(abs(h));
end
