function tests = testFIR
    tests = functiontests(localfunctions);
end

function testFIROutput(testCase)
    Fs = 16000;
    t = (0:1/Fs:0.1)';
    signal = sin(2*pi*1000*t) + 0.5*sin(2*pi*5000*t);
    
    result = runFIR(signal, Fs, 100, 3000, "low", "hamming");
    
    testCase.verifyEqual(length(result.output), length(signal));
    testCase.verifyEqual(result.filterType, "low");
    testCase.verifyEqual(result.order, 100);
end

function testFIRBandpass(testCase)
    Fs = 16000;
    t = (0:1/Fs:0.1)';
    signal = sin(2*pi*1000*t);
    
    result = runFIR(signal, Fs, 100, [500, 2000], "bandpass", "blackman");
    
    testCase.verifyEqual(length(result.output), length(signal));
    testCase.verifyEqual(result.filterType, "bandpass");
end

function testFIRResultFields(testCase)
    Fs = 8000;
    signal = randn(500, 1);
    result = runFIR(signal, Fs, 50, 1000, "high", "hann");
    
    requiredFields = {'input', 'output', 'coefficients', 'denominator', 'order', ...
                      'filterType', 'cutoffFrequency', 'normalizedCutoff', 'window', ...
                      'frequency', 'response', 'magnitude', 'magnitudeDB', 'phase', ...
                      'groupDelay', 'groupDelayFrequency', 'impulseResponse', ...
                      'impulseTime', 'peakFrequency', 'peakMagnitude', ...
                      'estimated3dBCutoff', 'samplingFrequency', 'nyquistFrequency', ...
                      'responsePoints'};
    
    for i = 1:length(requiredFields)
        testCase.verifyTrue(isfield(result, requiredFields{i}), ...
            sprintf('Missing field: %s', requiredFields{i}));
    end
end
