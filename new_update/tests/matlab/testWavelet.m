function tests = testWavelet
    tests = functiontests(localfunctions);
end

function testWaveletOutput(testCase)
    signal = randn(1, 512);
    [c, l] = runWavelet(signal, 'db4');
    testCase.verifyNotEmpty(c);
end
