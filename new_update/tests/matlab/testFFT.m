function tests = testFFT
    tests = functiontests(localfunctions);
end

function testFFTOutput(testCase)
    t = 0:1/1000:1;
    signal = sin(2*pi*50*t);
    [f, Y] = runFFT(signal, 1000);
    testCase.verifyNotEmpty(f);
    testCase.verifyNotEmpty(Y);
end
