function tests = testSTFT
    tests = functiontests(localfunctions);
end

function testSTFTOutput(testCase)
    t = 0:1/1000:1;
    signal = sin(2*pi*50*t);
    [S, F, T] = runSTFT(signal, 1000);
    testCase.verifyNotEmpty(S);
end
