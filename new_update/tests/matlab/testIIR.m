function tests = testIIR
    tests = functiontests(localfunctions);
end

function testIIROutput(testCase)
    signal = randn(1, 1000);
    filtered = runIIR(signal, 100, 1000);
    testCase.verifyEqual(length(filtered), length(signal));
end
