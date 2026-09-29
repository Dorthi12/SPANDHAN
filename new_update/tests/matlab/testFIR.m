function tests = testFIR
    tests = functiontests(localfunctions);
end

function testFIROutput(testCase)
    signal = randn(1, 1000);
    filtered = runFIR(signal, 100, 1000);
    testCase.verifyEqual(length(filtered), length(signal));
end
