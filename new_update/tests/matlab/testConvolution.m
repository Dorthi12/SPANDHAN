function tests = testConvolution
    tests = functiontests(localfunctions);
end

function testConvOutput(testCase)
    s = [1, 2, 3, 4];
    k = [0.5, 0.5];
    res = runConvolution(s, k);
    testCase.verifyEqual(length(res), length(s));
end
