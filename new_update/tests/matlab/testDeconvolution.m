function tests = testDeconvolution
    tests = functiontests(localfunctions);
end

function testDeconvOutput(testCase)
    s = [1, 2, 3, 4];
    k = [1];
    res = runDeconvolution(s, k);
    testCase.verifyNotEmpty(res);
end
