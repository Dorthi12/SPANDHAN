function tests = testClassifyImage
% TESTCLASSIFYIMAGE Unit tests for classifyImage MATLAB-Python bridge.

    tests = functiontests(localfunctions);
end

function testClassifyImageStructure(testCase)
    % Generate synthetic 128x128 sinusoidal pattern in [0, 1]
    [X, ~] = meshgrid(1:128, 1:128);
    dummyImg = 0.5 + 0.5 * sin(2 * pi * 0.05 * X);

    result = classifyImage(dummyImg);

    % Verify contract fields
    testCase.verifyTrue(isfield(result, 'class'));
    testCase.verifyTrue(isfield(result, 'class_id'));
    testCase.verifyTrue(isfield(result, 'confidence'));
    testCase.verifyTrue(isfield(result, 'probabilities'));

    % Verify types and values
    testCase.verifyTrue(isstring(result.class) || ischar(result.class));
    testCase.verifyTrue(isnumeric(result.class_id));
    testCase.verifyGreaterThanOrEqual(result.confidence, 0.0);
    testCase.verifyLessThanOrEqual(result.confidence, 1.0);

    % Verify probability distribution
    probs = result.probabilities;
    testCase.verifyTrue(isstruct(probs));
    testCase.verifyTrue(isfield(probs, 'impulse'));
    testCase.verifyTrue(isfield(probs, 'sinusoidal'));
    testCase.verifyTrue(isfield(probs, 'white_noise'));
    testCase.verifyTrue(isfield(probs, 'step'));
    testCase.verifyTrue(isfield(probs, 'chirp'));
end

function testClassifyImageWhiteNoise(testCase)
    % Generate synthetic 128x128 uniform noise in [0, 1]
    rng(42);
    noiseImg = rand(128, 128);

    result = classifyImage(noiseImg);
    testCase.verifyNotEmpty(result.class);
    testCase.verifyGreaterThan(result.confidence, 0.5);
end
