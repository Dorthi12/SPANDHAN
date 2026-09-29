function tests = testPreprocessing
    tests = functiontests(localfunctions);
end

function testAudioPreprocessing(testCase)
    signal = sin(2*pi*440*(0:1/44100:1));
    processed = preprocessAudio(signal, 44100);
    testCase.verifyEqual(length(processed), length(signal));
end
