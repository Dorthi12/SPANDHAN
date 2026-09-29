function tests = testWavelet
%TESTWAVELET Unit tests for runWavelet (SPANDHAN DSP module).
    tests = functiontests(localfunctions);
end

% =========================================================================
%  ALL STRUCT FIELDS PRESENT
% =========================================================================

function testAllFieldsPresent(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);

    requiredFields = { ...
        'input', 'analysisSignal', 'time', ...
        'samplingFrequency', 'waveletName', ...
        'decompositionLevel', 'maximumLevel', ...
        'coefficients', 'bookkeeping', ...
        'approximation', 'details', ...
        'approximationEnergy', 'detailEnergy', ...
        'totalWaveletEnergy', 'approximationEnergyRatio', ...
        'detailEnergyRatio', 'dominantLevel', 'waveletEntropy', ...
        'reconstructedApproximation', 'reconstructedDetails', ...
        'reconstructedSignal', 'reconstructionRMSE', 'reconstructionMAE', ...
        'approximationBand', 'detailBands' ...
    };

    for k = 1:numel(requiredFields)
        testCase.verifyTrue(isfield(r, requiredFields{k}), ...
            sprintf("Missing field: %s", requiredFields{k}));
    end
end

% =========================================================================
%  DETAIL AND APPROXIMATION STRUCTURE
% =========================================================================

function testDetailCellSize(testCase)
%.details must be a cell array of length decompositionLevel.
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    testCase.verifyClass(r.details, 'cell');
    testCase.verifyEqual(numel(r.details), L);
end

% -------------------------------------------------------------------------

function testReconstructedDetailsCellSize(testCase)
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    testCase.verifyClass(r.reconstructedDetails, 'cell');
    testCase.verifyEqual(numel(r.reconstructedDetails), L);
end

% -------------------------------------------------------------------------

function testDetailEnergyVectorLength(testCase)
%detailEnergy must have decompositionLevel elements.
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    testCase.verifyEqual(numel(r.detailEnergy), L);
end

% -------------------------------------------------------------------------

function testDetailEnergyRatioLength(testCase)
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    testCase.verifyEqual(numel(r.detailEnergyRatio), L);
end

% =========================================================================
%  ENERGY CONSISTENCY
% =========================================================================

function testTotalEnergyEqualsSum(testCase)
%totalWaveletEnergy must equal approximation + sum(detail energies).
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    expected = r.approximationEnergy + sum(r.detailEnergy);
    testCase.verifyEqual(r.totalWaveletEnergy, expected, "AbsTol", 1e-9);
end

% -------------------------------------------------------------------------

function testEnergyNonNegative(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyGreaterThanOrEqual(r.approximationEnergy, 0);
    testCase.verifyGreaterThanOrEqual(r.totalWaveletEnergy,  0);
    testCase.verifyTrue(all(r.detailEnergy >= 0));
end

% -------------------------------------------------------------------------

function testEnergyRatiosSumToOne(testCase)
%All energy ratios must sum to 1.
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    total = r.approximationEnergyRatio + sum(r.detailEnergyRatio);
    testCase.verifyEqual(total, 1, "AbsTol", 1e-10, ...
        "Energy ratios must sum to 1.");
end

% =========================================================================
%  RECONSTRUCTION ACCURACY
% =========================================================================

function testReconstructionRMSENearZero(testCase)
%Perfect reconstruction: sum of all reconstructed components ≈ xAnalysis.
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyLessThan(r.reconstructionRMSE, 1e-10, ...
        "DWT perfect reconstruction: RMSE must be < 1e-10.");
end

% -------------------------------------------------------------------------

function testReconstructionMAENearZero(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyLessThan(r.reconstructionMAE, 1e-10);
end

% -------------------------------------------------------------------------

function testReconstructedSignalLength(testCase)
%Reconstructed signal must have the same length as the input.
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(length(r.reconstructedSignal), length(x));
end

% =========================================================================
%  DC REMOVAL
% =========================================================================

function testInputPreserved(testCase)
%.input must be the original x (unchanged).
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(r.input, x(:));
end

% -------------------------------------------------------------------------

function testAnalysisSignalMeanFree(testCase)
    Fs = 16000;
    t  = (0:1/Fs:1-1/Fs)';
    x  = 7.3 + sin(2*pi*1000*t);      % large DC offset
    r  = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(mean(r.analysisSignal), 0, "AbsTol", 1e-10);
end

% =========================================================================
%  DOMINANT LEVEL TESTS
% =========================================================================

function testDominantLevelIsValidIndex(testCase)
%dominantLevel must be an integer in [1, decompositionLevel].
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    if ~isnan(r.dominantLevel)
        testCase.verifyGreaterThanOrEqual(r.dominantLevel, 1);
        testCase.verifyLessThanOrEqual(r.dominantLevel, L);
    end
end

% -------------------------------------------------------------------------

function testImpulseDominantLevelHighFrequency(testCase)
%A unit impulse has wideband energy — dominant level should be D1 or D2
%(highest frequency details) since db4 concentrates transient energy there.
    Fs = 16000;
    x  = zeros(16000, 1);
    x(8000) = 1;

    r = runWavelet(x, Fs, "db4", 5);

    % Impulse energy should be concentrated in high-frequency details
    % (D1 or D2), not in the low-frequency approximation
    testCase.verifyLessThanOrEqual(r.dominantLevel, 3, ...
        "Impulse dominant level should be in high-frequency details (D1–D3).");
end

% =========================================================================
%  WAVELET ENTROPY
% =========================================================================

function testEntropyNonNegative(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyGreaterThanOrEqual(r.waveletEntropy, 0);
end

% -------------------------------------------------------------------------

function testEntropyMaxBound(testCase)
%Shannon entropy over L+1 subbands is at most log2(L+1).
    [x, Fs] = makeSine();
    L = 5;
    r = runWavelet(x, Fs, "db4", L);
    maxEntropy = log2(L + 1);           % uniform distribution upper bound
    testCase.verifyLessThanOrEqual(r.waveletEntropy, maxEntropy + 1e-10);
end

% -------------------------------------------------------------------------

function testWhiteNoiseHigherEntropyThanSine(testCase)
%White noise energy is spread across all scales → higher entropy than sine.
    Fs    = 16000;
    t     = (0:1/Fs:1-1/Fs)';
    xSine = sin(2*pi*1000*t);
    xNoise = randn(size(t));

    rSine  = runWavelet(xSine,  Fs, "db4", 5);
    rNoise = runWavelet(xNoise, Fs, "db4", 5);

    testCase.verifyGreaterThan(rNoise.waveletEntropy, ...
                               rSine.waveletEntropy, ...
        "White noise must have higher wavelet entropy than a pure sine.");
end

% =========================================================================
%  FREQUENCY BAND STRUCTURE
% =========================================================================

function testApproximationBandShape(testCase)
%.approximationBand must be a 2-element row vector [0, upper].
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(size(r.approximationBand), [1, 2]);
    testCase.verifyEqual(r.approximationBand(1), 0);
    testCase.verifyGreaterThan(r.approximationBand(2), 0);
end

% -------------------------------------------------------------------------

function testDetailBandsShape(testCase)
%.detailBands must be decompositionLevel × 2.
    [x, Fs] = makeSine();
    L = 4;
    r = runWavelet(x, Fs, "db4", L);
    testCase.verifyEqual(size(r.detailBands), [L, 2]);
end

% -------------------------------------------------------------------------

function testDetailBandLowLessThanHigh(testCase)
%Lower band edge must be less than upper band edge for every level.
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    for k = 1:r.decompositionLevel
        testCase.verifyLessThan(r.detailBands(k,1), r.detailBands(k,2), ...
            sprintf("Detail band %d: lower >= upper frequency.", k));
    end
end

% =========================================================================
%  CONFIGURATION STORED CORRECTLY
% =========================================================================

function testWaveletNameStored(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "sym4", 4);
    testCase.verifyEqual(r.waveletName, "sym4");
end

% -------------------------------------------------------------------------

function testDecompositionLevelStored(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 4);
    testCase.verifyEqual(r.decompositionLevel, 4);
end

% -------------------------------------------------------------------------

function testSamplingFrequencyStored(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(r.samplingFrequency, Fs);
end

% =========================================================================
%  LEVEL CLAMPING
% =========================================================================

function testExcessiveLevelIsClamped(testCase)
%Requesting too many levels must not error — it should be clamped.
    Fs  = 16000;
    x   = randn(64, 1);               % very short signal
    r   = runWavelet(x, Fs, "db4", 100);   % unreachably high level
    testCase.verifyLessThanOrEqual(r.decompositionLevel, r.maximumLevel, ...
        "decompositionLevel must be ≤ maximumLevel after clamping.");
end

% =========================================================================
%  DEFAULT ARGUMENT HANDLING
% =========================================================================

function testDefaultsApplied(testCase)
%Calling with only x and Fs must use db4 and level 5 (or clamped).
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs);             % only 2 args
    testCase.verifyEqual(r.waveletName, "db4");
    testCase.verifyGreaterThanOrEqual(r.decompositionLevel, 1);
end

% =========================================================================
%  ROW-VECTOR INPUT
% =========================================================================

function testRowVectorInputAccepted(testCase)
    Fs = 16000;
    x  = sin(2*pi*1000*(0:Fs-1)/Fs);  % row vector
    r  = runWavelet(x, Fs, "db4", 4);
    testCase.verifyEqual(size(r.input, 2), 1, "input must be stored as column.");
end

% =========================================================================
%  TIME VECTOR
% =========================================================================

function testTimeVectorLength(testCase)
    [x, Fs] = makeSine();
    r = runWavelet(x, Fs, "db4", 5);
    testCase.verifyEqual(numel(r.time), length(x));
end

% =========================================================================
%  CHIRP — MULTI-SCALE DISTRIBUTION
% =========================================================================

function testChirpEnergySpreadAcrossLevels(testCase)
%A chirp sweeping 500→5000 Hz must have energy in multiple detail levels,
%not concentrated in a single level.
    Fs  = 16000;
    dur = 2;
    t   = (0:1/Fs:dur-1/Fs)';
    x   = chirp(t, 500, dur, 5000, "linear");

    r = runWavelet(x, Fs, "db4", 5);

    % Count levels that carry > 5 % of detail energy
    activeDetailLevels = sum(r.detailEnergyRatio > 0.05);
    testCase.verifyGreaterThan(activeDetailLevels, 1, ...
        "Chirp energy should be spread across more than one detail level.");
end

% =========================================================================
%  LOCAL HELPER
% =========================================================================

function [x, Fs] = makeSine()
%MAKESINE 1-second 1 kHz sinusoidal test signal at 16 kHz.
    Fs = 16000;
    t  = (0:1/Fs:1-1/Fs)';
    x  = sin(2*pi*1000*t);
end
