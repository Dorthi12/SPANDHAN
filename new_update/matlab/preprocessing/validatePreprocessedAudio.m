function validatePreprocessedAudio(metadata, outputDir, targetFs, targetSamples)
%VALIDATEPREPROCESSEDAUDIO  QA checks on all processed files.
%
%   Reads every processed WAV listed in metadata and verifies:
%     1. File exists on disk
%     2. Sample count equals targetSamples
%     3. Sampling rate equals targetFs
%     4. No NaN / Inf samples
%     5. Peak amplitude in (0, 1] (normalized)
%     6. DC offset is small (|mean| < 0.01)
%
%   Inputs
%   ------
%   metadata      : Table produced by preprocessAudioDataset.m
%   outputDir     : Path to datasets/audio_processed/
%   targetFs      : Expected sampling frequency (Hz)
%   targetSamples : Expected sample count
%
%   Prints a summary table and logs any failures.
%
%   ============================================================

fprintf('\n------------------------------------------------------------\n');
fprintf(' QA VALIDATION\n');
fprintf('------------------------------------------------------------\n');

CLASSES = unique(metadata.class);

totalFiles   = 0;
totalPassed  = 0;
totalFailed  = 0;
failureLog   = {};

for c = 1:numel(CLASSES)

    className = CLASSES{c};
    classRows = metadata(strcmp(metadata.class, className) & ...
                         ~metadata.wasSkipped, :);

    classPass = 0;
    classFail = 0;

    for k = 1:height(classRows)

        row      = classRows(k, :);
        wavPath  = fullfile(outputDir, className, row.filename{1});
        totalFiles = totalFiles + 1;

        %% Check 1: File exists
        if ~isfile(wavPath)
            failureLog{end+1} = sprintf('MISSING  | %s/%s', ...
                className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Read back the processed file
        try
            [xCheck, fsCheck] = audioread(wavPath);
        catch
            failureLog{end+1} = sprintf('UNREADABLE | %s/%s', ...
                className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 2: Sample count
        if numel(xCheck) ~= targetSamples
            failureLog{end+1} = sprintf( ...
                'WRONG_LENGTH %d (expected %d) | %s/%s', ...
                numel(xCheck), targetSamples, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 3: Sampling rate
        if fsCheck ~= targetFs
            failureLog{end+1} = sprintf( ...
                'WRONG_FS %d (expected %d) | %s/%s', ...
                fsCheck, targetFs, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 4: No NaN / Inf
        if any(~isfinite(xCheck))
            failureLog{end+1} = sprintf( ...
                'HAS_NAN_INF | %s/%s', className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 5: Peak normalized
        peakAbs = max(abs(xCheck));
        if peakAbs <= 0 || peakAbs > 1.001    % tiny tolerance for float
            failureLog{end+1} = sprintf( ...
                'BAD_PEAK %.4f | %s/%s', peakAbs, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 6: DC offset small
        dcVal = abs(mean(xCheck));
        if dcVal > 0.01
            failureLog{end+1} = sprintf( ...
                'HIGH_DC %.4f | %s/%s', dcVal, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        classPass = classPass + 1;

    end   % file loop

    totalPassed = totalPassed + classPass;
    totalFailed = totalFailed + classFail;

    fprintf('  %-12s  PASS: %4d  |  FAIL: %4d\n', ...
        className, classPass, classFail);

end   % class loop

%% ----------------------------------------------------------------
%  PRINT FAILURE LOG
% -----------------------------------------------------------------

fprintf('\n  TOTAL  |  PASS: %4d  |  FAIL: %4d  |  Files: %4d\n', ...
    totalPassed, totalFailed, totalFiles);

if ~isempty(failureLog)
    fprintf('\n  [FAILURES]\n');
    for i = 1:numel(failureLog)
        fprintf('    %s\n', failureLog{i});
    end
end

%% ----------------------------------------------------------------
%  SAVE VALIDATION REPORT
% -----------------------------------------------------------------

reportFile = fullfile(outputDir, 'qa_validation_report.txt');
fid = fopen(reportFile, 'w');

fprintf(fid, 'SPANDHAN Audio Preprocessing QA Report\n');
fprintf(fid, 'Generated : %s\n\n', datetime('now'));
fprintf(fid, 'Target FS      : %d Hz\n', targetFs);
fprintf(fid, 'Target Samples : %d\n\n', targetSamples);
fprintf(fid, 'TOTAL   PASS: %d  FAIL: %d  Files: %d\n\n', ...
    totalPassed, totalFailed, totalFiles);

if isempty(failureLog)
    fprintf(fid, 'ALL CHECKS PASSED.\n');
else
    fprintf(fid, 'FAILURES:\n');
    for i = 1:numel(failureLog)
        fprintf(fid, '  %s\n', failureLog{i});
    end
end

fclose(fid);

fprintf('\n  QA report saved: %s\n', reportFile);
fprintf('------------------------------------------------------------\n');

end
