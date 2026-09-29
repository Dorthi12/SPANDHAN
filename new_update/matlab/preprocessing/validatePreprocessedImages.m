function validatePreprocessedImages(metadata, outputDir, targetHeight, targetWidth)
%VALIDATEPREPROCESSEDIMAGES  QA checks on all processed PNG files.
%
%   Reads every processed PNG listed in metadata and verifies:
%     1. File exists on disk
%     2. Dimensions equal  targetHeight x targetWidth
%     3. Image is single-channel (grayscale)
%     4. No NaN / Inf pixels
%     5. Pixel values in [0, 255]  (uint8 stored)
%     6. Image is not entirely black  (processedMax > 0)
%
%   Inputs
%   ------
%   metadata      : Table produced by preprocessImageDataset.m
%   outputDir     : Path to datasets/image_processed/
%   targetHeight  : Expected image height in pixels  (e.g. 128)
%   targetWidth   : Expected image width  in pixels  (e.g. 128)
%
%   Prints a per-class summary and logs all failures.
%   Saves a QA report text file to outputDir.
%
%   ============================================================

fprintf('\n------------------------------------------------------------\n');
fprintf(' QA VALIDATION\n');
fprintf('------------------------------------------------------------\n');

CLASSES = unique(metadata.class);

totalFiles  = 0;
totalPassed = 0;
totalFailed = 0;
failureLog  = {};

for c = 1:numel(CLASSES)

    className = CLASSES{c};
    classRows = metadata( ...
        strcmp(metadata.class, className) & ~metadata.wasSkipped, :);

    classPass = 0;
    classFail = 0;

    for k = 1:height(classRows)

        row      = classRows(k, :);
        pngPath  = fullfile(outputDir, className, row.filename{1});
        totalFiles = totalFiles + 1;

        %% Check 1: File exists
        if ~isfile(pngPath)
            failureLog{end+1} = sprintf('MISSING      | %s/%s', ...
                className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Read back the processed file
        try
            I = imread(pngPath);
        catch
            failureLog{end+1} = sprintf('UNREADABLE   | %s/%s', ...
                className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 2: Dimensions
        [h, w, ch] = size(I);
        if ndims(I) == 2
            ch = 1;
        end

        if h ~= targetHeight || w ~= targetWidth
            failureLog{end+1} = sprintf( ...
                'WRONG_SIZE   %dx%d (expected %dx%d) | %s/%s', ...
                h, w, targetHeight, targetWidth, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 3: Single channel
        if ch ~= 1
            failureLog{end+1} = sprintf( ...
                'WRONG_CH     %d channels (expected 1) | %s/%s', ...
                ch, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 4: No NaN / Inf
        %  imread returns uint8 which cannot be NaN/Inf, but check anyway
        %  (covers edge case of non-uint8 PNG).
        Idouble = double(I);
        if any(~isfinite(Idouble(:)))
            failureLog{end+1} = sprintf( ...
                'HAS_NAN_INF  | %s/%s', className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 5: Pixel range [0, 255]
        pxMin = min(Idouble(:));
        pxMax = max(Idouble(:));
        if pxMin < 0 || pxMax > 255
            failureLog{end+1} = sprintf( ...
                'BAD_RANGE    [%.2f, %.2f] | %s/%s', ...
                pxMin, pxMax, className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        %% Check 6: Not entirely black
        if pxMax == 0
            failureLog{end+1} = sprintf( ...
                'ALL_BLACK    | %s/%s', className, row.filename{1});
            classFail = classFail + 1;
            continue;
        end

        classPass = classPass + 1;

    end  % file loop

    totalPassed = totalPassed + classPass;
    totalFailed = totalFailed + classFail;

    fprintf('  %-14s  PASS: %4d  |  FAIL: %4d\n', ...
        className, classPass, classFail);

end  % class loop

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
%  CLASS COUNT VALIDATION
%  Confirm each class has exactly 1000 processed files.
% -----------------------------------------------------------------

fprintf('\n  [CLASS COUNTS]\n');
EXPECTED_PER_CLASS = 1000;
countOk = true;

for c = 1:numel(CLASSES)
    className = CLASSES{c};
    classDir  = fullfile(outputDir, className);
    pngFiles  = dir(fullfile(classDir, '*.png'));
    n         = numel(pngFiles);
    status    = 'OK';
    if n ~= EXPECTED_PER_CLASS
        status   = 'MISMATCH';
        countOk  = false;
    end
    fprintf('  %-14s  %4d / %4d   [%s]\n', ...
        className, n, EXPECTED_PER_CLASS, status);
end

if countOk
    fprintf('\n  All class counts correct (%d x 5 = %d total).\n', ...
        EXPECTED_PER_CLASS, EXPECTED_PER_CLASS * numel(CLASSES));
else
    fprintf('\n  [WARN] Class count mismatch — inspect the log above.\n');
end

%% ----------------------------------------------------------------
%  SAVE VALIDATION REPORT
% -----------------------------------------------------------------

reportFile = fullfile(outputDir, 'image_qa_validation_report.txt');
fid = fopen(reportFile, 'w');

fprintf(fid, 'SPANDHAN Image Preprocessing QA Report\n');
fprintf(fid, 'Generated : %s\n\n', datetime('now'));
fprintf(fid, 'Target Height : %d px\n', targetHeight);
fprintf(fid, 'Target Width  : %d px\n', targetWidth);
fprintf(fid, 'Target Channels: 1 (grayscale)\n\n');
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
