function visualizePreprocessing(xRaw, xProcessed, rawFs, processedFs, ...
                                filename, className, plotsDir, plotIndex)
%VISUALIZEPREPROCESSING  QA plot: before vs after preprocessing.
%
%   Generates a 3×2 figure comparing:
%     Row 1: Time-domain waveform (raw vs processed)
%     Row 2: Amplitude histogram  (raw vs processed)
%     Row 3: Power spectrum       (raw vs processed)
%
%   Inputs
%   ------
%   xRaw        : Original audio signal (may be multi-channel)
%   xProcessed  : Preprocessed output (mono, length = targetSamples)
%   rawFs       : Original sampling frequency
%   processedFs : Target sampling frequency
%   filename    : WAV filename string (for title)
%   className   : Signal class string
%   plotsDir    : Directory to save PNG
%   plotIndex   : Integer index for filename uniqueness
%
%   ============================================================

%% Flatten raw to mono for comparison plot
if size(xRaw, 2) > 1
    xRaw = mean(xRaw, 2);
end
xRaw = xRaw(:);

tRaw  = (0:numel(xRaw)  -1)' / rawFs;
tProc = (0:numel(xProcessed)-1)' / processedFs;

%% ----------------------------------------------------------------
%  FIGURE
% -----------------------------------------------------------------

fig = figure('Visible', 'off', 'Position', [50 50 1200 700]);

sgtitle(sprintf('SPANDHAN | %s | %s', className, filename), ...
    'FontWeight', 'bold', 'FontSize', 11);

%% Row 1: Time domain ------------------------------------------

subplot(3, 2, 1);
plot(tRaw, xRaw, 'b', 'LineWidth', 0.6);
xlabel('Time (s)');  ylabel('Amplitude');
title(sprintf('RAW  [%d Hz | %.2f s | %d ch]', ...
    rawFs, tRaw(end), size(xRaw, 2)));
grid on;  xlim([tRaw(1) tRaw(end)]);

subplot(3, 2, 2);
plot(tProc, xProcessed, 'r', 'LineWidth', 0.6);
xlabel('Time (s)');  ylabel('Amplitude');
title(sprintf('PROCESSED  [%d Hz | %.2f s]', processedFs, tProc(end)));
grid on;  xlim([tProc(1) tProc(end)]);

%% Row 2: Histogram --------------------------------------------

subplot(3, 2, 3);
histogram(xRaw, 60, 'FaceColor', [0.2 0.4 0.8], 'EdgeColor', 'none');
xlabel('Amplitude');  ylabel('Count');
title(sprintf('RAW histogram  (peak=%.3f, dc=%.4f)', ...
    max(abs(xRaw)), mean(xRaw)));
grid on;

subplot(3, 2, 4);
histogram(xProcessed, 60, 'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'none');
xlabel('Amplitude');  ylabel('Count');
title(sprintf('PROCESSED histogram  (peak=%.3f, dc=%.4f)', ...
    max(abs(xProcessed)), mean(xProcessed)));
grid on;

%% Row 3: Power spectrum (magnitude via FFT) -------------------

% Raw power spectrum
NRaw     = numel(xRaw);
fftRaw   = fft(xRaw);
PSD_raw  = (1/(rawFs*NRaw)) * abs(fftRaw(1:floor(NRaw/2)+1)).^2;
PSD_raw(2:end-1) = 2*PSD_raw(2:end-1);
fVecRaw  = (0:floor(NRaw/2)) * (rawFs/NRaw);

subplot(3, 2, 5);
plot(fVecRaw, 10*log10(PSD_raw + eps), 'b', 'LineWidth', 0.6);
xlabel('Frequency (Hz)');  ylabel('Power (dB)');
title('RAW power spectrum');
grid on;  xlim([0 rawFs/2]);

% Processed power spectrum
NProc    = numel(xProcessed);
fftProc  = fft(xProcessed);
PSD_proc = (1/(processedFs*NProc)) * abs(fftProc(1:floor(NProc/2)+1)).^2;
PSD_proc(2:end-1) = 2*PSD_proc(2:end-1);
fVecProc = (0:floor(NProc/2)) * (processedFs/NProc);

subplot(3, 2, 6);
plot(fVecProc, 10*log10(PSD_proc + eps), 'r', 'LineWidth', 0.6);
xlabel('Frequency (Hz)');  ylabel('Power (dB)');
title('PROCESSED power spectrum');
grid on;  xlim([0 processedFs/2]);

%% ----------------------------------------------------------------
%  SAVE
% -----------------------------------------------------------------

[~, baseName, ~] = fileparts(filename);
outPath = fullfile(plotsDir, ...
    sprintf('%s_%s_%02d.png', className, baseName, plotIndex));

exportgraphics(fig, outPath, 'Resolution', 150);
close(fig);

end
