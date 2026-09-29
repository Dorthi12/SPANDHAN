function visualizeImagePreprocessing(Ioriginal, Iprocessed, ...
    filename, className, plotsDir, plotIndex)
%VISUALIZEIMAGEPREPROCESSING  Save a side-by-side QA figure for one image.
%
%   Generates a 2-column figure showing the original and processed image
%   side-by-side, with a pixel intensity histogram for each.
%
%   Inputs
%   ------
%   Ioriginal   : Original image as loaded by imread  (uint8 or double)
%   Iprocessed  : Processed image [0,1] double
%   filename    : Source filename string (for figure title)
%   className   : Class label string
%   plotsDir    : Folder to save the figure PNG
%   plotIndex   : Integer index within the class (1, 2, 3 ...)
%
%   Output
%   ------
%   Saves a PNG figure to:
%     plotsDir / <className>_qa_<plotIndex>.png
%
%   ============================================================

fig = figure('Visible', 'off', 'Position', [100 100 1000 420]);

%% Convert original to double for display
if strcmp(class(Ioriginal), 'uint8')
    Idisplay = double(Ioriginal) / 255;
else
    Idisplay = double(Ioriginal);
end
% If original was RGB/RGBA, show grayscale version
if ndims(Idisplay) == 3
    Idisplay = rgb2gray(uint8(Idisplay * 255));
    Idisplay = double(Idisplay) / 255;
end

%% ------- Column 1: Original ---------------------------------

subplot(2, 2, 1);
imshow(Idisplay, [0 1]);
title(sprintf('Original  [%dx%d]', size(Idisplay,1), size(Idisplay,2)), ...
    'FontSize', 9, 'Interpreter', 'none');
xlabel(sprintf('Min=%.3f  Max=%.3f', min(Idisplay(:)), max(Idisplay(:))), ...
    'FontSize', 8);

subplot(2, 2, 3);
histogram(Idisplay(:), 64, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
xlabel('Pixel intensity [0,1]', 'FontSize', 8);
ylabel('Count', 'FontSize', 8);
title('Original histogram', 'FontSize', 9);
xlim([0 1]);
grid on;

%% ------- Column 2: Processed --------------------------------

subplot(2, 2, 2);
imshow(Iprocessed, [0 1]);
title(sprintf('Processed  [%dx%d]', size(Iprocessed,1), size(Iprocessed,2)), ...
    'FontSize', 9, 'Interpreter', 'none');
xlabel(sprintf('Min=%.3f  Max=%.3f  Mean=%.3f', ...
    min(Iprocessed(:)), max(Iprocessed(:)), mean(Iprocessed(:))), ...
    'FontSize', 8);

subplot(2, 2, 4);
histogram(Iprocessed(:), 64, 'FaceColor', [0.2 0.7 0.4], 'EdgeColor', 'none');
xlabel('Pixel intensity [0,1]', 'FontSize', 8);
ylabel('Count', 'FontSize', 8);
title('Processed histogram', 'FontSize', 9);
xlim([0 1]);
grid on;

%% ------- Title and save -------------------------------------

sgtitle(sprintf('Class: %s  |  %s', className, filename), ...
    'FontSize', 10, 'Interpreter', 'none', 'FontWeight', 'bold');

outputPath = fullfile(plotsDir, ...
    sprintf('%s_qa_%02d.png', className, plotIndex));

exportgraphics(fig, outputPath, 'Resolution', 120);
close(fig);

end
