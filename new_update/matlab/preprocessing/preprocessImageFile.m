function [I, info] = preprocessImageFile(I, targetHeight, targetWidth)
%PREPROCESSIMAGEFILE  Core SPANDHAN image conditioner.
%
%   Applies signal-preserving standardization to one image.
%   No DSP analysis (FFT/filtering/edge-detection) is performed.
%   Those belong in the Python ML pipeline and MATLAB DSP stage.
%
%   Inputs
%   ------
%   I             : Raw image matrix (uint8 or double, 1 or 3 channels)
%   targetHeight  : Desired output height in pixels  (e.g. 128)
%   targetWidth   : Desired output width  in pixels  (e.g. 128)
%
%   Outputs
%   -------
%   I             : Processed image  [targetHeight x targetWidth]  double [0,1]
%   info          : Struct with processing metadata (see fields below)
%
%   Processing steps
%   ----------------
%   1.  Record original metadata
%   2.  Convert RGB / RGBA  →  grayscale (if needed)
%   3.  Convert to double via  double(I)/255  (NOT rescale / im2double alone)
%       Reason: im2double on uint8 divides by 255 which is what we want,
%               but we call double() explicitly for clarity and then /255
%               only when the original dtype was uint8.
%   4.  Handle NaN / Inf pixels  (set to 0)
%   5.  Clamp to [0,1]
%   6.  Resize if necessary  (aspect-ratio-preserving + centre pad)
%   7.  Final clamp to [0,1]  (bicubic can overshoot slightly)
%   8.  Record processed metadata
%
%   What this function does NOT do (intentional):
%     - Per-image min-max rescale  (destroys absolute intensity)
%     - Gaussian / median denoising
%     - Histogram equalization / CLAHE
%     - Edge detection or sharpening
%     - Any frequency-domain modification
%
%   ============================================================

%% ------------------------------------------------------------
%  STEP 0: RECORD ORIGINAL METADATA
% ------------------------------------------------------------

originalSize = size(I);

info.originalHeight   = originalSize(1);
info.originalWidth    = originalSize(2);
info.originalDatatype = class(I);

if ndims(I) == 3
    info.originalChannels = size(I, 3);
else
    info.originalChannels = 1;
end

I_double_raw = double(I);
info.originalMin = min(I_double_raw(:));
info.originalMax = max(I_double_raw(:));

% Initialise flags
info.wasGrayscaleConverted = false;
info.wasPadded             = false;
info.wasResized            = false;
info.invalidPixels         = 0;

%% ------------------------------------------------------------
%  STEP 1: CONVERT TO GRAYSCALE
%  rgb2gray uses ITU-R BT.601 luminance weights:
%    L = 0.299*R + 0.587*G + 0.114*B
%  This preserves relative brightness rather than simply
%  averaging channels.
% ------------------------------------------------------------

if ndims(I) == 3

    % Handle RGBA: drop alpha channel first
    if size(I, 3) == 4
        I = I(:, :, 1:3);
    end

    I = rgb2gray(I);
    info.wasGrayscaleConverted = true;

end

%% ------------------------------------------------------------
%  STEP 2: CONVERT TO DOUBLE [0,1]
%
%  Use double(I)/255 when the source is uint8.
%  Use double(I)     when the source is already floating-point.
%
%  We do NOT use rescale(I) here because rescale performs
%  per-image min-max normalization, which destroys the absolute
%  intensity relationship between images.
%
%  Example of what rescale would do (wrong):
%    image A: original range [0.2, 0.7] → rescale → [0.0, 1.0]
%    image B: original range [0.0, 1.0] → rescale → [0.0, 1.0]
%  Both become identical in range, losing amplitude information.
% ------------------------------------------------------------

if strcmp(info.originalDatatype, 'uint8')
    I = double(I) / 255;
elseif strcmp(info.originalDatatype, 'uint16')
    I = double(I) / 65535;
else
    % double, single, logical, etc.
    I = double(I);
end

%% ------------------------------------------------------------
%  STEP 3: HANDLE NaN / Inf
%  Safety net for corrupted pixels. Replace with 0 (background).
% ------------------------------------------------------------

invalidMask        = ~isfinite(I);
info.invalidPixels = sum(invalidMask(:));

if info.invalidPixels > 0
    I(invalidMask) = 0;
    if info.invalidPixels > 10
        warning('preprocessImageFile: %d invalid pixels replaced with 0.', ...
            info.invalidPixels);
    end
end

%% ------------------------------------------------------------
%  STEP 4: CLAMP TO [0,1]
%  Defensive clamp; values outside [0,1] should not exist after
%  proper uint8→double conversion, but guard against edge cases.
% ------------------------------------------------------------

I = min(max(I, 0), 1);

%% ------------------------------------------------------------
%  STEP 5: RESIZE IF NECESSARY
%
%  Strategy: aspect-ratio-preserving resize + centre-pad.
%
%  For the current SPANDHAN dataset all images are 128x128, so
%  this branch will not execute during normal preprocessing.
%  It exists to make the pipeline robust to future inputs such
%  as user-uploaded images with arbitrary dimensions.
%
%  Pad colour: 0 (black). For non-uniform background images
%  a border-reflection pad would be better, but zero is safe
%  because all five signal classes use the full image area.
% ------------------------------------------------------------

[currentH, currentW] = size(I);

if currentH ~= targetHeight || currentW ~= targetWidth

    info.wasResized = true;

    %% Scale to fit within target box while preserving aspect ratio
    scaleH = targetHeight / currentH;
    scaleW = targetWidth  / currentW;
    scale  = min(scaleH, scaleW);

    newH = round(currentH * scale);
    newW = round(currentW * scale);

    % Bicubic resampling — smooth for natural images; keeps aliasing
    % minimal without destroying spatial-frequency content.
    I = imresize(I, [newH newW], 'bicubic');

    %% Centre-pad to exact target size
    padTop  = floor((targetHeight - newH) / 2);
    padBot  = targetHeight - newH - padTop;
    padLeft = floor((targetWidth  - newW) / 2);
    padRight = targetWidth - newW - padLeft;

    I = padarray(I, [padTop  padLeft],  0, 'pre');
    I = padarray(I, [padBot  padRight], 0, 'post');

    info.wasPadded = (padTop > 0 || padBot > 0 || ...
                      padLeft > 0 || padRight > 0);

    % Safety clamp to exact target size (rounding edge case)
    I = I(1:targetHeight, 1:targetWidth);

end

%% ------------------------------------------------------------
%  STEP 6: FINAL CLAMP
%  Bicubic interpolation can produce values slightly outside [0,1]
%  (Gibbs-like ringing at sharp edges). Clamp them.
% ------------------------------------------------------------

I = min(max(I, 0), 1);

%% ------------------------------------------------------------
%  FINAL METADATA
% ------------------------------------------------------------

info.processedHeight   = size(I, 1);
info.processedWidth    = size(I, 2);
info.processedDatatype = class(I);    % should always be 'double'

info.processedMin  = min(I(:));
info.processedMax  = max(I(:));
info.processedMean = mean(I(:));
info.processedStd  = std(I(:));

end   % preprocessImageFile
