function [x, info] = preprocessAudioFile(x, originalFs, targetFs, targetSamples, className)
%PREPROCESSAUDIOFILE  Core SPANDHAN audio conditioner.
%
%   Applies signal-preserving standardization to one audio file.
%   No DSP analysis (FFT/STFT/filters) is performed here.
%   Those belong in the Python ML pipeline and MATLAB DSP stage.
%
%   Inputs
%   ------
%   x             : Raw audio matrix  [samples × channels]
%   originalFs    : Original sampling frequency (Hz)
%   targetFs      : Desired output sampling frequency (Hz)
%   targetSamples : Desired output length in samples
%   className     : Signal class string for class-aware cropping
%                   ("impulse" | "sinusoidal" | "white_noise" | "step" | "chirp")
%
%   Outputs
%   -------
%   x             : Processed audio column vector [targetSamples × 1]
%   info          : Struct with processing metadata fields (see below)
%
%   Processing steps
%   ----------------
%   1. Record original metadata
%   2. Convert stereo → mono
%   3. Remove NaN / Inf
%   4. Remove DC offset
%   5. Resample to targetFs
%   6. Peak amplitude normalization
%   7. Class-aware length standardization (crop / pad)
%   8. Final peak normalization (post-pad)
%
%   ============================================================

%% ------------------------------------------------------------
%  STEP 0: RECORD ORIGINAL METADATA
% ------------------------------------------------------------

info.originalFs       = originalFs;
info.processedFs      = targetFs;
info.originalSamples  = size(x, 1);
info.originalLength   = size(x, 1);
info.originalChannels = size(x, 2);
info.originalDuration = size(x, 1) / originalFs;
info.originalPeak     = max(abs(x(:)));
info.originalMean     = mean(x(:));

% Initialize flags
info.wasResampled = false;
info.wasPadded    = false;
info.wasCropped   = false;

%% ------------------------------------------------------------
%  STEP 1: CONVERT STEREO → MONO
%  Average channels. Safe for both mono and stereo input.
% ------------------------------------------------------------

if size(x, 2) > 1
    x = mean(x, 2);   % column vector
else
    x = x(:);         % ensure column
end

%% ------------------------------------------------------------
%  STEP 2: REMOVE NaN / Inf
%  Safety net for corrupted samples.
% ------------------------------------------------------------

x(~isfinite(x)) = 0;

%% ------------------------------------------------------------
%  STEP 3: REMOVE DC OFFSET
%  Prevents classifier from learning recording-pipeline artifact.
%  Applied BEFORE resampling so the resampler sees DC-free data.
% ------------------------------------------------------------

x = x - mean(x);

%% ------------------------------------------------------------
%  STEP 4: RESAMPLE TO TARGET_FS
%  Unifies spectral features across files with different fs.
%  Uses MATLAB built-in resample (polyphase anti-aliasing filter).
% ------------------------------------------------------------

if originalFs ~= targetFs
    x = resample(x, targetFs, originalFs);
    info.wasResampled = true;
end

%% ------------------------------------------------------------
%  STEP 5: AMPLITUDE NORMALIZATION (PRE-LENGTH)
%  Prevents the classifier from learning level differences
%  caused by the generation pipeline rather than signal class.
% ------------------------------------------------------------

peak = max(abs(x));
if peak > 0
    x = x / peak;
end

%% ------------------------------------------------------------
%  STEP 6: CLASS-AWARE LENGTH STANDARDIZATION
%
%  Goal: produce exactly targetSamples without destroying the
%  class-defining region of each signal type.
%
%  Strategy per class:
%
%  impulse     → event is a sharp spike; detect and center it
%  step        → transition is the key feature; locate and keep
%  chirp       → frequency sweep must survive; center crop
%  sinusoidal  → stationary; simple center crop / right-pad
%  white_noise → stationary; simple center crop / right-pad
%
% ------------------------------------------------------------

currentSamples = numel(x);

if currentSamples < targetSamples
    %% --------------------------------------------------------
    %  SHORTER THAN TARGET → ZERO PAD
    % --------------------------------------------------------
    x = zeroPadSignal(x, targetSamples, className);
    info.wasPadded = true;

elseif currentSamples > targetSamples
    %% --------------------------------------------------------
    %  LONGER THAN TARGET → INTELLIGENT CROP
    % --------------------------------------------------------
    x = cropSignal(x, targetSamples, className);
    info.wasCropped = true;
end
% If exact length, nothing to do.

%% ------------------------------------------------------------
%  STEP 7: FINAL NORMALIZATION
%  Re-normalize after padding (zeros can shift peak if original
%  signal was already at full scale and zero-padding introduced
%  no change, but defensive normalization costs nothing).
% ------------------------------------------------------------

peak = max(abs(x));
if peak > 0
    x = x / peak;
end

%% ------------------------------------------------------------
%  FINAL METADATA
% ------------------------------------------------------------

info.targetFs          = targetFs;
info.processedFs       = targetFs;
info.processedSamples  = numel(x);
info.processedLength   = numel(x);
info.processedDuration = numel(x) / targetFs;
info.processedPeak     = max(abs(x));
info.processedMean     = mean(x);

end   % preprocessAudioFile


%% ============================================================
%  LOCAL HELPER: zeroPadSignal
%  Pad a short signal to targetSamples with class awareness.
% =============================================================

function x = zeroPadSignal(x, targetSamples, className)
%ZEROPADSIGNAL  Pad signal with zeros to reach targetSamples.
%
%   Padding strategy:
%     impulse     → event near centre; pad symmetrically around it
%     step        → transition near centre; pad symmetrically
%     chirp       → sweep at start; left-align, right-pad
%     sinusoidal  → centre pad (symmetric)
%     white_noise → centre pad (symmetric)

    needed = targetSamples - numel(x);

    switch lower(char(className))

        case 'impulse'
            % Impulse: centre the spike in the output window.
            % Find the maximum-energy sample and centre it.
            [~, peakIdx] = max(abs(x));
            targetCenter = floor(targetSamples / 2);
            leftPad  = max(0, targetCenter - peakIdx + 1);
            rightPad = targetSamples - numel(x) - leftPad;
            if rightPad < 0
                % Spike is already right of centre; shift left pad
                leftPad  = 0;
                rightPad = targetSamples - numel(x);
            end

        case 'step'
            % Step: find the transition midpoint and centre it.
            % Transition = region of highest first-difference.
            dx = abs(diff(x));
            [~, transIdx] = max(dx);
            targetCenter = floor(targetSamples / 2);
            leftPad  = max(0, targetCenter - transIdx);
            rightPad = targetSamples - numel(x) - leftPad;
            if rightPad < 0
                leftPad  = 0;
                rightPad = targetSamples - numel(x);
            end

        case 'chirp'
            % Chirp: preserve the beginning of the sweep (left-align).
            leftPad  = 0;
            rightPad = needed;

        otherwise
            % Sinusoidal, white_noise: symmetric centre padding.
            leftPad  = floor(needed / 2);
            rightPad = needed - leftPad;

    end

    x = [zeros(leftPad, 1); x(:); zeros(rightPad, 1)];

    % Safety clamp to exact length (rounding edge case)
    x = x(1:targetSamples);

end   % zeroPadSignal


%% ============================================================
%  LOCAL HELPER: cropSignal
%  Crop a long signal to targetSamples with class awareness.
% =============================================================

function x = cropSignal(x, targetSamples, className)
%CROPSIGNAL  Crop signal to targetSamples with class awareness.
%
%   Cropping strategy:
%     impulse     → centre crop around the peak sample
%     step        → centre crop around the transition
%     chirp       → left-aligned (keep beginning of sweep)
%     sinusoidal  → centre crop (any window is representative)
%     white_noise → centre crop (stationary process)

    switch lower(char(className))

        case 'impulse'
            % Find the peak and extract a window centred on it.
            [~, peakIdx] = max(abs(x));
            halfWin  = floor(targetSamples / 2);
            startIdx = peakIdx - halfWin;
            startIdx = max(1, startIdx);
            endIdx   = startIdx + targetSamples - 1;
            if endIdx > numel(x)
                endIdx   = numel(x);
                startIdx = endIdx - targetSamples + 1;
            end

        case 'step'
            % Find the sharpest transition and extract centred window.
            dx = abs(diff(x));
            [~, transIdx] = max(dx);
            halfWin  = floor(targetSamples / 2);
            startIdx = transIdx - halfWin;
            startIdx = max(1, startIdx);
            endIdx   = startIdx + targetSamples - 1;
            if endIdx > numel(x)
                endIdx   = numel(x);
                startIdx = endIdx - targetSamples + 1;
            end

        case 'chirp'
            % Keep the beginning (left-align).
            startIdx = 1;
            endIdx   = targetSamples;

        otherwise
            % Centre crop for sinusoidal and white_noise.
            startIdx = floor((numel(x) - targetSamples) / 2) + 1;
            endIdx   = startIdx + targetSamples - 1;

    end

    x = x(startIdx:endIdx);

end   % cropSignal
