function figures = plotDSPResults(result, modality, varargin)
% PLOTDSPRESULTS
% Visualizes results produced by analyzeAudio() or analyzeImage().
%
% INPUTS:
%   result   - Output structure from analyzeAudio() or analyzeImage()
%   modality - "audio" or "image"
%
% OPTIONAL NAME-VALUE:
%
%   "ShowInput"
%       true/false, default true
%
%   "ShowFFT"
%       true/false, default true
%
%   "ShowSTFT"
%       true/false, default true
%
%   "ShowWavelet"
%       true/false, default true
%
%   "ShowFilter"
%       true/false, default true
%
%   "ShowSystem"
%       true/false, default true
%
%   "Visible"
%       "on" or "off", default "on"
%
% OUTPUT:
%   figures - Structure containing figure handles.
%
% AUDIO FIGURES:
%   .signal
%   .fft
%   .stft
%   .wavelet
%   .fir
%   .iir
%   .convolution
%   .deconvolution
%
% IMAGE FIGURES:
%   .input
%   .fft2
%   .wavelet2D
%   .filter
%   .convolution
%   .deconvolution
%
% IMPORTANT:
%   This function only visualizes existing analysis results.
%   It does not perform DSP calculations.

    %% ===============================================================
    % 1. INPUT VALIDATION
    % ===============================================================

    if nargin < 2
        error("plotDSPResults requires result and modality.");
    end

    if ~isstruct(result)
        error("result must be a structure.");
    end

    modality = lower(string(modality));

    if ~any(modality == ["audio", "image"])
        error("Modality must be audio or image.");
    end

    %% ===============================================================
    % 2. DEFAULT OPTIONS
    % ===============================================================

    showInput   = true;
    showFFT     = true;
    showSTFT    = true;
    showWavelet = true;
    showFilter  = true;
    showSystem  = true;
    visible     = "on";

    %% ===============================================================
    % 3. PARSE OPTIONS
    % ===============================================================

    if mod(length(varargin), 2) ~= 0
        error("Optional arguments must be supplied as name-value pairs.");
    end

    for k = 1:2:length(varargin)

        parameter = lower(string(varargin{k}));
        value     = varargin{k+1};

        switch parameter

            case "showinput"
                showInput = logical(value);

            case "showfft"
                showFFT = logical(value);

            case "showstft"
                showSTFT = logical(value);

            case "showwavelet"
                showWavelet = logical(value);

            case "showfilter"
                showFilter = logical(value);

            case "showsystem"
                showSystem = logical(value);

            case "visible"
                visible = string(value);

            otherwise
                error("Unknown plotting option: %s", parameter);

        end
    end

    %% ===============================================================
    % 4. INITIALIZE FIGURE STRUCTURE
    % ===============================================================

    figures          = struct();
    figures.modality = modality;

    %% ===============================================================
    % 5. AUDIO VISUALIZATION
    % ===============================================================

    if modality == "audio"

        %% -----------------------------------------------------------
        % SIGNAL WAVEFORM
        % -----------------------------------------------------------

        if showInput && isfield(result, "analysisSignal")

            x = result.analysisSignal;

            if isfield(result, "samplingFrequency")
                Fs = result.samplingFrequency;
            else
                Fs = 1;
            end

            t = (0:length(x)-1)' / Fs;

            fig = figure( ...
                "Name", "SPANDHAN - Audio Signal", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            plot(t, x, "LineWidth", 1);

            grid on;
            xlabel("Time (s)");
            ylabel("Amplitude");

            title("Audio Signal - " + string(result.signalClass));

            figures.signal = fig;

        end

        %% -----------------------------------------------------------
        % FFT
        % -----------------------------------------------------------

        if showFFT && ...
                isfield(result, "fft") && ~isempty(result.fft)

            fftResult = result.fft;

            fig = figure( ...
                "Name", "SPANDHAN - FFT", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(2, 1, "TileSpacing", "compact");

            nexttile;
            plot(fftResult.frequency, fftResult.magnitude, "LineWidth", 1);
            grid on;
            xlabel("Frequency (Hz)");
            ylabel("Magnitude");
            title("Magnitude Spectrum");
            xlim([0 max(fftResult.frequency)]);

            nexttile;
            plot(fftResult.frequency, fftResult.magnitudeDB, "LineWidth", 1);
            grid on;
            xlabel("Frequency (Hz)");
            ylabel("Magnitude (dB)");
            title("Magnitude Spectrum (dB)");
            xlim([0 max(fftResult.frequency)]);

            sgtitle("FFT Analysis - " + string(result.signalClass));

            figures.fft = fig;

        end

        %% -----------------------------------------------------------
        % STFT
        % -----------------------------------------------------------

        if showSTFT && ...
                isfield(result, "stft") && ~isempty(result.stft)

            stftResult = result.stft;

            fig = figure( ...
                "Name", "SPANDHAN - STFT", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            imagesc( ...
                stftResult.time, ...
                stftResult.frequency, ...
                stftResult.magnitudeDB);

            axis xy;
            xlabel("Time (s)");
            ylabel("Frequency (Hz)");
            title("Short-Time Fourier Transform (dB)");
            colorbar;
            colormap("jet");

            figures.stft = fig;

        end

        %% -----------------------------------------------------------
        % WAVELET (1-D)
        % -----------------------------------------------------------

        if showWavelet && ...
                isfield(result, "wavelet") && ~isempty(result.wavelet)

            waveletResult = result.wavelet;
            level         = waveletResult.decompositionLevel;

            fig = figure( ...
                "Name", "SPANDHAN - Wavelet", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            numberOfPlots = level + 1;

            tiledlayout(numberOfPlots, 1, "TileSpacing", "compact");

            %% Approximation
            nexttile;

            if isfield(waveletResult, "reconstructedApproximation")
                approxSignal = waveletResult.reconstructedApproximation;
            else
                approxSignal = waveletResult.approximation;
            end

            plot(approxSignal, "LineWidth", 1);
            grid on;
            title("Approximation A" + string(level));
            xlabel("Sample");
            ylabel("Amplitude");

            %% Detail levels
            for k = 1:level

                nexttile;

                if isfield(waveletResult, "reconstructedDetails") && ...
                        ~isempty(waveletResult.reconstructedDetails)

                    detailSignal = waveletResult.reconstructedDetails{k};

                elseif isfield(waveletResult, "details") && ...
                        ~isempty(waveletResult.details)

                    detailSignal = waveletResult.details{k};

                else
                    % Fall back: extract from coefficient vector.
                    detailSignal = detcoef( ...
                        waveletResult.coefficients, ...
                        waveletResult.bookkeeping, ...
                        k);
                end

                plot(detailSignal, "LineWidth", 1);
                grid on;
                title("Detail D" + string(k));
                xlabel("Sample");
                ylabel("Amplitude");

            end

            sgtitle( ...
                "Wavelet Decomposition - " + ...
                string(waveletResult.waveletName));

            figures.wavelet = fig;

        end

        %% -----------------------------------------------------------
        % FIR FILTER RESPONSE
        % -----------------------------------------------------------

        if showFilter && ...
                isfield(result, "fir") && ~isempty(result.fir)

            firResult = result.fir;

            fig = figure( ...
                "Name", "SPANDHAN - FIR Response", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(2, 1, "TileSpacing", "compact");

            nexttile;
            plot(firResult.frequency, firResult.magnitudeDB, "LineWidth", 1);
            grid on;
            xlabel("Frequency (Hz)");
            ylabel("Magnitude (dB)");
            title("FIR Frequency Response");

            nexttile;
            plot(firResult.impulseResponse, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("FIR Impulse Response");

            sgtitle("FIR Filter");

            figures.fir = fig;

        end

        %% -----------------------------------------------------------
        % IIR FILTER RESPONSE
        % -----------------------------------------------------------

        if showFilter && ...
                isfield(result, "iir") && ~isempty(result.iir)

            iirResult = result.iir;

            fig = figure( ...
                "Name", "SPANDHAN - IIR Response", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(2, 1, "TileSpacing", "compact");

            nexttile;
            plot(iirResult.frequency, iirResult.magnitudeDB, "LineWidth", 1);
            grid on;
            xlabel("Frequency (Hz)");
            ylabel("Magnitude (dB)");
            title("IIR Frequency Response");

            nexttile;
            plot(iirResult.impulseResponse, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("IIR Impulse Response");

            sgtitle("IIR Filter (" + string(iirResult.filterFamily) + ")");

            figures.iir = fig;

        end

        %% -----------------------------------------------------------
        % CONVOLUTION
        % -----------------------------------------------------------

        if showSystem && ...
                isfield(result, "convolution") && ~isempty(result.convolution)

            convResult = result.convolution;

            fig = figure( ...
                "Name", "SPANDHAN - Convolution", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(3, 1, "TileSpacing", "compact");

            nexttile;
            plot(convResult.input, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("Input x[n]");

            nexttile;
            plot(convResult.impulseResponse, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("System Response h[n]");

            nexttile;
            plot(convResult.output, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("Convolution Output y[n] = x[n] * h[n]");

            sgtitle("Linear Convolution");

            figures.convolution = fig;

        end

        %% -----------------------------------------------------------
        % DECONVOLUTION
        % -----------------------------------------------------------

        if showSystem && ...
                isfield(result, "deconvolution") && ~isempty(result.deconvolution)

            deconvResult = result.deconvolution;

            fig = figure( ...
                "Name", "SPANDHAN - Deconvolution", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(2, 1, "TileSpacing", "compact");

            nexttile;
            plot(deconvResult.input, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("Observed y[n]");

            nexttile;
            plot(deconvResult.reconstructed, "LineWidth", 1);
            grid on;
            xlabel("Sample");
            ylabel("Amplitude");
            title("Recovered x̂[n] (Method: " + string(deconvResult.method) + ")");

            sgtitle("Signal Deconvolution");

            figures.deconvolution = fig;

        end

    %% ===============================================================
    % 6. IMAGE VISUALIZATION
    % ===============================================================

    else

        %% -----------------------------------------------------------
        % ORIGINAL IMAGE
        % -----------------------------------------------------------

        if showInput && isfield(result, "input")

            fig = figure( ...
                "Name", "SPANDHAN - Input Image", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            imshow(result.input, []);

            title("Input Image - " + string(result.signalClass));

            figures.input = fig;

        end

        %% -----------------------------------------------------------
        % 2-D FFT
        % -----------------------------------------------------------

        if showFFT && ...
                isfield(result, "fft2") && ~isempty(result.fft2)

            fftResult = result.fft2;

            fig = figure( ...
                "Name", "SPANDHAN - 2-D FFT", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(1, 2, "TileSpacing", "compact");

            nexttile;
            imshow(fftResult.grayImage, []);
            title("Grayscale Input");

            nexttile;
            imagesc(fftResult.magnitudeDB);
            axis image;
            axis xy;
            colorbar;
            colormap("jet");
            xlabel("Spatial Frequency X");
            ylabel("Spatial Frequency Y");
            title("Log Magnitude Spectrum (dB)");

            sgtitle("2-D Fourier Transform");

            figures.fft2 = fig;

        end

        %% -----------------------------------------------------------
        % 2-D WAVELET
        % -----------------------------------------------------------

        if showWavelet && ...
                isfield(result, "wavelet2D") && ~isempty(result.wavelet2D)

            waveletResult = result.wavelet2D;
            level         = waveletResult.decompositionLevel;

            fig = figure( ...
                "Name", "SPANDHAN - 2-D Wavelet", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(2, 2, "TileSpacing", "compact");

            nexttile;
            imshow(waveletResult.approximation, []);
            title("Approximation A" + string(level));

            nexttile;
            imshow(waveletResult.horizontalDetails{level}, []);
            title("Horizontal H" + string(level));

            nexttile;
            imshow(waveletResult.verticalDetails{level}, []);
            title("Vertical V" + string(level));

            nexttile;
            imshow(waveletResult.diagonalDetails{level}, []);
            title("Diagonal D" + string(level));

            sgtitle( ...
                "2-D Wavelet Decomposition - " + ...
                string(waveletResult.waveletName));

            figures.wavelet2D = fig;

        end

        %% -----------------------------------------------------------
        % IMAGE FILTER
        % -----------------------------------------------------------

        if showFilter && ...
                isfield(result, "filter") && ~isempty(result.filter)

            filterResult = result.filter;

            fig = figure( ...
                "Name", "SPANDHAN - Image Filter", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(1, 3, "TileSpacing", "compact");

            nexttile;
            imshow(filterResult.inputDouble, []);
            title("Original");

            nexttile;
            imshow(filterResult.filteredImage, []);
            title("Filtered (" + string(filterResult.filterType) + ")");

            nexttile;
            imshow(abs(filterResult.differenceImage), []);
            title("Difference");

            sgtitle("Image Filtering");

            figures.filter = fig;

        end

        %% -----------------------------------------------------------
        % IMAGE CONVOLUTION
        % -----------------------------------------------------------

        if showSystem && ...
                isfield(result, "convolution") && ~isempty(result.convolution)

            convResult = result.convolution;

            fig = figure( ...
                "Name", "SPANDHAN - Image Convolution", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(1, 2, "TileSpacing", "compact");

            nexttile;
            imshow(convResult.inputDouble, []);
            title("Input Image");

            nexttile;
            imshow(convResult.output, []);
            title("Convolution Output (" + string(convResult.outputMode) + ")");

            sgtitle("2-D Image Convolution");

            figures.convolution = fig;

        end

        %% -----------------------------------------------------------
        % IMAGE DECONVOLUTION
        % -----------------------------------------------------------

        if showSystem && ...
                isfield(result, "deconvolution") && ~isempty(result.deconvolution)

            deconvResult = result.deconvolution;

            fig = figure( ...
                "Name", "SPANDHAN - Image Deconvolution", ...
                "NumberTitle", "off", ...
                "Visible", visible);

            tiledlayout(1, 2, "TileSpacing", "compact");

            nexttile;
            imshow(deconvResult.inputDouble, []);
            title("Blurred / Observed");

            nexttile;
            imshow(deconvResult.restoredImage, []);
            title("Restored (" + string(deconvResult.method) + ")");

            sgtitle("Image Deconvolution");

            figures.deconvolution = fig;

        end

    end

end
