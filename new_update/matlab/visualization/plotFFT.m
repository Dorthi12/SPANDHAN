function plotFFT(f, Y)
    % PLOTFFT Plots Single-Sided Amplitude Spectrum.
    figure;
    plot(f, Y, 'LineWidth', 1.5);
    xlabel('Frequency (Hz)');
    ylabel('|Y(f)|');
    title('Single-Sided Amplitude Spectrum');
    grid on;
end
