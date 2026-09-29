function plotWavelet(c, l)
    % PLOTWAVELET Plots Wavelet Coefficients.
    figure;
    plot(c, 'LineWidth', 1.2);
    title('Wavelet Decomposition Coefficients');
    xlabel('Coefficient Index'); ylabel('Value');
    grid on;
end
