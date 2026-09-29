function plotSignal(t, signal, titleStr)
    % PLOTSIGNAL Plots a time-domain signal.
    figure;
    plot(t, signal, 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title(titleStr);
    grid on;
end
