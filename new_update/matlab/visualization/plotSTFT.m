function plotSTFT(T, F, S)
    % PLOTSTFT Plots Spectrogram from STFT.
    figure;
    surf(T, F, 10*log10(abs(S)), 'EdgeColor', 'none');
    axis tight; view(0,90);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('STFT Spectrogram');
    colorbar;
end
