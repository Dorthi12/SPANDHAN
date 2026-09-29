function plotFilterResponse(b, a, fs)
    % PLOTFILTERRESPONSE Plots frequency response of digital filter.
    figure;
    freqz(b, a, 512, fs);
    title('Filter Frequency Response');
end
