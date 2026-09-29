function result = runConvolution(signal, kernel)
    % RUNCONVOLUTION Computes 1D linear convolution of signal and kernel.
    result = conv(signal, kernel, 'same');
end
