function result = runDeconvolution(signal, kernel)
    % RUNDECONVOLUTION Computes deconvolution (deconv) of signal and kernel.
    [result, ~] = deconv(signal, kernel);
end
