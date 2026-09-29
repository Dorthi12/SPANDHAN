function filteredSignal = runIIR(signal, cutoff, fs)
    % RUNIIR Applies a Butterworth IIR Low-Pass filter to signal.
    n = 4;
    Wn = cutoff / (fs/2);
    [b, a] = butter(n, Wn, 'low');
    filteredSignal = filter(b, a, signal);
end
