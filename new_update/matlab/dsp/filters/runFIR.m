function filteredSignal = runFIR(signal, cutoff, fs)
    % RUNFIR Applies a Low-Pass FIR filter to signal.
    n = 50;
    Wn = cutoff / (fs/2);
    b = fir1(n, Wn, 'low');
    filteredSignal = filter(b, 1, signal);
end
