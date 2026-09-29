function features = prepareMLInput(dspOutput)
    % PREPAREMLINPUT Converts DSP analysis output into machine learning feature vector.
    
    if isstruct(dspOutput) && isfield(dspOutput, 'magnitude')
        features = mean(dspOutput.magnitude, 2);
    else
        features = dspOutput(:);
    end
end
