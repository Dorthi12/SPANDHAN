function parsed = parseMLResult(mlOutput)
    % PARSEMLRESULT Formats ML classification results for visualization/reporting.
    
    parsed.label = mlOutput;
    parsed.timestamp = datestr(now);
end
