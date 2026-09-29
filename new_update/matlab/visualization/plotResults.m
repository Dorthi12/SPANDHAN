function plotResults(results)
    % PLOTRESULTS Displays comparative summary plots of analysis output.
    figure;
    if isfield(results, 'Y')
        plot(results.f, results.Y);
        title('DSP Results Overview');
    end
end
