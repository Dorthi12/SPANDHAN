function normImg = normalizeImage(img)
    % NORMALIZEIMAGE Normalizes image pixel values to range [0, 1].
    minVal = min(img(:));
    maxVal = max(img(:));
    if maxVal == minVal
        normImg = zeros(size(img));
    else
        normImg = (img - minVal) / (maxVal - minVal);
    end
end
