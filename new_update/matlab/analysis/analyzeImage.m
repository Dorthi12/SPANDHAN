function res = analyzeImage(imagePath)
    % ANALYZEIMAGE Full analysis pipeline for an image file.
    img = loadImage(imagePath);
    processed = preprocessImage(img);
    res.processedImage = processed;
    res.meanIntensity = mean(processed(:));
end
