import argparse
import sys
import numpy as np
from audio.audio_classifier import AudioClassifier
from image.image_classifier import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description="SPANDHAN Signal Classification CLI")
    parser.add_argument("--type", choices=["audio", "image"], required=True, help="Signal modality")
    parser.add_argument("--input", required=True, help="Input file path")
    
    args = parser.parse_args()
    print(f"Executing SPANDHAN predictor for {args.type} input: {args.input}")

    if args.type == "audio":
        classifier = AudioClassifier()
        dummy_features = np.random.randn(1, 4)
        result = classifier.predict(dummy_features)
    else:
        classifier = ImageClassifier()
        dummy_features = np.random.randn(1, 4)
        result = classifier.predict(dummy_features)

    print(f"Predicted Signal Class: {result}")

if __name__ == "__main__":
    main()
