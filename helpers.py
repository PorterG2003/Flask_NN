# 07-18-2023 Porter Gardiner

import numpy as np

def scale_weights(weights, scaling_factor=1000, decimal_places=2):
    # Compute the mean and standard deviation for each layer
    mean_values = [np.mean(layer) for layer in weights]
    std_values = [np.std(layer) for layer in weights]

    # Apply z-score normalization to each layer of the weights array
    normalized_weights = [
        [
            [
                ((w - mean) / std) if std != 0 else 0
                for w in row
            ]
            for row in layer
        ]
        for layer, mean, std in zip(weights, mean_values, std_values)
    ]

    # Scale and round the normalized weights
    scaled_rounded_weights = [
        [
            [
                round(w * scaling_factor, decimal_places)
                for w in row
            ]
            for row in layer
        ]
        for layer in normalized_weights
    ]

    return scaled_rounded_weights