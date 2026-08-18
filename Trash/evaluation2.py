import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from models.model2 import CIFAR10Classifier
from data.vision_dataset import get_cifar10_loaders


def evaluate_vision_model():
    print("Booting Vision Evaluator...")

    # 1. Load ONLY the test data (We use '_' to ignore the train_loader)
    _, test_loader = get_cifar10_loaders(batch_size=32)

    # 2. Load your trained model blueprint and weights
    model = CIFAR10Classifier()
    # Make sure this matches the exact name you used in your trainer.py!
    model.load_state_dict(torch.load("../models/checkpoints/model2_cifar10.pth", weights_only=True))

    # CRITICAL: This locks the model's weights and turns off training behaviors!
    model.eval()

    # 3. Setup our scorekeepers
    correct_guesses = 0
    total_images = 0

    print("Grading 10,000 unseen images...")

    # 4. The Evaluation Loop
    # torch.no_grad() disables all the heavy calculus memory tracking.
    # It makes evaluation run about 3x faster since we aren't learning!
    with torch.no_grad():
        for images, labels in test_loader:
            # A. The model outputs its 10 confidence scores
            outputs = model(images)

            # B. Find the class with the highest score (The "Guess")
            _, predicted_classes = torch.max(outputs, dim=1)

            # C. Tally up the score
            total_images += labels.size(0)
            correct_guesses += (predicted_classes == labels).sum().item()

    # 5. Calculate Final Percentage
    accuracy = 100 * correct_guesses / total_images
    print(f"\n[ RESULTS ]")
    print(f"Total Correct: {correct_guesses} / {total_images}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_vision_model()