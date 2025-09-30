# PyTorch MNIST Handwritten Digit Recognition Project

This is a classic MNIST handwritten digit recognition project implemented using PyTorch. The code includes the full process of data loading, model building, training, and evaluation.

## Features
- Uses a simple fully connected neural network for classification.
- Automatically detects and uses an available GPU (CUDA) for acceleration.
- Saves the trained model to the `checkpoints` directory after training.
- If a pre-trained model is found, it will load it and perform evaluation directly.
- Includes random seed fixation to ensure reproducible results.

## Requirements
This project depends on the following Python libraries. You can quickly install them using `pip`:

```bash
pip install -r requirements.txt
```

## How to Run

Simply run the Python script from your terminal:

```bash
python mnist_train.py
```

- **First Run**: The script will automatically download the MNIST dataset into the `data/` directory (which is ignored by `.gitignore`), then start training the model, and finally evaluate it. The trained model weights will be saved in the `checkpoints/` directory.
- **Subsequent Runs**: If the script detects an existing model file in `checkpoints/`, it will load the model directly and run the evaluation, skipping the training process.

## Results
After 5 epochs of training, the model can achieve an accuracy of over 97% on the test set.

**Example Output:**
```
Test results: 
 Average loss: 0.0745, Accuracy: 9768/10000 (97.68%)
```
