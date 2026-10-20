"""
VGG16 Transfer Learning Architecture for Aircraft Defect Classification.
Categorizes structural aircraft damage into binary classes: 'Dent' vs 'Crack'.
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 512,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0001
) -> Model:
    """
    Constructs and compiles the transfer learning model with a VGG16 backbone.

    Args:
        input_shape: Image dimensions (height, width, channels).
        dense_units: Number of neurons in fully connected classification head.
        dropout_rate: Dropout regularization probability.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model ready for training.
    """
    # Load pre-trained VGG16 without the original ImageNet top classification layers
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base feature extraction layers
    for layer in base_model.layers:
        layer.trainable = False

    # Attach custom classification head
    x = Flatten(name='flatten_features')(base_model.output)
    x = Dense(dense_units, activation='relu', name='dense_feature_head')(x)
    x = Dropout(dropout_rate, name='dropout_regularization')(x)
    predictions = Dense(1, activation='sigmoid', name='binary_defect_output')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='VGG16_Aircraft_Classifier')

    # Compile model with binary cross-entropy and Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_training_curves(history_dict: dict, save_path: str = None):
    """
    Plots training and validation accuracy and loss curves.
    """
    epochs = range(1, len(history_dict['accuracy']) + 1)

    plt.figure(figsize=(10, 4))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history_dict['val_loss'], 'r--', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], 'r--', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def test_model_on_image(test_generator, model: Model, index_to_plot: int = 0):
    """
    Evaluates model prediction on a specific batch sample and visualizes the outcome.
    """
    labels = {0: 'crack', 1: 'dent'}
    images, true_labels = test_generator[0]

    img = images[index_to_plot]
    true_label = labels[int(true_labels[index_to_plot])]

    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    pred_label = labels[int(np.round(pred))]

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f'Prediction: {pred_label} (prob: {pred:.2f})\nActual: {true_label}', fontsize=10)
    plt.axis('off')
    plt.show()
