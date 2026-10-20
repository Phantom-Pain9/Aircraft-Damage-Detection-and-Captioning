"""
Custom Keras Layer and Inference Pipeline utilizing Salesforce BLIP (Bootstrapping Language-Image Pre-training).
Integrates Multimodal Vision-Language capabilities into a TensorFlow/Keras environment for automated reporting.
"""

from typing import Union
from PIL import Image
import tensorflow as tf
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow/Keras Layer wrapping pretrained Salesforce BLIP model.
    Enables direct text generation (captioning or detailed summarization) inside a Keras pipeline.
    """

    def __init__(self, processor: BlipProcessor, model: BlipForConditionalGeneration, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path: tf.Tensor, task: tf.Tensor) -> tf.Tensor:
        """
        Executes text generation via tf.py_function wrapper.
        """
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path: tf.Tensor, task: tf.Tensor) -> str:
        """
        Loads the image, formats task-specific prompt, generates token sequence, and decodes.

        Args:
            image_path: Tensor string containing the filesystem path to the image.
            task: Tensor string ('caption' or 'summary').

        Returns:
            Decoded natural language description string.
        """
        try:
            image_path_str = image_path.numpy().decode("utf-8")
            image = Image.open(image_path_str).convert("RGB")

            task_str = task.numpy().decode("utf-8").lower()
            if task_str == "caption":
                prompt = "This is a picture of"
            else:
                prompt = "This is a detailed photo showing"

            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            output = self.model.generate(**inputs, max_new_tokens=50)
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            return f"Error during generation: {str(e)}"


def load_blip_components(model_name: str = "Salesforce/blip-image-captioning-base"):
    """
    Loads pretrained Hugging Face processor and model.
    """
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def generate_text(image_path: tf.Tensor, task: tf.Tensor, processor=None, model=None) -> tf.Tensor:
    """
    Convenience helper function to execute the custom layer on an image tensor.
    """
    if processor is None or model is None:
        processor, model = load_blip_components()

    blip_layer = BlipCaptionSummaryLayer(processor, model)
    return blip_layer(image_path, task)
