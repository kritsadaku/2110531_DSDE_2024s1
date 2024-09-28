# 1_Huggingface_image_classification

## Fine-Tuning Vision Transformers for Image Classification

## Get start 
```bash
pip install datasets transformers[torch]
```

## Import `bean` dataset
```python
from datasets import load_dataset
ds = load_dataset('beans',)
ds
```

## Profile the data 
check the 400th example from the 'train' split from the beans dataset.
```python
image = ds['train'][400]['image']
image
```
check the features of the train' dataset.
```python
labels = ds['train'].features['labels']
labels
```
output
```bash
ClassLabel(names=['angular_leaf_spot', 'bean_rust', 'healthy'], id=None)
```
## Show the example 

```python
from transformers.utils.dummy_vision_objects import ImageGPTFeatureExtractor
import random
from PIL import ImageDraw, ImageFont, Image

"""
Displays a grid of example images from the given dataset, with each row representing a different class label.

Args:
    ds (Dataset): The dataset to display examples from.
    seed (int, optional): The random seed to use for shuffling the examples. Defaults to 1234.
    examples_per_class (int, optional): The number of examples to display per class. Defaults to 3.
    size (tuple, optional): The size of each example image in the grid. Defaults to (350, 350).

Returns:
    PIL.Image: A grid of example images from the dataset.
"""
def show_examples(ds, seed: int = 1234, examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds['train'].features['labels'].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 24)

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it, and grab a few samples
        ds_slice = ds['train'].filter(lambda ex: ex['labels'] == label_id).shuffle(seed).select(range(examples_per_class))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            image = example['image']
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            draw.text(box, label, (255, 255, 255), font=font)

    return grid

show_examples(ds, seed=random.randint(0, 1337), examples_per_class=3)
```
## Import Vit Feature Extractor

```python
'''
Loads a pre-trained ViT (Vision Transformer) image processor from the specified model name or path. The ViTImageProcessor is responsible for preprocessing images for input to the ViT model, including resizing, normalization, and converting the image to the expected tensor format.

The `model_name_or_path` parameter specifies the name or path of the pre-trained ViT model to use. In this case, the 'google/vit-base-patch16-224' model is being used.
'''
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224'
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)
```

## Processing the data

```python
"""
Preprocesses an example from the dataset by applying the ViT image processor.

Args:
    example (dict): A dictionary containing the 'image' and 'labels' keys.

Returns:
    dict: A dictionary containing the preprocessed image tensor and the labels.
"""
def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs
```
apply process_example function to be the transform function.

```python
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['labels']
    return inputs

prepared_ds = ds.with_transform(transform)
```

## Training and Evaluation 
### Define data collate function
```python
import torch
"""
Collates a batch of examples into a dictionary of tensors.

Args:
    batch (list): A list of example dictionaries, where each dictionary contains 'pixel_values' and 'labels' keys.

Returns:
    dict: A dictionary containing the collated 'pixel_values' and 'labels' tensors.
"""
def collate_fn(batch):
    return { 'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
             'labels': torch.tensor([x['labels'] for x in batch])}
```
### Define an evaluation metric
fix  `from datasets import load_metric` to 
`from evaluate import load `
```python 
#!pip install evaluate --upgrade

import numpy as np
# from datasets import load_metric
from evaluate import load


metric = load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
```

 initializes a Vision Transformer model tailored for the number of classes present in your dataset, allowing you to fine-tune it for a specific image classification task. The ignore_mismatched_sizes=True option helps handle any mismatches that may arise due to architectural adjustments, such as changing the number of output classes.
```python
from transformers import ViTForImageClassification

labels = ds['train'].features['labels'].names
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(labels),
    ignore_mismatched_sizes=True
    )
```

set up the training arguments which are arguments configure the training process, including how the model is trained, evaluated, saved, and logged. They help to manage the training flow, ensure the best model is saved, and provide detailed logs that can be tracked in TensorBoard.
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="vit-base-beans-demo-v5",
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy='epoch',
  logging_steps=1,
  num_train_epochs=10,
  optim='adamw_torch',
  learning_rate=1e-5,
  remove_unused_columns=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
  metric_for_best_model="loss",
  seed=0
)
```
check a device for training
```python
training_args.device
```

## start training
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=feature_extractor,
)
```

## Before fine tuning

test the pre-fine ture model with test data set and print out the model's classification report
```python
from sklearn.metrics import  classification_report
'''
This code retrieves the model's predictions on the test dataset. It first calls the `predict()` method of the `Trainer` object, passing in the `prepared_ds['test']` dataset, to obtain the test predictions. It then uses the `np.argmax()` function to extract the label with the highest predicted probability for each prediction.
'''
test_predictions = trainer.predict(prepared_ds['test'])
# For each prediction, create the label with argmax
predictions = np.argmax(test_predictions[0], axis=1)
# Retrieve reference labels from test set
test_labels = np.array(prepared_ds["test"][:]["labels"])
print(classification_report(test_labels, predictions, target_names=['angular_leaf_spot', 'bean_rust', 'healthy'], digits=4))
```
output
```bash
                   precision    recall  f1-score   support

angular_leaf_spot     0.3231    0.4884    0.3889        43
        bean_rust     0.4746    0.6512    0.5490        43
          healthy     0.2500    0.0238    0.0435        42

         accuracy                         0.3906       128
        macro avg     0.3492    0.3878    0.3271       128
     weighted avg     0.3500    0.3906    0.3293       128
```

```python
'''
Trains the model using the provided training arguments, trainer, and dataset. Saves the trained model, logs the training metrics, and saves the training state.

This code is responsible for the actual training of the model. It uses the Trainer class from the transformers library to handle the training process, including data loading, model forward passes, and metric computation. After the training is complete, it saves the trained model, logs the training metrics, and saves the training state for potential future use or resumption of training.
'''

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```

```python
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
```

## Prediction 
test the fine ture model with test data set and print out the model's classification report
```python
from sklearn.metrics import  classification_report

test_predictions = trainer.predict(prepared_ds['test'])
# For each prediction, create the label with argmax
predictions = np.argmax(test_predictions[0], axis=1)
# Retrieve reference labels from test set
test_labels = np.array(prepared_ds["test"][:]["labels"])
print(classification_report(test_labels, predictions, target_names=['angular_leaf_spot', 'bean_rust', 'healthy'], digits=4))
```

```bash
'''
This code loads the TensorBoard extension in the Jupyter notebook and starts the TensorBoard visualization tool, pointing it to the log directory for the "vit-base-beans-demo-v5" experiment.

TensorBoard is a powerful tool for visualizing and analyzing the training process of machine learning models. By running this code, the user can view various metrics and visualizations related to the training of the "vit-base-beans-demo-v5" model, such as loss, accuracy, and more.
'''
%load_ext tensorboard
%tensorboard --logdir '{"vit-base-beans-demo-v5"}'/runs
```