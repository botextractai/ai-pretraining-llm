# Pretraining Large Language Models (LLMs) using Hugging Face

Pretraining is the first step in creating a new or improved Large Language Model (LLM). It can be used to create a new model from scratch, or to improve an existing untuned (foundation) model with further training.

This example uses an existing model called `TinySolar-248m-4k` and trains this model with additional unstructured text data. This process is known as "continued pretraining".

However, the training process would be the same for training a new model. The difference is just that for a new model, the model weights would initially be randomly initialised to start with, while this example uses the already existing weights of the `TinySolar-248m-4k` model to improve (learn) by further adjusting the weights. Because the weights get updated during the training, the model learns to better predict the next output token for the answers from the examples in the training data.

Please note that this example just shows how that the pretraining process works. It contains by far not enough training data to be really useful and does by far not enough training steps to effectively improve the model, but a "real" pretraining process would use the same code basis, just with much more training data and more steps.

Pretraining a new model from scratch requires huge amounts of unstructured text data. Continued pretraining as in this example can be done with any amount of unstructured text data, although it usually still requires very large amounts of unstructured text data to be effective. Pretraining also usually requires a lot more data than fine-tuning, which, unlike pretraining, uses structured (question/answer pairs) data.

Pretraining is very expensive, particularly when creating a model from scratch! Even for very small models, the costs can be in the hundreds of thousands of US Dollars and take weeks or even months of compute time. You can get a rough estimate of the training job costs using [this calculator from Hugging Face](https://huggingface.co/training-cluster). For training on other cloud infrastructure like Amazon Web Services, Microsoft Azure, or Google Cloud, please consult those providers for up to date cost estimates.

In most cases, pretraining is more efficient than fine-tuning for a model to learn new knowledge. Fine-tuning might only work well in areas where the model already has good knowledge. Fine-tuning should ideally only be used for a model to learn answering questions in a specific way.

This example uses Weights & Biases (W&B) for monitoring the training process. W&B is a MLOps platform that can help developers monitor and document Machine Learning training workflows from end to end. W&B is used to get an idea of how well the training is working and if the model is improving over time. You need a W&B API key for this project. [Get your free Weights & Biases API key here](https://wandb.ai/authorize). Insert your W&B API key in the `main.py` script.

You can check your W&B projects at https://wandb.ai/YOUR_WANDB_USER_NAME/projects .

This example runs on all machines, as it only uses the Central Processing Unit (CPU). It therefore runs slowly. However, if you have a Graphics Processing Unit (GPU), then you should change the value of the `device_map` setting in the `main.py` script from the current value `cpu` to the new value `auto`. You can further increase the training speed by experimenting with the `dataloader_num_workers` setting in the `main.py` script, which is currently commented out, as it might crash on some systems.

## The results

You can see the results of your pretraining in the project "Logs" on Weights & Biases.

Ideally, you should see the loss number decreasing while the training progresses. However, as this example only does 30 steps, this might not always be immediately evident. Here are the results from an example training run:

```
{'loss': 4.1206, 'grad_norm': 60.5, 'learning_rate': 5e-06, 'epoch': 0.0}
{'loss': 3.3299, 'grad_norm': 42.25, 'learning_rate': 1e-05, 'epoch': 0.0}
{'loss': 4.1796, 'grad_norm': 48.75, 'learning_rate': 1.5e-05, 'epoch': 0.0}
{'loss': 3.446, 'grad_norm': 56.25, 'learning_rate': 2e-05, 'epoch': 0.0}
{'loss': 4.9945, 'grad_norm': 53.0, 'learning_rate': 2.5e-05, 'epoch': 0.0}
{'loss': 4.3966, 'grad_norm': 46.5, 'learning_rate': 3e-05, 'epoch': 0.0}
{'loss': 3.6688, 'grad_norm': 56.25, 'learning_rate': 3.5e-05, 'epoch': 0.0}
{'loss': 3.984, 'grad_norm': 49.75, 'learning_rate': 4e-05, 'epoch': 0.0}
{'loss': 3.6471, 'grad_norm': 56.25, 'learning_rate': 4.5e-05, 'epoch': 0.0}
{'loss': 4.1498, 'grad_norm': 44.0, 'learning_rate': 5e-05, 'epoch': 0.0}
{'loss': 3.8475, 'grad_norm': 45.5, 'learning_rate': 4.75e-05, 'epoch': 0.0}
{'loss': 4.5935, 'grad_norm': 56.5, 'learning_rate': 4.5e-05, 'epoch': 0.0}
{'loss': 3.8125, 'grad_norm': 48.75, 'learning_rate': 4.25e-05, 'epoch': 0.0}
{'loss': 3.8998, 'grad_norm': 50.25, 'learning_rate': 4e-05, 'epoch': 0.0}
{'loss': 4.2316, 'grad_norm': 48.25, 'learning_rate': 3.7500000000000003e-05, 'epoch': 0.0}
{'loss': 4.7784, 'grad_norm': 47.0, 'learning_rate': 3.5e-05, 'epoch': 0.0}
{'loss': 3.6636, 'grad_norm': 53.25, 'learning_rate': 3.2500000000000004e-05, 'epoch': 0.0}
{'loss': 4.5996, 'grad_norm': 99.0, 'learning_rate': 3e-05, 'epoch': 0.0}
{'loss': 2.9029, 'grad_norm': 46.5, 'learning_rate': 2.7500000000000004e-05, 'epoch': 0.0}
{'loss': 3.5262, 'grad_norm': 50.0, 'learning_rate': 2.5e-05, 'epoch': 0.0}
{'loss': 4.0175, 'grad_norm': 43.5, 'learning_rate': 2.25e-05, 'epoch': 0.0}
{'loss': 4.0017, 'grad_norm': 45.5, 'learning_rate': 2e-05, 'epoch': 0.0}
{'loss': 3.8715, 'grad_norm': 58.75, 'learning_rate': 1.75e-05, 'epoch': 0.0}
{'loss': 3.259, 'grad_norm': 50.75, 'learning_rate': 1.5e-05, 'epoch': 0.0}
{'loss': 3.989, 'grad_norm': 42.75, 'learning_rate': 1.25e-05, 'epoch': 0.0}
{'loss': 3.9207, 'grad_norm': 44.0, 'learning_rate': 1e-05, 'epoch': 0.0}
{'loss': 3.4957, 'grad_norm': 46.25, 'learning_rate': 7.5e-06, 'epoch': 0.0}
{'loss': 4.3065, 'grad_norm': 49.75, 'learning_rate': 5e-06, 'epoch': 0.0}
{'loss': 3.2209, 'grad_norm': 60.0, 'learning_rate': 2.5e-06, 'epoch': 0.0}
{'loss': 3.3239, 'grad_norm': 260.0, 'learning_rate': 0.0, 'epoch': 0.0}
{'train_runtime': 6088.7471, 'train_samples_per_second': 0.01, 'train_steps_per_second': 0.005, 'train_loss': 3.905960440635681, 'epoch': 0.0}
```
