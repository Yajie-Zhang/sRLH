# sLRH
Sub-region localized hashing for fine-grained image retrieval

<details>
<summary>Dataset Preparation</summary>

```python
CUB-200-2011:
|--CUB_200_2011
  |--images
    |--001....
    |--002...
    ...
  |--classes.txt
  |--image_class_labels.txt
  |--image.txt
  |--train_test_split.txt


FGVC-Aircraft:
|--FGVC-aircraft
  |--data
	  |--images
		  |--...
		|--test.txt
		|--train.txt


Stanford Cars:
|--Stanford_Cars
  |--cars_test
    |--...
  |--cars_train
    |--...
  |--test.txt
  |--train.txt

Stanford Dogs:
|--dogs
  |--images
    |--Images
      |--file
      |--file
      ...
  |--lists
    |--file
    |--file
    ...
  |--test_data.mat
  |--train_data.mat
```
</details>


<details>
<summary>Finetune</summary>

```python
(1) Put the parameters of Resnet18 into the path .models/petrained. This parameters can be download at: .
(2) Finetune the network with the cross-entropy loss for classification. Such as: python finetune_cub.py. 
(3) Choose the network with minimized loss to initialize the sRLH models.
You can also use our pretrained models. The pretrained models can be download at: .
```
</details>

<details>
<summary>Train</summary>

```python
(1) Put the finetuned network into the path .checkpoint.
(2) Train the network. Such as: python cub_train.py	
```
</details>


<details>
<summary>Citation</summary>

```python

	
```
</details>
