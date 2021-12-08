# Sub-region localized hashing for fine-grained image retrieval (sRLH)

Dataset Preparation
---
CUB-200-2011: link：https://pan.baidu.com/s/1cWVu7JHSQV9Pvw-dLlkDQw, extraction code：zzlz <br>
<details>
<summary>Details</summary>

```python
|--CUB_200_2011 
  |--images 
       |--001...
       |--002... 
       ... <br>
  |--classes.txt 
  |--image_class_labels.txt 
  |--image.txt 
  |--train_test_split.txt
```
</details>

FGVC-Aircraft: link：https://pan.baidu.com/s/1MEiwAJbBGmsCbpZ5u19x8Q, extraction code：91su <br>
<details>
<summary>Details</summary>

```python
|--FGVC-aircraft
  |--data
    |--images
       |--...
    |--test.txt
    |--train.txt
```
</details>

Stanford Cars: link：https://pan.baidu.com/s/1c6mivvIXXEjERP2ilDtHNg, extraction code：o96t <br>
<details>
<summary>Details</summary>

```python
|--Stanford_Cars
  |--cars_test
    |--...
  |--cars_train
    |--...
  |--test.txt
  |--train.txt
```
</details>

Stanford Dogs: link：https://pan.baidu.com/s/1mBDOOVwgT0RAzjIITlwbgg, extraction code：ivsu <br>
<details>
<summary>Details</summary>

```python
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

Finetune
---
(1) Put the parameters of Resnet18 into the path .models/petrained. This parameters can be download at link：https://pan.baidu.com/s/1uGfo2JCiX4GmqkGE2waG7A, Extraction code：7bu5. <br>
(2) Finetune the network with the cross-entropy loss for classification. Such as: python finetune_cub.py.  <br>
(3) Choose the network with minimum loss as the finetuned network. <br>
You can also use our pretrained models. The pretrained models can be download at link：https://pan.baidu.com/s/15FlAAZD9NZtW9MVKdwy7RA, Extraction code：fith. <br>

Train
---
(1) Put the finetuned network into the path .checkpoint. <br>
(2) Train the network. Such as: python cub_train.py	 <br>

Citation
---
    @ARTICLE{9638382,
    author={Xiang, Xinguang and Zhang, Yajie and Jin, Lu and Li, Zechao and Tang, Jinhui},
    journal={IEEE Transactions on Image Processing},
    title={Sub-Region Localized Hashing for Fine-Grained Image Retrieval}, 
    year={2021},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TIP.2021.3131042}}
