# Classia

![](https://img.shields.io/pypi/pyversions/name)

Classia is a hierarchical classifier for images or text.

```shell
pip install classia
```

The fastest way to use Classia is to call the CLI on a folder of images or text files.

```shell
train_data/
|- animals/
    |- mammals/
       |- marsupials/
          |- koala/
             |- image001.jpg 
             |- image002.jpg
             |- ...
          |- ...
    |- reptiles/
       |- ...
    |- ...
|- ...
```
## 1) Train
```shell
classia train --model animals --images=train_data/animals
```

or

```shell
classia train --model animals --docs=train_data/animals
```

Some key arguments:
```shell
--model (str, Required): The name of the model or task you want to name. (e.g., animal, wiki)
--model_dir (str, Optional, default=~/.cache/classia/models): The path of the directory to store named models
--docs (str): The directory of training documents, if training a text classifier
--images (str): The directory of training images, if training a images classifier
--lr (float, Optional, default=0.001): The learning rate to use during training
--batch_size (int, Optional, default=8): The batch size to use during training
--epochs (int, Optional, default=10): The number of epochs to train for
```

For more detailed infromation about arguments, refer to 
[cli.py](src/classia/cli.py)

And then use your model with the predict command.

```shell
classia predict --model animals new_data/image304.jpg
> birds/raptors/eagle
```

## Pre-trained Models


## 2) Download
You can download a pre-trained model using the download command.

```shell
classia download --model=dbpedia --download_dir train_data
classia download --dataset=inaturalist21-mini.zip --download_dir train_data
```
Some key arguments:
```shell
--model or --dataset: Either of them is required.
--dataset (str): name of the dataset (e.g., inat21-mini) or file name (inat21-mini.zip)
--download_dir (str, Optional): The path of the directory to download data. If not passed, it will be downloaded to ~/.cache/classia/models or ~/.cache/classia/datasets depending on the the argument (--model or --dataset) chosen above.
```
In above example, data will be saved in `train_data/dbpedia/best.pth` and `train_data/inaturalist21-mini.zip` respectively.

Datasets:
| Task | Name | Dataset | Labels | Notes |
|-------------|-------------|---------------|--------|-------|
| Image | inaturalist21-mini  | [iNaturalist 2021 (Mini)](https://github.com/visipedia/inat_comp/tree/master/2021) |        | Non-commercial research/educational use |
| Text | dbpedia | [DBPedia](https://www.kaggle.com/datasets/danofer/dbpedia-classes) |        | CC0: Public Domain |
| Text | amazon_product_reviews | [Amazon product reviews](https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification) |        | CC0: Public Domain |

Note: `Name` in above table is the name of `--dataset` (with or without extension `.zip`) you pass when  `classia --download`

## 3) Evaluation

```shell
classia test --model animals --images=test_data/animals
```

## 4) Test unseen data
```shell
classia predict --model=inat21-mini new_data/*.jpg
```


## Licensing

Classia is provided under AGPLv3, or via a [commercial license](https://shop.aiml.team/products/classia). 