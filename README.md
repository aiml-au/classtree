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

```shell
classia train --model animals --images=train_data/animals
```

or

```shell
classia train --model animals --docs=train_data/animals
```

And then use your model with the predict command.

```shell
classia predict --model animals new_data/image304.jpg
> birds/raptors/eagle
```

## Pre-trained Models

You can download a pre-trained model using the download command.

```shell
classia download --model=inat21-mini
classia predict --model=inat21-mini new_data/*.jpg
```

| Name        | Dataset                                                                            | Labels | Notes |
|-------------|------------------------------------------------------------------------------------|--------|-------|
| inat21-mini | [iNaturalist 2021 (Mini)](https://github.com/visipedia/inat_comp/tree/master/2021) |        | Non-commercial research/educational use |

## Evaluation

```shell
classia test --model animals --images=test_data/animals
```

## Licensing

Classia is provided under AGPLv3, or via a [commercial license](https://shop.aiml.team/products/classia). 