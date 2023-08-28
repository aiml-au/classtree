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
classia train images --model animals --dir train_data/animals
```

or

```shell
classia train text --model animals --dir train_data/animals
```

And then use your model with the predict command.

```shell
classia predict --model animals new_data/image304.jpg
> birds/raptors/eagle
```

## Pre-trained Models

You can download a pre-trained model using the download command.

```shell
classia download model dbpedia
```

Or download a pre-prepared dataset.

```shell
classia download images inaturalist21-mini
classia download text dbpedia
```

If you want to fine-tune an existing model, you can use the `--from` flag during training with any downloaded model.

```shell
classia train text --model animals --from dbpedia --dir train_data/animals
```


### Available Models

| Task                 | Name               | Size | Dataset                | Notes                                        |
|----------------------|--------------------|------|------------------------|----------------------------------------------|
| Image Classification | inaturalist21-mini | M    | inaturalist21-mini     | Non-commercial research/educational use only |
| Text Classification  | dbpedia            | M    | dbpedia                |                                              |

### Available Datasets

| Type  | Name               | Dataset                                                                            | Notes                                        |
|-------|--------------------|------------------------------------------------------------------------------------|----------------------------------------------|
| Image | inaturalist21-mini | [iNaturalist 2021 (Mini)](https://github.com/visipedia/inat_comp/tree/master/2021) | Non-commercial research/educational use only |
| Text  | dbpedia            | [DBPedia](https://www.kaggle.com/datasets/danofer/dbpedia-classes)                 | CC0: Public Domain                           |

## Evaluation

You can test your model on a hold-out dataset using the `test` command.

```shell
classia test --model animals --dir=test_data/animals
```

## Licensing

Classia is available for non-commercial internal research use by academic institutions or not-for-profit organisations only, free of charge. Please, see the [license](./LICENSE.txt) for further details. To the extent permitted by applicable law, your use is at your own risk and our liability is limited. Interested in a commercial license? For commercial queries, please email <aimlshop@adelaide.edu.au> with subject line “Classia Commercial License”. 

This is an [AIML Shop](https://shop.aiml.team) project.
