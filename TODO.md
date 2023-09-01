<<<<<<< HEAD
### [Develpment]
=======
### [Develpment stage]
>>>>>>> f573081 (added more items)
>Pipeline
- [x] Define CLI
- [x] Define Images Model
- [x] Define Text Model
- [ ] Fill out CITATION.cff
- [ ] Review LICENSE
<<<<<<< HEAD
- [ ] Pypi documentation for detailed explanation of major classes and how-to-use

>Dataset
- [ ] Choose a demonstration dataset for images
- [x] Choose a demonstration dataset for text
- [x] (Optional) Script exists to make it into hierarchical folder structure
- [ ] (Optinal) Review and choose datasets with hierarchy for other domains (E.g. Audio)
- [ ] (Optional) Review and implement: dataset with Imbalanced Leaf Nodes
=======
- [ ] Docs for detailed explanation of major class and how-to-use

>Dataset
- [ ] Choose a demonstration dataset for images
- [ ] Choose a demonstration dataset for text
- [x] Collect more text datasets with commercial use license
- [x] (Optional) Script exists to make it into hierarchical folder structure
>>>>>>> f573081 (added more items)

>Training & Evaluation
- [ ] Choose and define which loss and classifer will be used
- [ ] Define training process (E.g., Save best weights at best-epoch when eval-metric was the best. Enable early stopping. Tidy but detailed-enough logging)
- [ ] Define the evaluation tool and implement evaluation function: show some metric or plots
<<<<<<< HEAD
- [ ] Enable fine-tuning for faster training (Just simply  loading pretrained weights will not fit for downstream task with different labels)
- [ ] (Optional) If datasets other than text and image are chosen, an appropriate model for this task should be defined and implemented for training.
=======
>>>>>>> f573081 (added more items)

>Inference
- [ ] Set up cloud storage to upload pre-trained weights and enable `download` command
- [ ] Define API for use as a library

<<<<<<< HEAD
### [Deployment]
=======
### [Deployment stage]
>>>>>>> f573081 (added more items)
- [ ] Publish to PyPi
- [ ] Move repository to GitHub
- [ ] Set up a GitHub Actions pipeline to build and publish wheels
- [ ] Build a live demo
- [ ] Announce on social media
- [ ] Add tools to export to other formats (e.g. ONNX)
<<<<<<< HEAD
=======

### [Issues]
- [x] Not allowing multiple parents. It would be a nice-to-have feature to allow this in the code, but ideally users shouldn't have the same label name under different hierarchy in their tree.
Sometimes, in text classification dataset, it seems that some categories may be put under different higher category. However, current hiercls code doesn't allow this. In L197 in hier.py, it raises value error not allowing the same name in another hierachy. (E.g., feeding=>gift_sets | bathing_skin_care=>gift_sets). This might occur reasonably because the model can just predict it as gift_sets regardless of whether it's put under whatever super-category. However, the strict tree hierachical structure should treat them as different label (e.g., gift_sets1, gift_sets2). An attempt to deal with this as a workaround is to make them distinct when creating a dataset (`make_folders_with_hierarchy.py`).


### [Miscellaneous]
- [x] data-typing for arguments passed to cli to avoid error (e.g., learning-rate)
>>>>>>> f573081 (added more items)
