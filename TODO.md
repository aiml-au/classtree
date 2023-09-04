### [Develpment]
>Pipeline
- [x] Define CLI
- [x] Define Images Model
- [x] Define Text Model
- [ ] Fill out CITATION.cff
- [ ] Review LICENSE
- [ ] Pypi documentation for detailed explanation of major classes and how-to-use

>Dataset
- [ ] Choose a demonstration dataset for images
- [x] Choose a demonstration dataset for text
- [x] (Optional) Script exists to make it into hierarchical folder structure
- [ ] (Optinal) Review and choose datasets with hierarchy for other domains (E.g. Audio)
- [ ] (Optional) Review and implement: dataset with Imbalanced Leaf Nodes

>Training & Evaluation
- [ ] Choose and define which loss and classifer will be used
- [ ] Define training process (E.g., Save best weights at best-epoch when eval-metric was the best. Enable early stopping. Tidy but detailed-enough logging)
- [ ] Define the evaluation tool and implement evaluation function: show some metric or plots
- [ ] Enable fine-tuning for faster training (Just simply  loading pretrained weights will not fit for downstream task with different labels)
- [ ] (Optional) If datasets other than text and image are chosen, an appropriate model for this task should be defined and implemented for training.

>Inference
- [ ] Set up cloud storage to upload pre-trained weights and enable `download` command
- [ ] Define API for use as a library

### [Deployment]
- [ ] Publish to PyPi
- [ ] Move repository to GitHub
- [ ] Set up a GitHub Actions pipeline to build and publish wheels
- [ ] Build a live demo
- [ ] Announce on social media
- [ ] Add tools to export to other formats (e.g. ONNX)
