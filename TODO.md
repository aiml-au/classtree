### Documentation
- [ ] Review LICENSE
- [ ] Pydocs
- [ ] Getting Started Guide

### Datasets
- [x] Choose a demonstration dataset for images
- [x] Choose a demonstration dataset for text
- [ ] Dataset balancing
- [ ] Stratified validation splits
- [ ] Collect and prepare more datasets
- [ ] Allow for manifest files instead of directories

### Benchmarking
- [ ] Create a benchmarking script
- [ ] Enable fine-tuning for faster training (Just simply  loading pretrained weights will not fit for downstream task with different labels)
- [ ] (Optional) If datasets other than text and image are chosen, an appropriate model for this task should be defined and implemented for training.

### Inference
- [ ] Set up cloud storage to upload pre-trained weights and enable `download` command
- [ ] Define API for use as a library

### Models
- [ ] Upgrade ResNet architecture to something more modern
- [ ] Upgrade RoBERTa architecture to something more modern
- [ ] Pre-train models at different sizes
- [ ] Pre-train models on new datasets

### Export
- [ ] Add tools to export to other formats (e.g. ONNX)

### Publishing
- [ ] Publish to PyPi
- [ ] Move repository to GitHub
- [ ] Set up a GitHub Actions pipeline to build and publish wheels
- [ ] Build a live demo
- [ ] Announce on social media
