# Model Card

## Purpose

The detector is intended to identify target wildlife classes in camera footage, with emphasis on red squirrel monitoring and reducing false positives from visually similar animals.

## Intended use

- Scan folders of recorded camera footage.
- Flag videos or frames containing target detections.
- Support wildlife monitoring and human review workflows.
- Compare several detector architectures in a reproducible training run.

## Not intended for

- Real-time safety-critical decisions.
- Species population estimates without additional ecological validation.
- Fully automated ecological conclusions without human review.

## Classes

Expected classes depend on the dataset passed to training. The recommended merged squirrel pipeline uses these canonical classes:

```text
red
grey
marten
rat
```

The `rat` class is included as a confuser class to help the detector distinguish rats from red squirrels.

## Training inputs

The training pipeline expects a YOLO-style object detection dataset with:

```text
data.yaml
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

Multiple datasets can be merged before training when their class names are mapped into one canonical class list.

## Known limitations

- Performance depends on video quality, lighting, camera angle, and species visibility.
- False positives and false negatives are expected.
- Results should be reviewed by a human before operational use.
- Model weights are not included in the public repository.

## Evaluation

Add model performance results here when publishing a trained model release. Recommended metrics include mAP50-95, mAP50, precision, recall, confusion matrix review, and targeted inspection of rat/red-squirrel false positives.
