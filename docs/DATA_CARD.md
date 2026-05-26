# Data Card

## Data sources

Training data may be acquired from Roboflow or prepared locally in a YOLO-compatible object detection format.

## Data included in this repository

The public repository does not include raw datasets, videos, trained weights, or experiment outputs.

## Expected local folders

```text
data/datasets/             Downloaded source datasets
data/merged_dataset/       Merged YOLO dataset
videos/                    Local videos for scanning
runs/                      Run outputs, logs, metrics, and model artifacts
```

These folders are ignored by Git.

## Colab folders

When running in Colab, `/content` is temporary. Use Google Drive for outputs that need to survive runtime disconnects.

```text
/content/datasets                                      Temporary dataset storage
/content/drive/MyDrive/squirrel_scanner/experiments    Persistent Colab run outputs
```

## Dataset merging

When multiple datasets are used, source labels should be mapped into one canonical class list. The recommended squirrel pipeline uses:

```text
red
grey
marten
rat
```

The merge step writes a new `data.yaml` and `merge_manifest.json` into the merged dataset folder.

## Data considerations

- Check dataset licences before redistribution.
- Do not publish private camera footage without permission.
- Keep API keys and private datasets outside the repository.
- Document Roboflow workspace, project, version, export format, and class mappings used for each trained model release.
- Review merged labels before training when combining datasets from different projects.
