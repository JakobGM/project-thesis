# Project Thesis Notes

## Caching

* 0.66 seconds per cadastre used for caching tiles

## Testing

* Testing tiles may partially overlap with training tiles

## Data augmentation

* Erlend says that data augmentation often improves the model a lot. Should be done earlier as previously expected.
* Should I rotate 90, 180, 270 for all tiles, or just a random subset of tiles
  and/or rotations?
* Should the original image be more heavily "weighted" in the augmentation?
* Should image augmentation change between each epoch?

__With augmentation__:

* loss: 0.0186
* mean_io_u: 0.43572019
* val_loss: 0.02970 (45)
* val_mean_io_u: 0.45318 (48)

__Without augmentation__:

* loss: 0.0190
* mean_io_u: 0.43572019
* val_loss: 0.02981 (46)
* val_mean_io_u: 0.43740 (43)

4 days of training results in

```
[0.026403871428443318, 0.5874174]
```

## Trees in data

* 26070
* 11 - detects that there is a building underneath, but not the exact shape

## Architecture

* Dataset should provide validation data splits
* Add `__len__` for showing number of tiles
* Add `__getitem__` for specific tile extraction
* Consider dumping splits into CSV for complete reproducibility

## Fixes

* Cadastre 47159 - wrong scaling in plot
* Remove nodata from plot_prediction and plot_tiles
* Should I add random noise for nodata regions?

## Visualization

* Show nodata regions in all plots

## Questions

* Is nodata = 0 really the best solution?

## Writing

* Organize thesis according to ETL (extract, transform, load)

# Extra

* Set mask value of partial overlapping pixels to proportion of overlap? Uncertainty in ground truth.
* Write about cadastre being mainly used to extract data of interest and is the basis for our sample space
