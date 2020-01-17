from remsen.data import Dataset
from remsen.models import unet
from remsen.training import Trainer

dataset = Dataset()

if False:
    model = unet(img_channels=1, loss="iou_loss")
    trainer = Trainer(name="lidar_jaccard_loss", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1, loss="dice_loss")
    trainer = Trainer(name="lidar_dice_loss", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1, dropout=False)
    trainer = Trainer(name="lidar_without_dropout", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1)
    trainer = Trainer(name="lidar_without_augment", model=model, dataset=dataset)
    trainer.train(epochs=100, augment=False)
elif False:
    # Remember to edit input_tile_normalizer
    model = unet(img_channels=1)
    trainer = Trainer(name="lidar_metric_normalization", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1, batch_normalization=False)
    trainer = Trainer(name="lidar_without_batch_normalization", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1)
    trainer = Trainer(name="lidar_no_area_filter", model=model, dataset=dataset)
    trainer.train(epochs=89, minimum_building_area=-1)
elif False:
    model = unet(img_channels=1)
    trainer = Trainer(name="lidar_rerun", model=model, dataset=dataset)
    trainer.train(epochs=89)
elif True:
    # TODO: Run this!
    model = unet(img_channels=1)
    trainer = Trainer(name="lidar_second_rerun", model=model, dataset=dataset)
    trainer.train(epochs=89)
