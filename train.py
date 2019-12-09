from remsen.data import Dataset
from remsen.models import unet
from remsen.training import Trainer

dataset = Dataset()

if False:
    model = unet(img_channels=1, loss="iou_loss")
    trainer = Trainer(name="lidar_jaccard_loss", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif True:
    model = unet(img_channels=1, loss="dice_loss")
    trainer = Trainer(name="lidar_dice_loss", model=model, dataset=dataset)
    trainer.train(epochs=100)
elif False:
    model = unet(img_channels=1, loss="iou_loss", dropout=False)
    trainer = Trainer(name="lidar_without_dropout", model=model, dataset=dataset)
    trainer.train(epochs=100)
