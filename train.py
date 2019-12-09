from remsen.data import Dataset
from remsen.models import unet
from remsen.training import Trainer


if True:
    dataset = Dataset()
    model = unet(img_channels=1, loss="iou_loss")
    trainer = Trainer(name="lidar_jaccard_loss", model=model, dataset=dataset)
    trainer.train(epochs=100)
