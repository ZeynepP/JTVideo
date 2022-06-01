# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pytorch_lightning

from classification import VideoClassificationLightningModule
from dataset import JTDataModule


def train():
    classification_module = VideoClassificationLightningModule()
    data_module = JTDataModule()

    trainer = pytorch_lightning.Trainer(auto_select_gpus=True)
    trainer.fit(classification_module, data_module)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
