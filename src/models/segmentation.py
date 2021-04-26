from torch import Tensor
from omegaconf import OmegaConf

from src.models import AbstractHydraModule


class SemanticSegmentationModule(AbstractHydraModule):

    def __init__(self, module: OmegaConf, loss: OmegaConf, metrics: OmegaConf, optimizer: OmegaConf,
                 scheduler: OmegaConf, sync_dist: bool = True, **kwargs):
        super().__init__(module, loss, metrics, optimizer, scheduler, sync_dist=sync_dist, **kwargs)
        self.last_logits = None

    def forward(self, batch: Tensor) -> Tensor:
        return self.model(batch)

    def training_step(self, batch, _batch_index):
        image, target = batch
        logits = self.model(image)
        self.last_logits = logits
        loss = self.loss(logits, target.long())
        self.log("train/loss", loss, on_step=True, sync_dist=self.sync_dist)
        self.log_dict(self.train_metrics(logits, target), on_step=True, sync_dist=self.sync_dist)
        return {"loss": loss, "preds": logits, "target": target}

    def validation_step(self, batch, _batch_index):
        image, target = batch
        logits = self.model(image)
        loss = self.loss(logits, target.long())
        self.log("valid/loss", loss, on_epoch=True, sync_dist=self.sync_dist)
        self.log_dict(self.valid_metrics(logits, target), on_epoch=True, sync_dist=self.sync_dist)
        return {"loss": loss, "preds": logits, "target": target}

    def test_step(self, batch, _batch_index):
        image, target = batch
        logits = self.model(image)
        loss = self.loss(logits, target.long())
        self.log_dict(self.test_metrics(logits, target), on_epoch=True, sync_dist=self.sync_dist)
        return {"loss": loss}
