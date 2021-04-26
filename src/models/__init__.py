import hydra
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection


class AbstractHydraModule(LightningModule):

    def __init__(self,
                 module: OmegaConf,
                 loss: OmegaConf,
                 metrics: OmegaConf,
                 optimizer: OmegaConf,
                 scheduler: OmegaConf,
                 sync_dist: bool = True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        config = self.hparams
        # instantiate models and losses
        self.model = hydra.utils.instantiate(config.module)
        self.loss = hydra.utils.instantiate(config.loss)
        # instantiate metrics as DictModule
        metrics_conf =  hydra.utils.instantiate(config.metrics)
        metrics_stages = ("train", "valid", "test")
        assert all(key in metrics_conf for key in metrics_stages), \
            f"Metrics config must contain the following sections: {metrics_stages}"

        # metrics dictionary needs to be converted from OmegaConf to python types
        metrics_dict = dict()
        for stage in metrics_stages:
            metrics_dict[stage] = {f"{stage}/{n}": m for n, m in metrics_conf[stage].items()}
        # MetricsCollections allow to compute multiple metrics in a single pass
        self.train_metrics = MetricCollection(metrics_dict["train"])
        self.valid_metrics = MetricCollection(metrics_dict["valid"])
        self.test_metrics = MetricCollection(metrics_dict["test"])
        self.sync_dist = sync_dist

    def configure_optimizers(self):
        optim = self.hparams.optimizer
        sched = self.hparams.scheduler
        optimizer = hydra.utils.instantiate(optim, params=self.model.parameters())
        scheduler = hydra.utils.instantiate(sched, optimizer=optimizer)
        return {"optimizer": optimizer, "scheduler": scheduler, "monitor": self.hparams.monitored_metric}
