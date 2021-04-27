import io
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import Image
from typing import Any, Dict

from torch import Tensor
from pytorch_lightning import Callback, LightningModule, Trainer

from src.utils.ml import mask_to_rgb


class SegmentationPlotsCallback(Callback):  # pragma: no cover

    def __init__(
        self,
        color_palette: Dict[int, tuple],
        channels_first: bool = True,
        logging_batch_interval: int = 20,
    ):
        super().__init__()
        self.channels_first = channels_first
        self.logging_batch_interval = logging_batch_interval

    def _rgb(self, image: Tensor) -> Tensor:
        img = image.detach().cpu().numpy()
        if self.channels_first:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        return img[:, :, :3]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # show images only every N batches
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:
            return
        # pick the last batch and logits
        images, masks = batch
        preds = pl_module.last_logits.argmax(dim=1)
        batch_size = preds.shape[0]
        # prepare canvas
        fig, axarr = plt.subplots(nrows=3, ncols=batch_size, figsize=(15, 10))
        plt.tight_layout()
        for index in range(batch_size):
            image = self._rgb(images[index])
            mask = mask_to_rgb(masks[index].detach().cpu().numpy(), palette=self.color_palette)
            pred = mask_to_rgb(preds[index].detach().cpu().numpy(), palette=self.color_palette)
            self.__draw_sample(fig, axarr, row_idx=0, col_idx=index, img=image, title=f"input_{batch_idx}_{index}")
            self.__draw_sample(fig, axarr, row_idx=1, col_idx=index, img=mask, title=f"true_{batch_idx}_{index}")
            self.__draw_sample(fig, axarr, row_idx=2, col_idx=index, img=pred, title=f"pred_{batch_idx}_{index}")
        # send to logger
        trainer.logger.experiment.add_figure("segmentations", fig, global_step=trainer.global_step)
        plt.close(fig)

    @staticmethod
    def __draw_sample(fig: Figure, axarr: Axes, row_idx: int, col_idx: int, img: Tensor, title: str) -> None:
        axarr[row_idx, col_idx].imshow(img)
        axarr[row_idx, col_idx].get_xaxis().set_visible(False)
        axarr[row_idx, col_idx].get_yaxis().set_visible(False)
        axarr[row_idx, col_idx].set_title(title, fontsize=18)

    @staticmethod
    def _convert(fig: Figure) -> Image:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        return Image.open(buffer)
