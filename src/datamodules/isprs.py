from src.datamodules import SegmentationDataModule


class VaihingenDataModule(SegmentationDataModule):

    def __init__(self,
                 image_height: int,
                 image_width: int,
                 image_channels: int,
                 include_dsm: bool,
                 *args, **kwargs):
        channels = 4 if include_dsm else 3
        assert channels == image_channels, \
            f"Declared channel count ({image_channels}) differs from actual count ({channels})."
        dims = (channels, image_height, image_width)
        # TODO: check why the name is RGB instead of IRRG
        extra_params = dict(channel_names="rgb", channel_count=channels, include_dsm=include_dsm)
        super().__init__(*args, **kwargs, dims=dims, dataset_params=extra_params)


class PotsdamDataModule(SegmentationDataModule):

    def __init__(self,
                 image_height: int,
                 image_width: int,
                 image_channels: int,
                 include_dsm: bool,
                 include_ir: bool,
                 *args, **kwargs):
        channels = 3
        if include_dsm:
            channels += 1
        if include_ir:
            channels += 1
        assert channels == image_channels, \
            f"Declared channel count ({image_channels}) differs from actual count ({channels})."
        dims = (channels, image_height, image_width)
        extra_params = dict(channel_names="rgbir", channel_count=channels, include_dsm=include_dsm)
        super().__init__(*args, **kwargs, dims=dims, dataset_params=extra_params)
