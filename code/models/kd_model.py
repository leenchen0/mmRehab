import torch.nn as nn

from feature_extractors.mmrehab_kd import MmRehabKD

class Model(nn.Module):
    def __init__(self, config, in_channels, out_channels, num_domains=0):
        super(Model, self).__init__()

        backbone = config.model.backbone
        self.backbone = backbone
        self.estimator = (
            MmRehabKD(config, 1, out_channels) if backbone == 'mmrehab_kd' or backbone == 'mmfreshv2' else
            None
        )

    def forward(self, *args, **kwargs):
        res = self.estimator(*args, **kwargs)
        return res

    def forward_discriminator(self, pose, shape):
        if hasattr(self.estimator, 'forward_discriminator'):
            return self.estimator.forward_discriminator(pose, shape)
        return None
