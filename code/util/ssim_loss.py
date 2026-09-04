from piqa import SSIM

class SSIMLoss(SSIM):
    def __init__(self, *args, **kwargs):
        super(SSIMLoss, self).__init__(*args, **kwargs)


    def forward(self, x, y):
        return 1. - super().forward(x, y)

