import math

class config:

    def __init__(self):

        # # Mix version
        # self.anchor_ratio = [1., 0.5, 2.]
        # self.anchor_scale = [8., 16., 32.]
        # self.downscale_ratio = 32

        # Normal version
        self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
        self.anchor_box_scales = [128, 256, 512]
        self.downscale_ratio = 32

        self.horizontal_flips = True
        self.vertical_flips = True
        self.rot = True

        self.positive_thr = 0.7
        self.negative_thr = 0.3

        self.num_samples = 256
        self.pos_ratio = 0.5