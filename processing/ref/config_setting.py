class config :

    def __init__(self):

        # Anchor box setting
        self.anchor_box_ratio = [0.5, 1., 1.5]
        self.anchor_box_scale = [16., 32., 64.]
        self.downscale_ratio = 16

        # Augmentation setting
        self.augmentation_bool = True
        self.horizontal_bool = True
        self.vertical_bool = True
        self.rotate_bool = True

        # Positive / Negative thr
        self.positive_thr = 0.7
        self.negative_thr = 0.3