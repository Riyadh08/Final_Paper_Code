
class Config:
    """AdaLog backend settings for the COCO detection experiments (W6/A6).

    See ``configs/4bit_det.py`` for why the calibration sizes differ from the
    ImageNet configs.
    """
    def __init__(self):
        # calibration settings
        self.calib_size = 4
        self.optim_size = 32
        self.calib_batch_size = 1
        self.optim_batch_size = 1
        self.w_bit = 6
        self.a_bit = 6
        self.s_bit = 6
        self.qconv_a_bit = 8
        self.qhead_a_bit = 6
        self.matmul_head_channel_wise = True
        self.post_softmax_quantizer = 'adalog'
        self.post_gelu_quantizer = 'adalog'
        # search settings
        self.eq_n = 128
        self.search_round = 3
        self.fpcs = True
        self.steps = 6
        # optimization settings
        self.keep_gpu = True
        self.train_act = True
