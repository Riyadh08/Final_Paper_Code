
class Config:
    """AdaLog backend settings for the COCO detection experiments (W4/A4).

    Same search hyper-parameters as ``configs/4bit.py``; only the calibration
    sizes differ, because a single COCO image at 800x1344 carries roughly two
    orders of magnitude more tokens than a 224x224 ImageNet crop.

    ``calib_batch_size`` is a construction-time default only -- the detection
    calibrator (``utils/det_calibrator.py``) replaces it per module with a
    chunk size derived from that module's own activation shape.
    """
    def __init__(self):
        # calibration settings
        self.calib_size = 4
        self.optim_size = 32
        self.calib_batch_size = 1
        self.optim_batch_size = 1
        self.w_bit = 4
        self.a_bit = 4
        self.s_bit = 4
        self.qconv_a_bit = 8
        self.qhead_a_bit = 4
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
