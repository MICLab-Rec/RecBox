class SASRecConfig(object):
    def __init__(self, **kwargs):
        self.dim = kwargs.pop('dim', 128)
        self.max_len = kwargs.pop('max_len', 64)
        self.dropout_ratio = kwargs.pop('dropout_ratio', 0.5)
        self.num_head = kwargs.pop('num_head', 4)
        self.exp_factor = kwargs.pop('exp_factor', 4)
        self.depth = kwargs.pop('depth', 2)
