class DataConfig(object):
    def __init__(self, **kwargs):
        self.raw_data_path_prefix = kwargs.pop('raw_data_path_prefix',
                                               '/home/admin/桌面/MICRecBox/SeqRec/dataset/raw_data/')
        self.clean_data_path_prefix = kwargs.pop('clean_data_path_prefix',
                                                 '/home/admin/桌面/MICRecBox/SeqRec/dataset/clean_data/')
        self.data_family = kwargs.pop('data_family', 'lbsn/')
        self.data_name = kwargs.pop('data_name', 'nyc')
        self.raw_data_path = kwargs.pop('raw_data_path',
                                        self.raw_data_path_prefix + self.data_family + self.data_name + '.inter')
        self.clean_data_path = kwargs.pop('clean_data_path',
                                          self.clean_data_path_prefix + self.data_family + self.data_name + '.data')
        self.cold_item_threshold = kwargs.pop('cold_item_threshold', 20)
        self.cold_user_threshold = kwargs.pop('cold_user_threshold', 10)
        self.max_len = kwargs.pop('max_len', 64)
