import torch


class GeneralRecommender(torch.nn.Module):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    def __init__(self,config,dataGenerator):
        super(GeneralRecommender, self).__init__()
        self.device = config.Device
        self.n_items = dataGenerator.datainfo.item_num
        self.n_users = dataGenerator.datainfo.user_num

    def _xavier_uniform_initialization(self,module):
        '''
        using `xavier_uniform_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.
        @return:
        '''
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.histiory_items()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

    def get_rating_matrix(self, user):
        '''
        Get a batch of user's feature with the user's id and history interaction matrix.
        @param user: The input tensor that contains user's id, shape: [batch_size, ]
        @return: The user's feature of a batch of user, shape: [batch_size, n_items]
        '''
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1, device=self.device).repeat(
            user.shape[0], self.n_items
        )
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        # rating_matrix = torch.FloatTensor(self.history_item_matrix[user.cpu().numpy()].todense())
        # rating_matrix = rating_matrix.to(self.device)
        # # col_indices = self.history_item_id[user].flatten()
        # # row_indices = torch.arange(user.shape[0]).repeat_interleave(
        # #     self.history_item_id.shape[1], dim=0
        # # )
        # # rating_matrix = torch.zeros(1, device=self.device).repeat(
        # #     user.shape[0], self.n_items
        # # )
        # # rating_matrix.index_put_(
        # #     (row_indices, col_indices), self.history_item_value[user].flatten()
        # )
        return rating_matrix

