class LinearApproximator(torch.nn.Module):
    """
    Simple linear approximator
    """

    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.head = torch.nn.Linear(10, 4)

    def forward(self, X):
        """
        Architecture of DQN
        """
        return self.head(X)
