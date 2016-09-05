class DecisionTreeNode:
    def __init__(self, col=-1, value=None,results=None,tb=None,fb=None):
        self.tb=tb
        self.fb=fb
        self.col= col
        self.value=value
        self.results=results