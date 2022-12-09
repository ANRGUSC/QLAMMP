
class Swap():
    def __init__(self, tolerance, init_urgency, userNum):
        self.tolerance = tolerance  # 0.1% to 5%
        self.urgency = init_urgency # ln(1) to ln(2)
        self.userNum = userNum      # 0 to numUsers - 1
        self.amt = None
        self.idx = None
        self.new = True
        self.tries = 0