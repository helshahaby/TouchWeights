class RewardFunction:

    def compute(self, prediction, target):

        prediction = prediction.strip()
        target = target.strip()

        if prediction == target:
            return 1.0

        return -1.0