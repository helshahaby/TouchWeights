def calculate_reward(
        prediction,
        expected):


    prediction=str(prediction).strip()

    expected=str(expected).strip()


    if expected in prediction:

        return 1


    return -1