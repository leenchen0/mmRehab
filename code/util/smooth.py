import bisect

class ExponentialSmoothing:
    def __init__(self, alpha):
        self.alpha = alpha
        self.smoothed_value = None

    def update(self, new_value):
        if self.smoothed_value is None:
            self.smoothed_value = new_value
        else:
            self.smoothed_value = self.alpha * new_value + (1 - self.alpha) * self.smoothed_value
        return self.smoothed_value

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def update(self, new_value):
        self.values.append(new_value)
        self.sum += new_value

        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)

        return self.sum / len(self.values)

class MedianFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = []

    def update(self, new_value):
        bisect.insort(self.window, new_value)

        if len(self.window) > self.window_size:
            oldest_value = self.window.pop(0)

        mid_index = len(self.window) // 2
        if len(self.window) % 2 == 0:
            return (self.window[mid_index - 1] + self.window[mid_index]) / 2.0
        else:
            return self.window[mid_index]
