import numpy as np

'''
# --- Level definition ---
    gt_func = lambda x: np.sin(x**2)
    gt_xs = [-1, 0, 1, 2]
    gt_points = [Enemy(x, gt_func(x)) for x in gt_xs]
    hit_thresh = 1e-2  # how close the function value must be to hit
    tree_depth = 5 # 2 is min, 5 is max
'''

class Level:
    def __init__(self, depth, gt_func, gt_xs, hit_thresh = 1e-2, spellbook = ['x', '1', '2', '+', '-', '*', '/', 'sin', 'cos']):
        self.depth = depth
        self.gt_func = gt_func
        self.gt_xs = gt_xs
        self.gt_points = [(x, gt_func(x)) for x in gt_xs]
        self.hit_thresh = hit_thresh
        self.spellbook = spellbook

class LevelManager:
    def __init__(self):
        self.levels = []
        self.current = 0

    def add_level(self, level):
        self.levels.append(level)

    def get_current_level(self):
        if 0 <= self.current < len(self.levels):
            return self.levels[self.current]
        return None

    def next_level(self):
        if self.current < len(self.levels) - 1:
            self.current += 1
            return True
        return False  # no more levels
    
    def prev_level(self):
        if self.current > 0:
            self.current -= 1
            return True
        return False  # already at first level
    
    def has_next_level(self):
        return self.current < len(self.levels) - 1
    
    def has_prev_level(self):
        return self.current > 0

    def reset(self):
        self.current = 0
