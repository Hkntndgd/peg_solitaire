# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:57:18 2022

@author: htand_000
"""

import util
import numpy as np

#To start from scratch:
learning_list = util.learn() # default episode = 500
ones = util.learning_result(learning_list)

# or you can use an exixting ones 

with open('ones.npy', 'rb') as f:
    ones = np.load(f)
    
underlying_axe_row_wise,underlying_axe_column_wise = util.weak_link(ones)
n,motion_dir = util.play_until_success(ones, 0.65, underlying_axe_row_wise, underlying_axe_column_wise)
game=motion_dir
util.plot_a_succesfull_game(game)
util.create_a_gif(game)
