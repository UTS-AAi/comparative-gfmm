# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:21:29 2018

@author: Thanh Tung Khuat

Initial file for the python directory
"""

# Copyright (c) 2018, the Hyperbox-based classifier project authors.  Please see the AUTHORS file
# for details. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import torch

GPU_Computing_Threshold = 1000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
float_def = torch.cuda.FloatTensor if  torch.cuda.is_available() else torch.float
long_def = torch.cuda.LongTensor if torch.cuda.is_available() else torch.int64
is_Have_GPU = torch.cuda.is_available()
