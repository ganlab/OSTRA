# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/run_system.py

import json
import argparse
import time
import datetime
import os, sys
from os.path import isfile

import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rgbdreconstruction.initialize_config import initialize_config


def reconstruction_with_RGBD(config_path, dataset_path):


    # check folder structure
    if config_path is not None:
        with open(config_path) as json_file:
            config = json.load(json_file)
            initialize_config(config)
    assert config is not None

    config['debug_mode'] = True
    config['device'] = "cpu:0"
    config['path_dataset'] = dataset_path

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0, 0, 0, 0, 0, 0]
    start_time = time.time()
    from rgbdreconstruction import make_fragments
    make_fragments.run(config)
    times[0] = time.time() - start_time

    start_time = time.time()
    from rgbdreconstruction import register_fragments
    register_fragments.run(config)
    times[1] = time.time() - start_time

    start_time = time.time()
    from rgbdreconstruction import refine_registration
    refine_registration.run(config)
    times[2] = time.time() - start_time

    start_time = time.time()
    from rgbdreconstruction import integrate_scene
    integrate_scene.run(config)
    times[3] = time.time() - start_time


    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- SLAC                %s" % datetime.timedelta(seconds=times[4]))
    print("- SLAC Integrate      %s" % datetime.timedelta(seconds=times[5]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()
