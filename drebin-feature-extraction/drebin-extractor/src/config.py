#!/usr/bin/env python
#
#########################################################################################
#                                     Disclaimer                                        #
#########################################################################################
# (c) 2014, Mobile-Sandbox
# Michael Spreitzenbarth (research@spreitzenbarth.de)
#
# This program is free software you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#########################################################################################
#                          Imports  & Global Variables                                  #
#########################################################################################
import platform

# important files and folders
TMPDIR = "/scratch1/NOT_BACKED_UP/cavallarogrp/datasets/processed_dataset/Androzoo_info/drebin_feature_22_23/"
AAPT = "/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/tools/aapt-macos" if platform.system() == 'Darwin' else "/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/tools/aapt-linux"  # location of the aapt-linux binary
APICALLS = "/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/src/APIcalls.txt"
BACKSMALI = "/cs/academic/phd3/xinrzhen/xinran/drebin-feature-extraction/drebin-extractor/tools/baksmali-2.0.3.jar"  # location of the baksmali.jar file
ADSLIBS = "src/lib/ads.csv"
