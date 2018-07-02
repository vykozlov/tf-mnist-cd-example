#!/usr/bin/env bash

# Script to run pylint static code analysis
#
# Created on Mon Jul  2 16:16:37 2018
# @author: valentin

pylint $(find . -maxdepth 4 -name "*.py") --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}"
