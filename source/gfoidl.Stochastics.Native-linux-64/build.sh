#!/bin/bash

clang++ -o libgfoidl-Stochastics-Native.so -Ofast -std=c++11 -shared -fPIC special_functions.cpp
