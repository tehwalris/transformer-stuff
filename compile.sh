#!/usr/bin/env bash
set -eu

g++ \
  -Wl,--no-as-needed \
  -lpthread \
  -lm \
  -ldl \
  -O3 \
  -march=native \
  -ffast-math \
  matrix_multiplication.cpp
