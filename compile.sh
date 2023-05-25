#!/usr/bin/env bash
set -eu

g++ \
  -Wl,--no-as-needed \
  -lpthread \
  -lm \
  -ldl \
  -O3 \
  -march=native \
  -mavx2 \
  -mfma \
  -ffast-math \
  matrix_multiplication.cpp