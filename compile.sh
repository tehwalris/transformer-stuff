#!/usr/bin/env bash
set -eu

g++ \
  -O3 \
  -march=native \
  -mavx2 \
  -mfma \
  -ffast-math \
  matrix_multiplication.cpp