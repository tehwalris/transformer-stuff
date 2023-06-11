#pragma once

#include <cstdint>

class CarrierInterface
{
public:
  virtual ~CarrierInterface(){};
  virtual void hidden_in_thing(int n, float *hidden_in) noexcept = 0;
};

CarrierInterface *create_carrier(uint32_t n_hidden);