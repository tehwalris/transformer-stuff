#pragma once

#include <cstdint>

namespace cml
{

    struct LlamaHyperparams
    {
        uint32_t n_hidden;
        uint32_t n_context;
        uint32_t n_heads;
        uint32_t n_ff;
    };

    class SimpleTransformerLayer
    {
    public:
        virtual ~SimpleTransformerLayer() {}
        virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path) = 0;
        virtual uint32_t next_i() const = 0;
        virtual void retain(const uint32_t n_retain, const uint32_t *retain) = 0;
    };

    void simple_transformer_layer_delete(SimpleTransformerLayer *layer);
    void simple_transformer_layer_forward(SimpleTransformerLayer *layer, const int n_in, const float *hidden_in, const int n_out, float *hidden_out, const uint32_t n_path, const uint32_t *path);
    uint32_t simple_transformer_layer_next_i(const SimpleTransformerLayer *layer);
    void simple_transformer_layer_retain(SimpleTransformerLayer *layer, const uint32_t n_retain, const uint32_t *retain);

};