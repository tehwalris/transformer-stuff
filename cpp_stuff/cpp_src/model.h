#pragma once

namespace cml
{

    class SimpleTransformerLayer
    {
    public:
        virtual ~SimpleTransformerLayer() {}
        virtual void forward(const int n_in, const float *hidden_in, const int n_out, float *hidden_out) = 0;
        virtual void reset() = 0;
    };

    void simple_transformer_layer_delete(SimpleTransformerLayer *layer);
    void simple_transformer_layer_forward(SimpleTransformerLayer *layer, const int n_in, const float *hidden_in, const int n_out, float *hidden_out);
    void simple_transformer_layer_reset(SimpleTransformerLayer *layer);
};