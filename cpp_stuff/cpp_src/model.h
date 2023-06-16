#pragma once

namespace cml
{

    class SimpleTransformerLayer
    {
    public:
        virtual ~SimpleTransformerLayer() {}
        virtual void forward(int n, const float *hidden_in, float *hidden_out) = 0;
        virtual void reset() = 0;
    };

    void simple_transformer_layer_delete(SimpleTransformerLayer *layer);
    void simple_transformer_layer_forward(SimpleTransformerLayer *layer, int n, const float *hidden_in, float *hidden_out);
    void simple_transformer_layer_reset(SimpleTransformerLayer *layer);
};