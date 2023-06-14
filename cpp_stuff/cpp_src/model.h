#pragma once

namespace cml
{

    class SimpleTransformerLayer
    {
    public:
        virtual ~SimpleTransformerLayer() {}
        virtual void forward(int n, float *hidden_in, float *hidden_out) = 0;
        virtual void reset() = 0;
    };

    void delete_simple_transformer_layer(SimpleTransformerLayer *layer)
    {
        delete layer;
    }
};