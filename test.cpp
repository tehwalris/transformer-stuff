#include "loading.h"
#include "baseline.h"

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("usage: %s <model_path>\n", argv[0]);
    exit(1);
  }
  char *model_path = argv[1];

  SimpleLlamaModelLoader *loader = new SimpleLlamaModelLoader(model_path);
  SimpleTransformerLayer *baseline_layer = create_baseline_llama_layer(loader, 0);
}