// Independent image preprocessing/encoder oracle from the pinned mtmd build.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "clip-impl.h"
#include "clip-model.h"
#include "clip.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-image.h"

struct CaptureState {
  std::filesystem::path directory;
};
static bool Capture(ggml_tensor* tensor, bool ask, void* opaque) {
  if (ask && tensor->op == GGML_OP_MUL_MAT)
    ggml_prec_set_acc(tensor, GGML_PREC_F32);
  const std::string name = ggml_get_name(tensor);
  const bool wanted =
      name == "inp" || name == "pos_embd" ||
      (name.starts_with("layer_out-") && name.find(' ') == std::string::npos) ||
      name == "std_scaled" || name == "projected" ||
      name == "layer_inp_normed-0" || name == "Qcur_pos-0" ||
      name == "Kcur_pos-0" || name == "Vcur_normed-0" || name == "attn_out-0" ||
      name == "ffn_inp_normed-0" || name == "ffn_gate-0" ||
      name == "ffn_up-0" || name == "ffn_geglu-0" || name == "ffn_out-0";
  if (ask)
    return wanted;
  if (wanted) {
    if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float))
      throw std::runtime_error("unexpected encoder reference layout: " + name);
    std::vector<float> data(ggml_nelements(tensor));
    std::size_t cursor = 0;
    for (int i3 = 0; i3 < tensor->ne[3]; ++i3)
      for (int i2 = 0; i2 < tensor->ne[2]; ++i2)
        for (int i1 = 0; i1 < tensor->ne[1]; ++i1) {
          ggml_backend_tensor_get(
              tensor, data.data() + cursor,
              i3 * tensor->nb[3] + i2 * tensor->nb[2] + i1 * tensor->nb[1],
              tensor->ne[0] * sizeof(float));
          cursor += tensor->ne[0];
        }
    if (!std::all_of(data.begin(), data.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("non-finite vision reference");
    std::ofstream out(
        static_cast<CaptureState*>(opaque)->directory / (name + ".bin"),
        std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float));
  }
  return true;
}
int main(int argc, char** argv) {
  try {
    if (argc != 6)
      throw std::invalid_argument(
          "usage: llama-vision-reference MMPROJ RGB WIDTH HEIGHT DIRECTORY");
    CaptureState capture{argv[5]};
    std::filesystem::create_directories(capture.directory);
    llama_backend_init();
    clip_context_params params{};
    params.use_gpu = true;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.image_min_tokens = 70;
    params.image_max_tokens = 280;
    params.cb_eval = Capture;
    params.cb_eval_user_data = &capture;
    auto loaded = clip_init(argv[1], params);
    if (!loaded.ctx_v)
      throw std::runtime_error("vision load failed");
    // These pinned projectors omit clip.use_gelu. Google's pinned 31B config
    // specifies gelu_pytorch_tanh; the generic CLIP fallback is GELU_QUICK.
    // Select the architecture's activation, without changing any weights.
    const_cast<clip_hparams*>(clip_get_hparams(loaded.ctx_v))->ffn_op =
        FFN_GELU;
    std::cerr << "Gemma 4 vision activation: GELU tanh (Google config "
                 "419b2efe421994fdfd3394e621983d4cc511cd4f)\n";
    const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
    std::ifstream input(argv[2], std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                    {});
    if (bytes.size() != std::size_t(width) * height * 3)
      throw std::invalid_argument("invalid RGB fixture");
    clip_image_u8 image;
    image.set_size({width, height}, false);
    image.cpy_buf(bytes);
    mtmd_image_preprocessor_dyn_size preprocessor(loaded.ctx_v);
    auto images = preprocessor.preprocess(image);
    if (images.entries.size() != 1)
      throw std::runtime_error("unexpected reference image tiling");
    const auto& pixels = images.entries[0];
    std::ofstream dimensions(capture.directory / "dimensions.txt");
    dimensions << pixels.nx() << ' ' << pixels.ny() << '\n';
    dimensions.close();
    std::ofstream normalized(capture.directory / "pixels.f32",
                             std::ios::binary);
    normalized.write(reinterpret_cast<const char*>(pixels.get_ro_buf().data()),
                     pixels.n_elements() * sizeof(float));
    normalized.close();
    std::vector<float> output(clip_n_output_tokens(loaded.ctx_v, &pixels) *
                              5376);
    if (!clip_image_encode(loaded.ctx_v, 8, &pixels, output))
      throw std::runtime_error("vision encode failed");
    if (output.empty() ||
        !std::all_of(output.begin(), output.end(),
                     [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("invalid reference image embedding");
    clip_free(loaded.ctx_v);
    llama_backend_free();
    std::cout << "tokens=" << output.size() / 5376 << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
