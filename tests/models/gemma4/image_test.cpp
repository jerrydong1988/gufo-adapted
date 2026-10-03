#include <cassert>
#include <filesystem>
#include <iostream>

#include "src/core/crypto/sha256.hpp"
#include "src/models/gemma4/vision.hpp"

int main() {
  using namespace gufo;
  const auto input = core::DecodeImage(core::ReadImageFile(
      (std::filesystem::path(GEMMA4_FIXTURES) / "shapes.png").string()));
  const auto resized = models::gemma4::PreprocessImage(input);
  assert(resized.width == 576 && resized.height == 288);
  // Pinned llama.cpp/mtmd Pillow RGB8 output, independently captured before
  // implementing the native encoder. The complete RGB image is compared.
  assert(crypto::Sha256Hex(resized.pixels) ==
         "35ec5330cf0686eb5cbe93b091d7bdc13fcdfd9ada78992def2900f9dde9c3ae");
  for (const auto dimensions : {std::pair{48, 48}, {123, 97}, {1920, 1080}}) {
    core::Image pixels;
    pixels.width = dimensions.first;
    pixels.height = dimensions.second;
    pixels.pixels.resize(std::size_t(pixels.width) * pixels.height * 3, 127);
    const auto output = models::gemma4::PreprocessImage(pixels);
    const auto tokens = (output.width / 48) * (output.height / 48);
    assert(output.width % 48 == 0 && output.height % 48 == 0);
    assert(tokens >= 70 && tokens <= 280);
    assert(output.pixels.size() ==
           std::size_t(output.width) * output.height * 3);
  }
  std::cout << "PASS: independent RGB8 fixture and bounded patch geometry\n";
}
