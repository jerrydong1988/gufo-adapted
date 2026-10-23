// WINDOWS: Run the upstream pinned-budget contracts independently of its
// POSIX disk-orphan fixture. Keep all original assertions and test inputs.
#define main UpstreamComponentCacheMain
#include "tests/cli/component_text_cache_test.cpp"
#undef main

int main() {
  PinnedSlotBytesCountOnce();
  UnpinnedRunnersKeepTheirLimit();
}
