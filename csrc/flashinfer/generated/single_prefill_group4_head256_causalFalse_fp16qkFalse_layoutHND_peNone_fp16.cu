#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 4, 256, false, false, QKVLayout::kHND, PosEncodingMode::kNone)