#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 1, 256, false, true, QKVLayout::kNHD, PosEncodingMode::kALiBi)