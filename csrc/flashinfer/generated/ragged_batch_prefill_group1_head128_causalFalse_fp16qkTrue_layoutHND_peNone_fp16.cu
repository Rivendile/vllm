#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_half, 1, 128, false, true, QKVLayout::kHND, PosEncodingMode::kNone)