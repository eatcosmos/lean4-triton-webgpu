// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: check that intel_reqd_sub_group_size is added to functions
module attributes { "triton_gpu.threads-per-warp" = 16 : i32, "triton_gpu.num-warps" = 4 : i32 } {
// CHECK: llvm.func spir_kernelcc @func_name() attributes {intel_reqd_sub_group_size = 16 : i32, triton_gen.max_work_group_size = [64 : i32, 1 : i32, 1 : i32]} {
  tt.func public @func_name() {
    tt.return
  }
}
