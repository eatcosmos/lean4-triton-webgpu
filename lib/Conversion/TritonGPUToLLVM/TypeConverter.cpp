#include "TypeConverter.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

TritonGPUToLLVMTypeConverter::TritonGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
}

Type TritonGPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  auto ctx = type.getContext();
  auto pointeeType = type.getPointeeType();
  if (pointeeType.isa<RankedTensorType>()) {
    auto rankedTensorType = pointeeType.cast<RankedTensorType>();
    // struct { offset0, offset1, shape0, shape1, stride0,
    // stride1, base_ptr};
    auto eleType = rankedTensorType.getElementType();
    auto shape = rankedTensorType.getShape();
    SmallVector<Type, 4> types;
    // offsets
    for (size_t i = 0; i < shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 32));
    // shapes, strides
    for (size_t i = 0; i < 2 * shape.size(); ++i)
      types.push_back(IntegerType::get(ctx, 64));

    types.push_back(LLVM::LLVMPointerType::get(ctx, type.getAddressSpace()));

    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
  return LLVM::LLVMPointerType::get(ctx, type.getAddressSpace());
}

Type TritonGPUToLLVMTypeConverter::getElementTypeForStruct(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());

  if (auto mmaLayout = layout.dyn_cast<DpasEncodingAttr>()) {
    auto elemsPerThread = product<unsigned>(mmaLayout.getSizePerThread());
    return vec_ty(elemTy, elemsPerThread);
  }

  auto dotOpLayout = layout.dyn_cast<DotOperandEncodingAttr>();
  if (!dotOpLayout)
    return elemTy;

  if (auto dpasParent = dotOpLayout.getParent().dyn_cast<DpasEncodingAttr>()) {
    if (dotOpLayout.getOpIdx() == 0) {
      // For A operand, pack the bits size <=16 scalar to opaque i16 and bits size = 32 to opaque i32.
      unsigned opsPerChannel = dpasParent.getOpsPerChannel();
      switch(opsPerChannel) {
        case 4:
          elemTy = type::i32Ty(ctx);
          break;
        case 2:
        case 1:
          elemTy = type::i16Ty(ctx);
          break;
        default:
          assert(false && "unexpected ops per channel");
      };
    } else {
      // For B operand, pack the all scalar to opaque i32.
      elemTy = type::i32Ty(ctx);
    }
  }

  auto mmaParent = dotOpLayout.getParent().dyn_cast<NvidiaMmaEncodingAttr>();
  if (!mmaParent || mmaParent.isHopper())
    return elemTy;
  int bitwidth = elemTy.getIntOrFloatBitWidth();
  assert(bitwidth <= 32);
  return IntegerType::get(ctx, 32);
}

Type TritonGPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(type);

  if (auto shared_layout = layout.dyn_cast<SharedEncodingAttr>()) {
    SmallVector<Type, 4> types;
    // base ptr
    auto ptrType = LLVM::LLVMPointerType::get(ctx, 3);
    types.push_back(ptrType);
    // shape dims
    auto rank = type.getRank();
    // offsets + strides
    for (auto i = 0; i < rank * 2; i++) {
      types.push_back(IntegerType::get(ctx, 32));
    }
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }

  if (auto mmaLayout = layout.dyn_cast<DpasEncodingAttr>()) {
    unsigned numElementsPerThread = getTotalElemsPerThread(type);
    auto elemsPerThread = product<unsigned>(mmaLayout.getSizePerThread());
    SmallVector<Type, 4> types(numElementsPerThread / elemsPerThread, eltType);
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
