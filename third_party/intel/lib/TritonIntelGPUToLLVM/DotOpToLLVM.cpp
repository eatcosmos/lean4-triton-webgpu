#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::utils::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

namespace fma_details {
LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonIntelGPUToLLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA884(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

LogicalResult convertMMA1688(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

LogicalResult convertMMA16816(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

LogicalResult convertDPAS(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          TritonIntelGPUToLLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);
LogicalResult convertWGMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                           const LLVMTypeConverter *typeConverter,
                           ConversionPatternRewriter &rewriter, Value thread);

LogicalResult convertAsyncWGMMA(triton::nvidia_gpu::DotAsyncOp op,
                                triton::nvidia_gpu::DotAsyncOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter,
                                Value thread);
} // namespace fma_details
namespace {
struct DotOpConversion : public ConvertTritonGPUOpToLLVMPattern<triton::DotOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DotOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    NvidiaMmaEncodingAttr mmaLayout = D.getType()
                                          .cast<RankedTensorType>()
                                          .getEncoding()
                                          .dyn_cast<NvidiaMmaEncodingAttr>();
    if (!isOuter && mmaLayout && supportMMA(op, mmaLayout.getVersionMajor())) {
      if (mmaLayout.isVolta())
        return fma_details::convertMMA884(op, adaptor, getTypeConverter(),
                                          rewriter);
      if (mmaLayout.isTuring())
        return fma_details::convertMMA1688(op, adaptor, getTypeConverter(),
                                           rewriter);
      if (mmaLayout.isAmpere())
        return fma_details::convertMMA16816(op, adaptor, getTypeConverter(),
                                            rewriter);
      if (mmaLayout.isHopper())
        return fma_details::convertWGMMA(op, adaptor, getTypeConverter(),
                                         rewriter, getThreadId(rewriter, loc));

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (!isOuter && D.getType()
                        .cast<RankedTensorType>()
                        .getEncoding()
                        .isa<DpasEncodingAttr>()) {
      return fma_details::convertDPAS(op, adaptor, getTypeConverter(),
                                      rewriter);
    }

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return fma_details::convertFMADot(op, adaptor, getTypeConverter(),
                                        rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct DotAsyncOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::DotAsyncOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::DotAsyncOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::DotAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    NvidiaMmaEncodingAttr mmaLayout = D.getType()
                                          .cast<RankedTensorType>()
                                          .getEncoding()
                                          .dyn_cast<NvidiaMmaEncodingAttr>();
    if (!isOuter && mmaLayout &&
        supportMMA(op.getOperand(0), mmaLayout.getVersionMajor())) {
      if (mmaLayout.isHopper()) {
        return fma_details::convertAsyncWGMMA(op, adaptor, getTypeConverter(),
                                              rewriter,
                                              getThreadId(rewriter, loc));
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotAsyncOp to LLVM.");
    }

    llvm::report_fatal_error(
        "Unsupported DotAsyncOp found when converting TritonGPU to LLVM.");
  }
};

struct DotWaitOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::nvidia_gpu::DotWaitOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::DotWaitOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::DotWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pendings = op.getPendings();
    Location loc = op.getLoc();
    if (adaptor.getInputs().size() <= 1) {
      Value intput =
          adaptor.getInputs().size() == 1 ? adaptor.getInputs()[0] : Value();
      rewriter.replaceOpWithNewOp<triton::nvgpu::WGMMAWaitGroupOp>(op, intput,
                                                                   pendings);
      return success();
    }
    std::vector<Type> types;
    // Pack the inputs into a single struct.
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().dyn_cast<LLVM::LLVMStructType>();
      if (!structType)
        return failure();
      for (Type type : structType.getBody())
        types.push_back(type);
    }
    auto packedType =
        LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    unsigned outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().dyn_cast<LLVM::LLVMStructType>();
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, structType.getBody()[i], input, i);
        packed = rewriter.create<LLVM::InsertValueOp>(
            loc, packedType, packed, value, outputStructIndex++);
      }
    }
    Value packedOutput =
        rewriter.create<triton::nvgpu::WGMMAWaitGroupOp>(loc, packed, pendings);
    // Unpack the output into the original struct types.
    SmallVector<Value> outputs;
    outputStructIndex = 0;
    for (Value input : adaptor.getInputs()) {
      auto structType = input.getType().cast<LLVM::LLVMStructType>();
      Value unpacked = rewriter.create<LLVM::UndefOp>(loc, structType);
      for (unsigned i = 0; i < structType.getBody().size(); ++i) {
        Value value = rewriter.create<LLVM::ExtractValueOp>(
            loc, packedType.getBody()[outputStructIndex], packedOutput,
            outputStructIndex);
        outputStructIndex++;
        unpacked = rewriter.create<LLVM::InsertValueOp>(loc, structType,
                                                        unpacked, value, i);
      }
      outputs.push_back(unpacked);
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }
};
} // namespace

void mlir::triton::intel::populateDotOpToLLVMPatterns(
    TritonIntelGPUToLLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, Target target, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, target, benefit);
  patterns.add<DotAsyncOpConversion>(typeConverter, target, benefit);
  patterns.add<DotWaitOpConversion>(typeConverter, target, benefit);
}