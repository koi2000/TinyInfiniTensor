#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj* graph, Tensor A, Tensor B, Tensor C, bool transA, bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA), transB(transB) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]") << ",A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
       << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec& inputs) {
    // =================================== 作业 ===================================
    // TODO：返回经过 matmul 操作后的 shape
    // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
    // =================================== 作业 ===================================
    const auto& shapeA = inputs[0]->getDims();
    const auto& shapeB = inputs[1]->getDims();

    // 确定矩阵 A 和 B 的形状
    size_t M = transA ? shapeA[shapeA.size() - 1] : shapeA[shapeA.size() - 2];
    size_t K_A = transA ? shapeA[shapeA.size() - 2] : shapeA[shapeA.size() - 1];
    size_t K_B = transB ? shapeB[shapeB.size() - 1] : shapeB[shapeB.size() - 2];
    size_t N = transB ? shapeB[shapeB.size() - 2] : shapeB[shapeB.size() - 1];

    // 确保两个矩阵可以相乘
    IT_ASSERT(K_A == K_B, "Incompatible dimensions for matrix multiplication.");

    // 计算输出形状
    Shape outputShape;
    for (size_t i = 0; i < shapeA.size() - 2; ++i) {
        outputShape.push_back(std::max(shapeA[i], shapeB[i]));  // 处理批次维度的广播
    }
    outputShape.push_back(M);
    outputShape.push_back(N);

    // 更新 m, n, k 的值
    m = M;
    n = N;
    k = K_A;

    return {{outputShape}};
}

}  // namespace infini