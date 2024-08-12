#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj* graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec& inputs) {
    Shape dims = inputs[0]->getDims();
    size_t rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    // 确保所有输入张量在除拼接维度以外的其他维度上具有相同的形状
    for (size_t i = 1; i < inputs.size(); ++i) {
        const Shape& currentDims = inputs[i]->getDims();
        for (int j = 0; j < (int)rank; ++j) {
            if (j != dim && dims[j] != currentDims[j]) {
                // 如果在非拼接维度上，形状不同，返回空以表示形状推断失败
                return std::nullopt;
            }
        }
        // 在拼接维度上累加大小
        dims[dim] += currentDims[dim];
    }
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

}  // namespace infini
