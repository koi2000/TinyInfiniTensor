#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape& A, const Shape& B) {
    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    Shape result;

    // 逆序遍历两个形状，以确保从最后一个维度开始进行广播
    auto itA = A.rbegin();
    auto itB = B.rbegin();

    while (itA != A.rend() || itB != B.rend()) {
        size_t dimA = (itA != A.rend()) ? *itA : 1;
        size_t dimB = (itB != B.rend()) ? *itB : 1;

        // 检查广播规则
        if (dimA == dimB || dimA == 1 || dimB == 1) {
            result.push_back(std::max(dimA, dimB));
        } else {
            // 如果维度不兼容，返回空的 Shape，表示广播失败
            return {};
        }

        if (itA != A.rend())
            ++itA;
        if (itB != B.rend())
            ++itB;
    }

    // 逆序返回结果，因为我们从最后一个维度开始构造的
    std::reverse(result.begin(), result.end());

    return result;
}

int get_real_axis(const int& axis, const int& rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape& shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape& shapeIndex, const Shape& shape, const Shape& stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
        case Device::CPU: return "CPU";
        default: IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs& kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

}  // namespace infini
