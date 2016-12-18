#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ParamConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[1] is the filter weight input
  const Dtype* weight = this->bottom[1]->gpu_data();
  // bottom[0] is the input data
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_gpu_gemm(bottom_data + bottom[0]->offset(n), weight,
        top_data + top[0]->offset(n));
    if (this->bias_term_) {
      // bottom[2] is the filter bias input
      const Dtype* bias = this->bottom[2]->gpu_data();
      this->forward_gpu_bias(top_data + top[0]->offset(n), bias);
    }
  }
}

template <typename Dtype>
void ParamConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // bottom[1] is the filter weight input
  const Dtype* weight = this->bottom[1]->gpu_data();
  Dtype* weight_diff = this->bottom[1]->mutable_gpu_diff();
  // bottom[0] is the input data
  const Dtype* top_diff = top[0]->gpu_diff();
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    // bottom[2] is the filter bias input
    Dtype* bias_diff = this->bottom[2]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_gpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        this->weight_gpu_gemm(bottom_data + bottom[0]->offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        this->backward_gpu_gemm(top_diff + top[0]->offset(n), weight,
            bottom_diff + bottom[0]->offset(n));
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ParamConvolutionLayer);

}  // namespace caffe
