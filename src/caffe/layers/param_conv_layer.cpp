#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ParamConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ParamConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[1] is the filter weight input
  const Dtype* weight = bottom[1]->cpu_data();

  // bottom[0] is the data input
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_cpu_gemm(bottom_data + bottom[0]->offset(n), weight,
        top_data + top[0]->offset(n));
    if (this->bias_term_ && bottom.size() > 2) {
      // bottom[2] is the filter bias input
      const Dtype* bias = bottom[2]->cpu_data();
      this->forward_cpu_bias(top_data + top[0]->offset(n), bias);
    }
  }
}

template <typename Dtype>
void ParamConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  // bottom[1] is the filter weight input
  const Dtype* weight = bottom[1]->cpu_data();
  Dtype* weight_diff = bottom[1]->mutable_cpu_diff();

  // bottom[0] is the data input
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1] && bottom.size() > 2) {
    Dtype* bias_diff = bottom[2]->mutable_cpu_diff();
    for (int n = 0; n < this->num_; ++n) {
      // accumulate updates for nums, this is to restart at initial n
      if (n > 0)
        _acc_weight_update = true;
      this->backward_cpu_bias(bias_diff, top_diff + top[0]->offset(n));
    }
  }
  if (this->param_propagate_down_[0] || propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      if (this->param_propagate_down_[0]) {
        // accumulate updates for nums, this is to restart at initial n
        if (n > 0)
          _acc_weight_update = true;

        this->weight_cpu_gemm(bottom_data + bottom[0]->offset(n),
            top_diff + top[0]->offset(n), weight_diff);
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        this->backward_cpu_gemm(top_diff + top[0]->offset(n), weight,
            bottom_diff + bottom[0]->offset(n));
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ParamConvolutionLayer);
#endif

INSTANTIATE_CLASS(ParamConvolutionLayer);
REGISTER_LAYER_CLASS(ParamConvolution);

}  // namespace caffe
