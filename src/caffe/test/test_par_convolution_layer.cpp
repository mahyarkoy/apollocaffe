#include <cstring>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv_separate_channel(const Blob<Dtype>* in, ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_size()) {
    kernel_h = kernel_w = conv_param->kernel_size();
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param->has_pad_h()) {
    pad_h = pad_w = conv_param->pad();
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param->has_stride_h()) {
    stride_h = stride_w = conv_param->stride();
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                    out_data[out->offset(n, o + o_head, y, x)] +=
                        in_data[in->offset(n, k + k_head, in_y, in_x)]
                        * weight_data[weights[0]->offset(n, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term() && weights.size() > 1) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
            out_data[out->offset(n, o, y, x)] += bias_data[n];
          }
        }
      }
    }
  }
}

template void caffe_conv_separate_channel(const Blob<float>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv_separate_channel(const Blob<double>* in,
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class ParamConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ParamConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 4, 3, 3)),
        blob_bottom_1_(new Blob<Dtype>(2, 4, 6, 6)),
        blob_bottom_2_(new Blob<Dtype>(2, 4, 3, 3)),
        blob_top_(new Blob<Dtype>()) {}
  
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    //filler.Fill(this->blob_bottom_);
    //filler.Fill(this->blob_bottom_1_);
    //filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~ParamConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ParamConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(ParamConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  shared_ptr<Layer<Dtype> > layer(
      new ParamConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(ParamConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ParamConvolutionLayer<Dtype>(layer_param));

  // Make 1*4*1*1 filter and 1*4*3*3input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(1,4,1,1));
  weights[1].reset(new Blob<Dtype>(1,1,1,1));
  Dtype* input_data = this->blob_bottom_->mutable_cpu_data();
  Dtype* weight_data = weights[0]->mutable_cpu_data();
  Dtype* bias_data = weights[1]->mutable_cpu_data();
  weight_data[0] = 0;
  weight_data[1] = 1;
  weight_data[2] = 2;
  weight_data[3] = 3;
  bias_data[0] = 100;
  for(int i=0; i<36; i++)
    input_data[i] = i;

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_separate_channel(this->blob_bottom_, convolution_param, weights,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  std::cout << "====CONV====" << std::endl;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    std::cout << top_data[i] << "__" << ref_top_data[i] << std::endl;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ParamConvolutionLayerTest, TestRandomConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ParamConvolutionLayer<Dtype>(layer_param));

  // Make 2*4*3*3 filter for 2*4*6*6input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(2,4,3,3));
  weights[1].reset(new Blob<Dtype>(2,1,1,1));

  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_1_);
  filler.Fill(weights[0].get());
  filler.Fill(weights[1].get());

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_1_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_separate_channel(this->blob_bottom_1_, convolution_param, weights,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ParamConvolutionLayerTest, Test2ChannelConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<Dtype> > layer(
      new ParamConvolutionLayer<Dtype>(layer_param));

  // Make 2*4*1*1 filter and 2*4*3*3input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(2,4,1,1));
  weights[1].reset(new Blob<Dtype>(2,1,1,1));
  Dtype* input_data = this->blob_bottom_2_->mutable_cpu_data();
  Dtype* weight_data = weights[0]->mutable_cpu_data();
  Dtype* bias_data = weights[1]->mutable_cpu_data();
  weight_data[0] = 0;
  weight_data[1] = 1;
  weight_data[2] = 2;
  weight_data[3] = 3;
  weight_data[4] = 4;
  weight_data[5] = 5;
  weight_data[6] = 6;
  weight_data[7] = 7;
  bias_data[0] = 100;
  bias_data[1] = 200;
  for(int i=0; i<72; i++)
    input_data[i] = i;

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;

  caffe_conv_separate_channel(this->blob_bottom_2_, convolution_param, weights,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  std::cout << "====CONV====" << std::endl;
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    std::cout << top_data[i] << "__" << ref_top_data[i] << std::endl;
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(ParamConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  
  // Make 1*4*1*1 filter and 1*4*3*3input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(1,4,1,1));
  weights[1].reset(new Blob<Dtype>(1,1,1,1));
  Dtype* input_data = this->blob_bottom_->mutable_cpu_data();
  Dtype* weight_data = weights[0]->mutable_cpu_data();
  Dtype* bias_data = weights[1]->mutable_cpu_data();
  weight_data[0] = 0;
  weight_data[1] = 1;
  weight_data[2] = 2;
  weight_data[3] = 3;
  bias_data[0] = 100;
  for(int i=0; i<36; i++)
    input_data[i] = i;

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  ParamConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
/*
TYPED_TEST(ParamConvolutionLayerTest, Test2ChannelGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  
  // Make 2*4*1*1 filter and 2*4*3*3input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(2,4,1,1));
  weights[1].reset(new Blob<Dtype>(2,1,1,1));
  Dtype* input_data = this->blob_bottom_2_->mutable_cpu_data();
  Dtype* weight_data = weights[0]->mutable_cpu_data();
  Dtype* bias_data = weights[1]->mutable_cpu_data();
  weight_data[0] = 0;
  weight_data[1] = 1;
  weight_data[2] = 2;
  weight_data[3] = 3;
  weight_data[4] = 4;
  weight_data[5] = 5;
  weight_data[6] = 6;
  weight_data[7] = 7;
  bias_data[0] = 100;
  bias_data[1] = 200;
  for(int i=0; i<72; i++)
    input_data[i] = i;

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  ParamConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/

TYPED_TEST(ParamConvolutionLayerTest, TestRandomGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  
  // Make 5*4*3*3 filter for 2*4*6*6input
  vector<shared_ptr<Blob<Dtype> > > weights (2);
  weights[0].reset(new Blob<Dtype>(2,4,3,3));
  weights[1].reset(new Blob<Dtype>(2,1,1,1));

  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_1_);
  filler.Fill(weights[0].get());
  filler.Fill(weights[1].get());

  // Load inputs to vectors
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(this->blob_bottom_1_);
  this->blob_bottom_vec_.push_back(weights[0].get());
  this->blob_bottom_vec_.push_back(weights[1].get());

  this->blob_top_vec_.clear();
  shared_ptr< Blob<Dtype> > top_data (new Blob<Dtype>());
  this->blob_top_vec_.push_back(top_data.get());

  ParamConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
