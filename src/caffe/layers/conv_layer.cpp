#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

    template<typename Dtype>
    void ConvolutionLayer<Dtype>::compute_output_shape() {
        const int *kernel_shape_data = this->kernel_shape_.cpu_data();
        const int *stride_data = this->stride_.cpu_data();
        const int *pad_data = this->pad_.cpu_data();
        const int *dilation_data = this->dilation_.cpu_data();
        this->output_shape_.clear();
        for (int i = 0; i < this->num_spatial_axes_; ++i) {
            // i + 1 to skip channel axis
            const int input_dim = this->input_shape(i + 1);
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
                                   / stride_data[i] + 1;
            this->output_shape_.push_back(output_dim);
        }
    }

    template<typename Dtype>
    void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top) {
        const Dtype *weight = this->blobs_[0]->cpu_data();
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype *bottom_data = bottom[i]->cpu_data();
            Dtype *top_data = top[i]->mutable_cpu_data();
            for (int n = 0; n < this->num_; ++n) {
                this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
                                       top_data + n * this->top_dim_);
                if (this->bias_term_) {
                    const Dtype *bias = this->blobs_[1]->cpu_data();
                    this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
                }
            }
        }
    }

    template<typename Dtype>
    void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                               const vector<bool> &propagate_down,
                                               const vector<Blob<Dtype> *> &bottom) {
        /* 反向传播的执行流程，首先需要计算本层偏置bias和权重weight的梯度，即bias_diff = top_diff x 1,
         * weight_diff = top_diff x input。 然后如果本层需要继续向下层传递梯度，则需要求出本层的输入（即下一层的输出）
         * input的梯度，即input_diff = top_diff * weight*/
        const Dtype *weight = this->blobs_[0]->cpu_data();
        Dtype *weight_diff = this->blobs_[0]->mutable_cpu_diff();
        for (int i = 0; i < top.size(); ++i) {/*挨个top blob进行反向传播*/
            const Dtype *top_diff = top[i]->cpu_diff();/*取出上一层计算得到diff值（与GT真值的偏差）*/
            const Dtype *bottom_data = bottom[i]->cpu_data();/*取出input数据*/
            Dtype *bottom_diff = bottom[i]->mutable_cpu_diff();/*指针，用于保存本层的diff结果*/
            // Bias gradient, if necessary.
            if (this->bias_term_ && this->param_propagate_down_[1]) {
                Dtype *bias_diff = this->blobs_[1]->mutable_cpu_diff();/*指针，保存本层的bias diff结果*/
                for (int n = 0; n < this->num_; ++n) {/*num_ = batch_size*/
                    /*反向传播计算计算bias的diff，并更新bias（即bias_diff+bias = new bias），注意这里包含了计算diff和更新两步*/
                    this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
                }
            }
            if (this->param_propagate_down_[0] || propagate_down[i]) {
                for (int n = 0; n < this->num_; ++n) {/* num_ = batch_size*/
                    // gradient w.r.t. weight. Note that we will accumulate diffs.
                    if (this->param_propagate_down_[0]) {
                        /*反向传播计算权重weight的导数diff*/
                        this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                                              top_diff + n * this->top_dim_,
                                              weight_diff);
                    }
                    // gradient w.r.t. bottom data, if necessary.
                    if (propagate_down[i]) {/*如果当前层需要继续向下一层反向传递梯度，则需要计算input的导数diff*/
                        this->backward_cpu_gemm(top_diff + n * this->top_dim_,
                                                weight,
                                                bottom_diff + n * this->bottom_dim_
                                                );
                    }
                }
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(ConvolutionLayer);
#endif

    INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
