#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top) {
        // Configure the kernel size, padding, stride, and inputs.
        ConvolutionParameter conv_param = this->layer_param_.convolution_param();
        force_nd_im2col_ = conv_param.force_nd_im2col();/*是否强制将图像按照卷积核大小进行列化(转换为caffe计算矩阵)*/
        channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());/*channel所在的axis*/
        const int first_spatial_axis = channel_axis_ + 1;/*图像矩阵的空间dim起始位置，即channel * row * colum中row所在的axis*/
        const int num_axes = bottom[0]->num_axes();/*传入数据的dim数量，注意dim与shape之间的不同*/
        num_spatial_axes_ = num_axes - first_spatial_axis;/*得到传入数据的shape中表述空间的dim有几个*/
        CHECK_GE(num_spatial_axes_, 0);/*检查传入数据的shape是否至少包含1个空间dim，即是否至少包含row*/
        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
        vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));/*存储传入图像的row colum值*/
        // Setup filter kernel dimensions (kernel_shape_).
        kernel_shape_.Reshape(spatial_dim_blob_shape);/*根据传入数据的row colum数据初始化卷积核的dim数据，从而区分是几维的卷积*/
        int *kernel_shape_data = kernel_shape_.mutable_cpu_data();/*获取卷积核shape的数据指针*/
        /*开始从param中读取参数，对卷积核的shape进行赋值*/
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                << "kernel_h & kernel_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.kernel_size_size())
                << "Either kernel_size or kernel_h/w should be specified; not both.";
            kernel_shape_data[0] = conv_param.kernel_h();
            kernel_shape_data[1] = conv_param.kernel_w();
        } else {
            const int num_kernel_dims = conv_param.kernel_size_size();
            CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
            << "kernel_size must be specified once, or once per spatial dimension "
            << "(kernel_size specified " << num_kernel_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";
            for (int i = 0; i < num_spatial_axes_; ++i) {
                kernel_shape_data[i] =
                        conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
            }
        }
        /*检查卷积核各个dim中的值是否大于0，即正常数据[batch_size,channel,row,colum]中任何一个参数都不应为0*/
        for (int i = 0; i < num_spatial_axes_; ++i) {
            CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
        }
        // Setup stride dimensions (stride_).
        stride_.Reshape(spatial_dim_blob_shape);/*记录步长的shape dim数量同样应与输入数据的空间dim相等*/
        int *stride_data = stride_.mutable_cpu_data();/*取出数据指针*/
        /*根据传入的参数对stride进行赋值*/
        if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                << "stride_h & stride_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.stride_size())
                << "Either stride or stride_h/w should be specified; not both.";
            stride_data[0] = conv_param.stride_h();
            stride_data[1] = conv_param.stride_w();
        } else {
            const int num_stride_dims = conv_param.stride_size();
            CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
                  num_stride_dims == num_spatial_axes_)
            << "stride must be specified once, or once per spatial dimension "
            << "(stride specified " << num_stride_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";
            const int kDefaultStride = 1;
            for (int i = 0; i < num_spatial_axes_; ++i) {
                stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
                                 conv_param.stride((num_stride_dims == 1) ? 0 : i);
                CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
            }
        }
        // Setup pad dimensions (pad_).
        pad_.Reshape(spatial_dim_blob_shape);/*逻辑同上，开始设置pad参数*/
        int *pad_data = pad_.mutable_cpu_data();
        if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
            CHECK_EQ(num_spatial_axes_, 2)
                << "pad_h & pad_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.pad_size())
                << "Either pad or pad_h/w should be specified; not both.";
            pad_data[0] = conv_param.pad_h();
            pad_data[1] = conv_param.pad_w();
        } else {
            const int num_pad_dims = conv_param.pad_size();
            CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
                  num_pad_dims == num_spatial_axes_)
            << "pad must be specified once, or once per spatial dimension "
            << "(pad specified " << num_pad_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";
            const int kDefaultPad = 0;
            for (int i = 0; i < num_spatial_axes_; ++i) {
                pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
                              conv_param.pad((num_pad_dims == 1) ? 0 : i);
            }
        }
        // Setup dilation dimensions (dilation_).
        dilation_.Reshape(spatial_dim_blob_shape);/*设置膨胀参数*/
        int *dilation_data = dilation_.mutable_cpu_data();
        const int num_dilation_dims = conv_param.dilation_size();
        CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
              num_dilation_dims == num_spatial_axes_)
        << "dilation must be specified once, or once per spatial dimension "
        << "(dilation specified " << num_dilation_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
        const int kDefaultDilation = 1;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                               conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
        }
        // Special case: im2col is the identity for 1x1 convolution with stride 1
        // and no padding, so flag for skipping the buffer and transformation.
        /*判断卷积核是不是1x1且stride==1，pad==0，如果是1x1的那么换算成caffe计算矩阵会简略很多*/
        is_1x1_ = true;
        for (int i = 0; i < num_spatial_axes_; ++i) {
            is_1x1_ &= kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;/*与操作判断卷积核的shape都为1且pad都为0*/
            if (!is_1x1_) { break; }
        }
        // Configure output channels and groups.
        channels_ = bottom[0]->shape(channel_axis_);/*获取bottom的channel数量*/
        num_output_ = this->layer_param_.convolution_param().num_output();/*输出的channel数量*/
        CHECK_GT(num_output_, 0);
        group_ = this->layer_param_.convolution_param().group();/*是否有分组卷积*/
        CHECK_EQ(channels_ % group_, 0);/*分组是均分的，所以必须要能够整除*/
        CHECK_EQ(num_output_ % group_, 0)
            << "Number of output should be multiples of group.";
        if (reverse_dimensions()) {/*是否要颠倒卷积的in_channel和out_channel，因为反卷积时的输出和输入参数与正向卷积刚好是相反的*/
            conv_out_channels_ = channels_;
            conv_in_channels_ = num_output_;
        } else {
            conv_out_channels_ = num_output_;
            conv_in_channels_ = channels_;
        }
        // Handle the parameters: weights and biases.
        // - blobs_[0] holds the filter weights
        // - blobs_[1] holds the biases (optional)
        vector<int> weight_shape(2);/*kenel权重的shape，假设in_ch=3,out_ch=64，kernel size=5则有weight shape：[64,3,5,5]*/
        weight_shape[0] = conv_out_channels_;
        weight_shape[1] = conv_in_channels_ / group_;

        for (int i = 0; i < num_spatial_axes_; ++i) {
            weight_shape.push_back(kernel_shape_data[i]);
        }
        bias_term_ = this->layer_param_.convolution_param().bias_term();/*标识是否使用bias*/
        vector<int> bias_shape(bias_term_, num_output_);/*声明一个bias_shape,注意这里bias_term_是自动将bool转为了0/1*/
        if (this->blobs_.size() > 0) {/*在conv层中有一个blobs数组，默认blobs[0]用于保存kenel的权重weight，blobs[1]保存偏置bias*/
            CHECK_EQ(1 + bias_term_, this->blobs_.size())
                << "Incorrect number of weight blobs.";
            if (weight_shape != this->blobs_[0]->shape()) {/*判断weight的shape形状是否和上文中设置的一致*/
                Blob<Dtype> weight_shaped_blob(weight_shape);
                LOG(FATAL) << "Incorrect weight shape: expected shape "
                           << weight_shaped_blob.shape_string() << "; instead, shape was "
                           << this->blobs_[0]->shape_string();
            }
            if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
                Blob<Dtype> bias_shaped_blob(bias_shape);
                LOG(FATAL) << "Incorrect bias shape: expected shape "
                           << bias_shaped_blob.shape_string() << "; instead, shape was "
                           << this->blobs_[1]->shape_string();
            }
            LOG(INFO) << "Skipping parameter initialization";
        } else {
            if (bias_term_) {
                this->blobs_.resize(2);
            } else {
                this->blobs_.resize(1);
            }
            // Initialize and fill the weights:
            // output channels x input channels per-group x kernel height x kernel width
            /*上文都是关于超参数的判断，这里开始初始化*/
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            /*定义一个初始化滤波器，根据参数的不同（比如bilinear）对weight值进行初始化*/
            shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
                    this->layer_param_.convolution_param().weight_filler()));
            weight_filler->Fill(this->blobs_[0].get());/*根据blob[0]初始化的值进行填充*/
            // If necessary, initialize and fill the biases.
            if (bias_term_) {
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
                shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                        this->layer_param_.convolution_param().bias_filler()));
                bias_filler->Fill(this->blobs_[1].get());
            }
        }
        kernel_dim_ = this->blobs_[0]->count(1);/*计算一个完整的kernel的数据量，比如weight shape：[64,3,5,5]，则一个完整的kernel 数据量为3x5x5*/
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;/*分组权重的指针偏移量，如果需要分组卷积，则需要得到每一组的权重起始位置*/
        // Propagate gradients to the parameters (as directed by backward pass).
        this->param_propagate_down_.resize(this->blobs_.size(), true);/*这个参数决定了各个blob是否需要计算反向传播*/
    }

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::Reshape(
            const vector<Blob<Dtype> *> &bottom,
            const vector<Blob<Dtype> *> &top) {
        const int first_spatial_axis = channel_axis_ + 1;
        CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)/*检查bottom的输入dim*/
                    << "bottom num_axes may not change.";
        num_ = bottom[0]->count(0, channel_axis_);
        CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)/*检查bottom的channel*/
                    << "Input size incompatible with convolution kernel.";
        // TODO: generalize to handle inputs of different shapes.
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {/*比对每个bottom，确保shape相等*/
            CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
            << "All inputs must have the same shape.";
        }
        // Shape the tops.
        /*根据参数设置top的shape*/
        bottom_shape_ = &bottom[0]->shape();
        compute_output_shape();/*计算输出的shape，虚函数，由各个子函数重写*/
        /*这里利用指针获取了shape中CxHxW之前的所有数据，即BatchSize*/
        vector<int> top_shape(bottom[0]->shape().begin(),
                              bottom[0]->shape().begin() + channel_axis_);
        top_shape.push_back(num_output_);/*添加输出channel*/
        for (int i = 0; i < num_spatial_axes_; ++i) {
            top_shape.push_back(output_shape_[i]);/*添加输出的空间形状，output_shape_会在各个子类中计算得到*/
        }
        for (int top_id = 0; top_id < top.size(); ++top_id) {
            top[top_id]->Reshape(top_shape);/*将top进行reshape*/
        }
        /*如果是反卷积，则输出的dim总数应从bottom中取*/
        if (reverse_dimensions()) {
            conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
        } else {
            conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
        }
        /* 转变为caffe计算矩阵后，col_offset_即每个完整的卷积核按照参数卷积整个输入图像所需的计算量，
         * 在img2col操作后，每一个卷积核都展开reshape为一行以进行矩阵运算*/
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;/*分组卷积的情况下，得到不同分组的输出数据偏移*/
        // Setup input dimensions (conv_input_shape_).
        /*设置卷积输出的shape大小*/
        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);/*num_spatial_axes_+1是在空间维度信息的基础上添加了channel*/
        conv_input_shape_.Reshape(bottom_dim_blob_shape);
        int *conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        /*设置具体的conv_input_shape值*/
        for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
            if (reverse_dimensions()) {/*反卷积的情况下要top才是输入*/
                conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
            } else {
                conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
            }
        }
        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1x1 convolution
        // it goes lazily unused to save memory.
        col_buffer_shape_.clear();/*用于保存col化的输入图像的shape，注意是shape*/
        col_buffer_shape_.push_back(kernel_dim_ * group_);
        for (int i = 0; i < num_spatial_axes_; ++i) {/*最终得到[kernel_dim*group,row,colum]*/
            if (reverse_dimensions()) {
                col_buffer_shape_.push_back(input_shape(i + 1));
            } else {
                col_buffer_shape_.push_back(output_shape_[i]);
            }
        }
        col_buffer_.Reshape(col_buffer_shape_);
        bottom_dim_ = bottom[0]->count(channel_axis_);/*获得输入的dim总数（CHW）*/
        top_dim_ = top[0]->count(channel_axis_);/*获得输出的dim总数（CHW）*/
        num_kernels_im2col_ = conv_in_channels_ *
                              conv_out_spatial_dim_;/*输入的channel * 输出的空间维度部分的数据总量（即row*colum），im2col_nd_gpu函数才会用到*/
        num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;/*同上im2col_nd_gpu函数才会用到*/
        // Set up the all ones "bias multiplier" for adding biases by BLAS
        out_spatial_dim_ = top[0]->count(first_spatial_axis);/*输出的空前维度部分的数据总量*/
        if (bias_term_) {/*如果使用bias则设置bias*/
            vector<int> bias_multiplier_shape(1, out_spatial_dim_);/*bias的一个倍数矩阵，应该是bias的一个权重*/
            bias_multiplier_.Reshape(bias_multiplier_shape);
            caffe_set(bias_multiplier_.count(), Dtype(1),
                      bias_multiplier_.mutable_cpu_data());
        }
    }

/*---正向运算------------------------------------------------------------------------------------------------------------*/

    template<typename Dtype>
    /*alpha*weight*input + beta*bias的前半部分（alpha*weight*input）*/
    void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(
            const Dtype *input,/*输入的数据的内存指针*/
            const Dtype *weights,/*权重，即卷积核*/
            Dtype *output,/*储存结果的内存指针*/
            bool skip_im2col/*是否跳过列化操作*/
    ) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {/*是否是单位卷积，即卷积核为1x1且stride=1，pad=0*/
            if (!skip_im2col) {/*是否要跳过列化*/
                /*将输入数据列化后存入col_buffer_对象中*/
                conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
            }
            col_buff = col_buffer_.cpu_data();
        }
        for (int g = 0; g < group_; ++g) {/*分组进行卷积*/
            /* cblas函数，output = alpha * aX + beta * output，当beta恒为0时可以用于矩阵乘法
             * 这里是用weight * input
             * 注意这个函数只计算的权重，没有计算偏移bias */
            caffe_cpu_gemm<Dtype>(CblasNoTrans,/*不需要旋转变换*/
                                  CblasNoTrans,/*不需要旋转变换*/
                                  conv_out_channels_ / group_,/*如果有多组卷积，则每次矩阵乘法所需的weight行（row）数会按倍减少*/
                                  conv_out_spatial_dim_,/*col_buff的列(colum)数，也是output的colum数*/
                                  kernel_dim_,/*weight矩阵的colum数，即矩阵运算过程中被吃掉的维度数*/
                                  (Dtype) 1.,/*alpha*/
                                  weights + weight_offset_ * g,/*weight，offset是分组卷积时指针指针的偏移*/
                                  col_buff + col_offset_ * g,/*input*/
                                  (Dtype) 0.,/*beta*/
                                  output + output_offset_ * g);/*output*/
        }
    }

    template<typename Dtype>
    /*alpha*weight*input + beta*bias的后半部分（+ beta*bias）*/
    void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype *output,
                                                       const Dtype *bias) {
        /* 利用[c = alpha * a * b + beta * c]函数完成矩阵加
         * output = 1*bias*1 + 1*output= bias + output
         * 这里的output是经过weight运算的，事实上就是 output=weight*input+bias 中的 +bias 操作*/
        caffe_cpu_gemm<Dtype>(CblasNoTrans,
                              CblasNoTrans,
                              num_output_,
                              out_spatial_dim_,
                              1,
                              (Dtype) 1., /*alpha*/
                              bias, /**/
                              bias_multiplier_.cpu_data(),/*在上文中该对象恒为1*/
                              (Dtype) 1.,
                              output);
    }

/*---反向运算------------------------------------------------------------------------------------------------------------*/
    template<typename Dtype>
    /*如果本层的梯度需要继续向bottom层传递,则还要计算input的梯度,因为本层的input是bottom层的output*/
    void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype *output,
                                                        const Dtype *weights, Dtype *input) {
        Dtype *col_buff = col_buffer_.mutable_cpu_data();
        if (is_1x1_) {
            col_buff = input;
        }
        for (int g = 0; g < group_; ++g) {
            /*c = alpha*a*b + beta*c函数的运用,按照公式input_diff = output_diff * weights,
             * 即alpha = 1, a = weights, b = output_diff, beta = 0即input_diff不参与运算, c=input_diff是指向了储存位置的指针*/
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                  kernel_dim_,
                                  conv_out_spatial_dim_,
                                  conv_out_channels_ / group_,
                                  (Dtype) 1.,
                                  weights + weight_offset_ * g,
                                  output + output_offset_ * g,
                                  (Dtype) 0.,
                                  col_buff + col_offset_ * g);
        }
        if (!is_1x1_) {
            conv_col2im_cpu(col_buff, input);
        }
    }

    template<typename Dtype>
    /*权重weight的梯度运算，wx+bias中的wx部分的梯度*/
    void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype *input, /*bottom input*/
                                                      const Dtype *output, /*top diff*/
                                                      Dtype *weights /*保存本层的weight diff指针*/
    ) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {
            conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
            col_buff = col_buffer_.cpu_data();
        }
        for (int g = 0; g < group_; ++g) {
            /* weight_diff = 1 x top_diff x input + 1 x weight_diff
             * 这里beta = 1 是因为外部循环会根据batch size调用多次，所以diff是整个batch相加后得到的*/
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                                  conv_out_channels_ / group_,
                                  kernel_dim_,
                                  conv_out_spatial_dim_,
                                  (Dtype) 1.,/*alpha*/
                                  output + output_offset_ * g,/*a*/
                                  col_buff + col_offset_ * g,/*b*/
                                  (Dtype) 1.,/*beta*/
                                  weights + weight_offset_ * g/*c*/
                                  );
        }
    }

    template<typename Dtype>
    /*偏置bias的梯度运算，wx+bias中的+bias部分的梯度*/
    void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype *bias,/*储存本层bias diff的指针*/
                                                        const Dtype *input/*top diff*/
    ) {
        /* 利用[c = alpha * a * b + beta * c]函数完成矩阵乘
         * 在卷积公式wx+bias中，bias的梯度（导数）恒定为1，
         * 根据diff*导函数 = 本层diff，这里实际上的运算为 1 * top_diff * 1(bias导数) + 1*bias，
         * 即top_diff*bias导数 + bias = 本层bias_diff + bias，从而更新了bias*/
        caffe_cpu_gemv<Dtype>(CblasNoTrans,
                              num_output_,
                              out_spatial_dim_,
                              1.,/*alpha*/
                              input,/*a*/
                              bias_multiplier_.cpu_data(),/*b*/
                              1.,/*beta*/
                              bias/*c*/
        );
    }

#ifndef CPU_ONLY

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype *input,
                                                       const Dtype *weights, Dtype *output, bool skip_im2col) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {
            if (!skip_im2col) {
                conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
            }
            col_buff = col_buffer_.gpu_data();
        }
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
                                                              group_, conv_out_spatial_dim_, kernel_dim_,
                                  (Dtype) 1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype) 0., output + output_offset_ * g);
        }
    }

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype *output,
                                                       const Dtype *bias) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                              out_spatial_dim_, 1, (Dtype) 1., bias, bias_multiplier_.gpu_data(),
                              (Dtype) 1., output);
    }

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype *output,
                                                        const Dtype *weights, Dtype *input) {
        Dtype *col_buff = col_buffer_.mutable_gpu_data();
        if (is_1x1_) {
            col_buff = input;
        }
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
                                  conv_out_spatial_dim_, conv_out_channels_ / group_,
                                  (Dtype) 1., weights + weight_offset_ * g, output + output_offset_ * g,
                                  (Dtype) 0., col_buff + col_offset_ * g);
        }
        if (!is_1x1_) {
            conv_col2im_gpu(col_buff, input);
        }
    }

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype *input,
                                                      const Dtype *output, Dtype *weights) {
        const Dtype *col_buff = input;
        if (!is_1x1_) {
            conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
            col_buff = col_buffer_.gpu_data();
        }
        for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
                                  kernel_dim_, conv_out_spatial_dim_,
                                  (Dtype) 1., output + output_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype) 1., weights + weight_offset_ * g);
        }
    }

    template<typename Dtype>
    void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype *bias,
                                                        const Dtype *input) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
                              input, bias_multiplier_.gpu_data(), 1., bias);
    }

#endif  // !CPU_ONLY

    INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
