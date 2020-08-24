#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif  // USE_OPENCV

#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {

    template<typename Dtype>
    AnnotatedMatrixDataLayer<Dtype>::AnnotatedMatrixDataLayer(const LayerParameter &param)
            : BasePrefetchingDataLayer<Dtype>(param),
              reader_(param) {
    }

    template<typename Dtype>
    AnnotatedMatrixDataLayer<Dtype>::~AnnotatedMatrixDataLayer() {
        this->StopInternalThread();
    }

    template<typename Dtype>
    void AnnotatedMatrixDataLayer<Dtype>::DataLayerSetUp(
            const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
        const int batch_size = this->layer_param_.data_param().batch_size(); //layer_param_ 在父类中定义的类成员，储存了算子的所有参数数据
        const AnnotatedDataParameter &anno_data_param = //从 layer_param_ 中读取lmdb文件中的配置
                this->layer_param_.annotated_data_param();
        for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) { //提取配置文件中batch_sampler部分的配置，即prior box配置
            batch_samplers_.push_back(anno_data_param.batch_sampler(i));
        }
        label_map_file_ = anno_data_param.label_map_file();
        // Make sure dimension is consistent within batch.
        const TransformationParameter &transform_param = //从 layer_param_ 中读取图像处理相关参数
                this->layer_param_.transform_param();
        if (transform_param.has_resize_param()) {
            if (transform_param.resize_param().resize_mode() ==
                ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                CHECK_EQ(batch_size, 1)
                    << "Only support batch size of 1 for FIT_SMALL_SIZE.";
            }
        }

        // 读取annotation datum数据中的第一个数据
        AnnotatedDatum &anno_datum = *(reader_.full().peek());

        // Use data_transformer to infer the expected blob shape from anno_datum.
        vector<int> top_shape = //data_transformer是在父类中定义的类成员，保存了算子所有需要调用的图像预处理函数
                this->data_transformer_->InferBlobShape(anno_datum.datum());//通过InferBlobShape函数获取算子的输出大小
        this->transformed_data_.Reshape(top_shape); //transformed_data_主要用于存放图像增强后的图片，是在父类中定义的类成员，Reshape则是进行内存分配
        // Reshape top[0] and prefetch_data according to the batch_size.
        top_shape[0] = batch_size;
        top[0]->Reshape(top_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) { // 一共有 PREFETCH_COUNT 个数据读取线程
            this->prefetch_[i].data_.Reshape(top_shape); //将预读线程中的数据全部reshape
        }
        LOG(INFO) << "output data size: " << top[0]->num() << ","
                  << top[0]->channels() << "," << top[0]->height() << ","
                  << top[0]->width();
        // label
        if (this->output_labels_) { //数据层是否输出label
            has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type(); //是否包含特定的label的数据类型
            vector<int> label_shape(4, 1);
            if (has_anno_type_) {
                anno_type_ = anno_datum.type(); //detect任务中label的数据类型为box
                if (anno_data_param.has_anno_type()) {
                    // If anno_type is provided in AnnotatedDataParameter, replace
                    // the type stored in each individual AnnotatedDatum.
                    LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
                    anno_type_ = anno_data_param.anno_type();
                }
                // Infer the label shape from anno_datum.AnnotationGroup().
                int num_bboxes = 0;
                if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                    // we store the bbox information in a specific format. In specific:
                    // All bboxes are stored in one spatial plane (num and channels are 1)
                    // And each row contains one and only one box in the following format:
                    // BOX的保存格式为：
                    // [item_id(当前batch的图idx), group_label(类别的idx), instance_id(某一类下box的idx), xmin, ymin, xmax, ymax, diff(是否难分样本)]
                    for (int g = 0; g < anno_datum.annotation_group_size(); ++g) { //逐分类统计box的个数
                        int annotation_size = anno_datum.annotation_group(g).annotation_size(); //这里得到该分类下box的格数
                        num_bboxes += annotation_size;
                    }
                    label_shape[0] = 1;
                    label_shape[1] = 1;
                    // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
                    // cpu_data and gpu_data for consistent prefetch thread. Thus we make
                    // sure there is at least one bbox. 确保输出的label至少有一个，否则loss计算会报错
                    label_shape[2] = std::max(num_bboxes, 1);
                    label_shape[3] = 8;
                } else {
                    LOG(FATAL) << "Unknown annotation type.";
                }
            } else {
                label_shape[0] = batch_size;
            }
            top[1]->Reshape(label_shape); //将label的输出reshape
            for (int i = 0; i < this->PREFETCH_COUNT; ++i) { //将每个预取线程的label都reshape
                this->prefetch_[i].label_.Reshape(label_shape);
            }
        }
    }

// This function is called on prefetch thread
    template<typename Dtype>
    void AnnotatedMatrixDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(batch->data_.count());
        CHECK(this->transformed_data_.count());

        // Reshape according to the first anno_datum of each batch
        // on single input batches allows for inputs of varying dimension.
        const int batch_size = this->layer_param_.data_param().batch_size(); //从layer_param_中调出batch_size参数
        const AnnotatedDataParameter &anno_data_param = //从layer_param_中调出 anno_data_param 参数
                this->layer_param_.annotated_data_param();
        const TransformationParameter &transform_param = //从layer_param_中调出 transform_param 参数
                this->layer_param_.transform_param();
        AnnotatedDatum &anno_datum = *(reader_.full().peek()); //读取datum中的第一个数据，用于计算top shape
        // Use data_transformer to infer the expected blob shape from anno_datum.
        vector<int> top_shape = //得到算子的输出shape
                this->data_transformer_->InferBlobShape(anno_datum.datum());
        this->transformed_data_.Reshape(top_shape); //将transformed_data_的内存重新分配
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape); //根据shape分配batch的内存空间，一个batch包含batch size个图像样本

        Dtype *top_data = batch->data_.mutable_cpu_data(); // 获取batch存放数据的指针
        Dtype *top_label = NULL;  // suppress warnings about uninitialized variables
        if (this->output_labels_ && !has_anno_type_) {
            top_label = batch->label_.mutable_cpu_data();
        }

        // Store transformed annotation.
        map<int, vector<AnnotationGroup> > all_anno;
        int num_bboxes = 0;

        for (int item_id = 0; item_id < batch_size; ++item_id) {
            timer.Start();
            // get a anno_datum
            AnnotatedDatum &anno_datum = *(reader_.full().pop("Waiting for data")); //读取一个batch size大小的anno datum数据
            read_time += timer.MicroSeconds();
            timer.Start();
            AnnotatedDatum distort_datum; //新建一个AnnotatedDatum类存储distort后的图像
            AnnotatedDatum *expand_datum = NULL; //新建一个AnnotatedDatum类存储expand后的图像
            if (transform_param.has_distort_param()) { //是否进行distort
                distort_datum.CopyFrom(anno_datum);
                this->data_transformer_->DistortImage(anno_datum.datum(), //调用data_transformer_成员类中的对应方法进行图像处理
                                                      distort_datum.mutable_datum());
                if (transform_param.has_expand_param()) { //distort后是否进行expand
                    expand_datum = new AnnotatedDatum();
                    this->data_transformer_->ExpandImage(distort_datum, expand_datum); //调用data_transformer_成员类中的对应方法进行图像处理
                } else {
                    expand_datum = &distort_datum;
                }
            } else {
                if (transform_param.has_expand_param()) {
                    expand_datum = new AnnotatedDatum();
                    this->data_transformer_->ExpandImage(anno_datum, expand_datum);
                } else { //在不做任何图像增强的情况下，则将anno_datum直接赋给expand_datum
                    expand_datum = &anno_datum;
                }
            }
            AnnotatedDatum *sampled_datum = NULL; // 新建一个AnnotatedDatum类
            bool has_sampled = false;
            if (batch_samplers_.size() > 0) {
                /*对label boxes进行采样增强
                 *
                 * 每个采样器只生成max_sample个match box，坐标表示也是普通的(lx,ly,w,h)
                 * 且这个match box只需要与任意一个target box匹配即可。
                 *
                 * 对于batchsize中的每一幅图像，为每个采样器(batch_sampler)生成max_sample个match box
                 * 每个采样器生成的boundingbox与目标的IOU=0.1,0.3,0.5,0.7,0.9，这个与论文的描述是一致的
                 * 示例：
                    batch_sampler
                    {
                      sampler
                      {
                        min_scale: 0.3
                        max_scale: 1.0
                        min_aspect_ratio: 0.5
                        max_aspect_ratio: 2.0
                      }
                      sample_constraint
                      {
                        min_jaccard_overlap: 0.7
                      }
                      max_sample: 1
                      max_trials: 50
                    }
                 *  对于该采样器，随机生成的满足条件的matchbox与图像中任一目标的IOU>0.7
                 *  注意：
                 *    1. 生成的boundingbox坐标是归一化的坐标，这样不受resize的影响，目标检测的回归都是采用的这种形式(比如MTCNN)
                 *    2. 随机生成boundingbox的时候，根据每个batch_sampler的参数：尺度，宽高比，每个采样器最多尝试max_trials次
                 *
                 */
                vector<NormalizedBBox> sampled_bboxes;
                GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);// match box生成
                /* 从生成的所有match box中随机挑选一个match box
                 * 裁剪出该match box对应的图像(大小就是sampled_bboxes[rand_idx]在原图中的大小)并重新计算该match box中所有目标的坐标
                 * 注意：
                 *    1. bounding box中目标的坐标=(原图中ground truth的坐标-该bounding box的坐标)/(bounding box的边长)
                 *     这里groundtruth与boundingbox的坐标都相对于原图,在mtcnn中也是采用了该计算方式
                 */
                if (sampled_bboxes.size() > 0) {
                    // Randomly pick a sampled bbox and crop the expand_datum.
                    int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
                    sampled_datum = new AnnotatedDatum();
                    this->data_transformer_->CropImage(*expand_datum,
                                                       sampled_bboxes[rand_idx],
                                                       sampled_datum);
                    has_sampled = true;
                } else {
                    sampled_datum = expand_datum; //如果Crop后的图中不包含任何一个box，则直接按原图
                }
            } else {
                sampled_datum = expand_datum;
            }
            CHECK(sampled_datum != NULL); //检查sampled datum是否不为空
            timer.Start();
            vector<int> shape =
                    this->data_transformer_->InferBlobShape(sampled_datum->datum());
            if (transform_param.has_resize_param()) { //是否resize
                if (transform_param.resize_param().resize_mode() ==
                    ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                    this->transformed_data_.Reshape(shape);
                    batch->data_.Reshape(shape);
                    top_data = batch->data_.mutable_cpu_data();
                } else {
                    CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                                     shape.begin() + 1));
                }
            }
//            else {
//                std::cout<<"CHECK"<<std::endl;
//                CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
//                                 shape.begin() + 1));
//            }

            // Apply data transformations (mirror, scale, crop...) 应用图像处理操作
            int offset = batch->data_.offset(item_id);
            this->transformed_data_.set_cpu_data(top_data + offset);
            vector<AnnotationGroup> transformed_anno_vec;
            if (this->output_labels_) { //如果要输出label
                if (has_anno_type_) {
                    // Make sure all data have same annotation type.
                    CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
                    if (anno_data_param.has_anno_type()) {
                        sampled_datum->set_type(anno_type_);
                    } else {
                        CHECK_EQ(anno_type_, sampled_datum->type()) <<
                                                                    "Different AnnotationType.";
                    }
                    // Transform datum and annotation_group at the same time //如果要输出label，则对datum和label（annotation_group）同步处理
                    transformed_anno_vec.clear();
                     /* 将crop出来的AnnotatedDatum转换为数据部分和标注部分
                     *  数据部分会resize到数据层设置的大小(比如300x300)并保存到top[0]中
                     *  标注是所有目标在图像中的坐标
                     *
                     * 注意：
                     *  1. 这里的图像并不一定是原始crop的图像，如果transform_param有crop_size这个参数，原来crop出来的图像会再次crop的
                     *  2. 由于这里对crop出来的图像进行了一次resize,所以如果生成lmdb的时候，进行resize会导致数据层对原图进行两次resize，
                     *     这样有可能会影响到目标的宽高比，所以在SFD(Single Shot Scale-invariant Face Detector)中，对此处做了一点改进，即在第一步
                     *     生成boundingbox的时候，保证每个boundingbox都是正方形，这样resize到300x300的时候就不会改变目标的宽高比
                     */
                    this->data_transformer_->Transform(*sampled_datum,//将datum数据执行处理后放入内存
                                                       &(this->transformed_data_),
                                                       &transformed_anno_vec);
                    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                        // Count the number of bboxes.
                        for (int g = 0; g < transformed_anno_vec.size(); ++g) {
                            num_bboxes += transformed_anno_vec[g].annotation_size();
                        }
                    } else {
                        LOG(FATAL) << "Unknown annotation type.";
                    }
                    // batchsize中第item_id个图像的标注
                    all_anno[item_id] = transformed_anno_vec;
                } else {
                    this->data_transformer_->Transform(sampled_datum->datum(),
                                                       &(this->transformed_data_));
                    // Otherwise, store the label from datum.
                    CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
                    top_label[item_id] = sampled_datum->datum().label();
                }
            } else {
                this->data_transformer_->Transform(sampled_datum->datum(),
                                                   &(this->transformed_data_));
            }
            // 内存清理
            if (has_sampled) {
                delete sampled_datum;
            }
            if (transform_param.has_expand_param()) {
                delete expand_datum;
            }
            trans_time += timer.MicroSeconds();

            reader_.free().push(const_cast<AnnotatedDatum *>(&anno_datum));
        }

        // Store "rich" annotation if needed.
        /*最后将标注信息保存到top[1]中，top[1]的shape:[1,1,numberOfBoxes,8]
        *每一行格式：[item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        *这个8维向量表示的含义：batchsize个图像中的第item_id幅图像中的第group_label个类别下的第instance_id个box的坐标为[xmin, ymin, xmax, ymax]
        */
        if (this->output_labels_ && has_anno_type_) {
            vector<int> label_shape(4);
            if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                label_shape[0] = 1;
                label_shape[1] = 1;
                label_shape[3] = 8;
                if (num_bboxes == 0) {
                    // Store all -1 in the label.
                    label_shape[2] = 1;
                    batch->label_.Reshape(label_shape);
                    caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
                } else {
                    // Reshape the label and store the annotation.
                    // num_bboxes就是前面crop出来的所有图像中所有目标的数量
                    label_shape[2] = num_bboxes;
                    batch->label_.Reshape(label_shape);
                    top_label = batch->label_.mutable_cpu_data();
                    int idx = 0;
                    // 遍历bachsizes中每一幅图像的label信息
                    for (int item_id = 0; item_id < batch_size; ++item_id) {
                        // 第ite_id幅图像的label信息
                        const vector<AnnotationGroup> &anno_vec = all_anno[item_id];
                        for (int g = 0; g < anno_vec.size(); ++g) {
                            const AnnotationGroup &anno_group = anno_vec[g];
                            for (int a = 0; a < anno_group.annotation_size(); ++a) {
                                const Annotation &anno = anno_group.annotation(a);
                                const NormalizedBBox &bbox = anno.bbox();
                                top_label[idx++] = item_id;
                                top_label[idx++] = anno_group.group_label();
                                top_label[idx++] = anno.instance_id();
                                top_label[idx++] = bbox.xmin();
                                top_label[idx++] = bbox.ymin();
                                top_label[idx++] = bbox.xmax();
                                top_label[idx++] = bbox.ymax();
                                top_label[idx++] = bbox.difficult();
                            }
                        }
                    }
                }
            } else {
                LOG(FATAL) << "Unknown annotation type.";
            }
        }
        timer.Stop();
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    INSTANTIATE_CLASS(AnnotatedMatrixDataLayer);

    REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
