#ifdef USE_OPENCV

#include <opencv2/core/core.hpp>

#endif

#include <vector>
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/obj_detect_data_layer.hpp"

namespace caffe {

//    template<typename Dtype>
//    ObjDetectDataLayer<Dtype>::ObjDetectDataLayer(const LayerParameter &param)
//            :BasePrefetchingDataLayer<Dtype>(param), reader_(param) {}
    template<typename Dtype>
    ObjDetectDataLayer<Dtype>::ObjDetectDataLayer(const LayerParameter &param)
            : BasePrefetchingDataLayer<Dtype>(param),
              reader_(param) {
    }

    template<typename Dtype>
    ObjDetectDataLayer<Dtype>::~ObjDetectDataLayer() {
        this->StopInternalThread();
    }

    template<typename Dtype>
    void ObjDetectDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                   const vector<Blob<Dtype> *> &top) {
        const int batch_size = this->layer_param_.data_param().batch_size();
        AnnotatedDatum &annotated_datum = *(reader_.full().peek());
        label_map_file = this->layer_param_.obj_detect_param().label_map_file();
        vector<int> top_shape = this->data_transformer_->InferBlobShape(annotated_datum.datum());
        this->transformed_data_.Reshape(top_shape);
        top_shape[0] = batch_size;
        printf("shape:[%i,%i,%i,%i]", top_shape[0], top_shape[1], top_shape[2], top_shape[3]);
        fflush(stdout);
        /*调整输出0的shape*/
        top[0]->Reshape(top_shape);
        /*调整预读取数据的shape*/
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(top_shape);
        }
        LOG(INFO) << "output data size: " << top[0]->num() << ","
                  << top[0]->channels() << "," << top[0]->height() << ","
                  << top[0]->width();
        /*当输出的top.size>1时,表示不单要输出data,还要输出label*/
        if (this->output_labels_) {
            CHECK(annotated_datum.has_type()) << "ERR: 你的datum数据没有type属性";
            anno_type_ = annotated_datum.type();
            /*调整输出1(GT bbox)的shape*/
            /*[batch_size,1,num_bboxes,[item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]]*/
            vector<int> label_shape(4, 1);
            int num_bboxes = 0;
            for (int cls_idx = 0; cls_idx < annotated_datum.annotation_group_size(); cls_idx++) {
                num_bboxes += annotated_datum.annotation_group(cls_idx).annotation_size();
            }
            label_shape[0] = 1;
            label_shape[1] = 1;
            label_shape[2] = std::max(num_bboxes, 1);
            label_shape[3] = 8;
            top[1]->Reshape(label_shape);
            for (int idx = 0; idx < this->PREFETCH_COUNT; ++idx) {
                this->prefetch_[idx].label_.Reshape(label_shape);
            }
        }
    }

    template<typename Dtype>
    void ObjDetectDataLayer<Dtype>::load_batch(caffe::Batch<Dtype> *batch) {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0, trans_time = 0;
        CPUTimer timer;
        CHECK(batch->data_.count());
        CHECK(this->transformed_data_.count());

        const int batch_size = this->layer_param().data_param().batch_size();
        AnnotatedDatum &annotated_datum = *(reader_.full().peek());
        vector<int> top_shape = this->data_transformer_->InferBlobShape(annotated_datum.datum());
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);
        /*取出存数据的内存指针*/
        Dtype *top_data = batch->data_.mutable_cpu_data();
        Dtype *top_label = NULL;
//        if (this->output_labels_) {/*如果需要输出label*/
//            top_label = batch->label_.mutable_cpu_data();
//        }
        map<int, vector<AnnotationGroup>> all_anno;/*整个batch的GT_bbox集合,按照item_id:[cls_idx,[xmin,ymin,xmax,ymax]]的格式储存*/
        int num_boxes = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            timer.Start();
            /*从预读取的队列中取出datum数据*/
            AnnotatedDatum &annotated_datum = *(reader_.full().pop("Waiting for data"));
            AnnotatedDatum *expand_datum = NULL;
            /*不做图像填充*/
            expand_datum = &annotated_datum;

            AnnotatedDatum *sampled_datum = NULL;
            bool has_sample = false;
            /*不对图像做抽样*/
            sampled_datum = expand_datum;
            CHECK(sampled_datum != NULL);
            /*不做图像预处理*/
            timer.Start();
//            vector<int> shape = this->InferBlobShape(sampled_datum->datum());
            vector<int> shape = this->data_transformer_->InferBlobShape(sampled_datum->datum());
            /*检测经过处理后的图像数据shape是否与输出shape匹配,注意掉过了batch size,只匹配了CHW*/
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4, shape.begin() + 1));

            /*得到batch数组的内存偏移量*/
            int offset = batch->data_.offset(item_id);
            /* transformed_data_中维护一个内存指针变量cpu_ptr_,将这个内存指针也指向batch数组的偏移位置
             * 这样操作以后对transformed_data_进行操作就等价于对batch[item_id]操作*/
            this->transformed_data_.set_cpu_data(top_data + offset);/*将数组中的内存地址映射给中间成员*/
            vector<AnnotationGroup> transformed_anno_vec;/*存储label的中间成员,储存格式:[cls_idx,[xmin,ymin,xmax,ymax]]*/
            if (this->output_labels_) {
                CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
                sampled_datum->set_type(anno_type_);
                transformed_anno_vec.clear();
                /*数据转换,包含:
                 * crop裁切
                 * 镜像翻转
                 * 减去各通道均值
                 * 将GT box数据归一化到0~1之间
                 最终得到直接进入模型参与计算的数据*/
                this->data_transformer_->Transform(*sampled_datum, &(this->transformed_data_), &transformed_anno_vec);
                for (int cls_idx = 0; cls_idx < transformed_anno_vec.size(); ++cls_idx) {
                    num_boxes += transformed_anno_vec[cls_idx].annotation_size();
                }
                all_anno[item_id] = transformed_anno_vec;
            } else {
                this->data_transformer_->Transform(sampled_datum->datum(),
                                                   &(this->transformed_data_));
            }


            trans_time += timer.MicroSeconds();
            reader_.free().push(const_cast<AnnotatedDatum*>(&annotated_datum));/*将用完的内存放回数组等待下一次使用*/
//            reader_.free().push(&annotated_datum);
        }
        /*和img不同,一个batch只保存一份label数据,这份数据会标注每个box属于哪张img*/
        if (this->output_labels_) {
            vector<int> label_shape(4);//[batch_size,1,box_num,[item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]]
            label_shape[0] = 1;
            label_shape[1] = 1;
            label_shape[3] = 8;
            if(num_boxes == 0){/*没有数据的时候,设置默认值*/
                label_shape[2] = 1;
                batch->label_.Reshape(label_shape);
                caffe_set<Dtype>(8,-1,batch->label_.mutable_cpu_data());
            } else{
                label_shape[2] = num_boxes;
                batch->label_.Reshape(label_shape);
                top_label = batch->label_.mutable_cpu_data();
                int idx = 0;
                for(int item_id = 0;item_id<batch_size;++item_id){
                    const vector<AnnotationGroup> &anno_vec = all_anno[item_id];
                    for(int cls_idx=0;cls_idx<anno_vec.size();++cls_idx){
                        const AnnotationGroup &anno_group = anno_vec[cls_idx];
                        for(int box_idx=0;box_idx<anno_group.annotation_size();++box_idx){
                            const Annotation& anno = anno_group.annotation(box_idx);
                            const NormalizedBBox &bbox = anno.bbox();
                            top_label[idx++] = item_id;
                            top_label[idx++] = anno_group.group_label();
                            top_label[idx++] = anno.instance_id();
                            top_label[idx++] = bbox.xmin();
                            top_label[idx++] = bbox.ymin();
                            top_label[idx++] = bbox.xmax();
                            top_label[idx++] = bbox.ymax();
                            top_label[idx++] = bbox.difficult();
//                            printf("item_id: %i,group_label: %i",int(item_id),int(anno_group.group_label()));
//                            fflush(stdout);
                        }
                    }
                }
            }
        }
        timer.Stop();
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    template<typename Dtype>
    vector<int> ObjDetectDataLayer<Dtype>::InferBlobShape(const Datum &datum) {
        vector<int> shape(4,1);
        shape[1] = datum.channels();
        shape[2] = datum.height();
        shape[3] = datum.width();

        return shape;
    }


    INSTANTIATE_CLASS(ObjDetectDataLayer);

    REGISTER_LAYER_CLASS(ObjDetectData);

}