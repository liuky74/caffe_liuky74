#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif

#include <vector>
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/obj_detect_data_layer.hpp"

namespace caffe{
    template<typename Dtype>
    ObjDetectDataLayer<Dtype>::ObjDetectDataLayer(const LayerParameter &param)
    :BasePrefetchingDataLayer<Dtype>(param),reader_(param){}

    template <typename Dtype>
    ObjDetectDataLayer<Dtype>::~ObjDetectDataLayer() {
        this->StopInternalThread();
    }
    template <typename Dtype>
    void ObjDetectDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                   const vector<Blob<Dtype> *> &top) {
        const int batch_size = this->layer_param_.data_param().batch_size();
        AnnotatedDatum &annotated_datum = *(reader_.full().peek());
        label_map_file = this->layer_param_.obj_detect_param().label_map_file();
        vector<int> top_shape = this->InferBlobShape(annotated_datum.datum());
        this->transformed_data_.Reshape(top_shape);
        top_shape[0]=batch_size;
        printf("shape:[%i,%i,%i,%i]",top_shape[0],top_shape[1],top_shape[2],top_shape[3]);
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

        /*调整输出1(GT bbox)的shape*/
        /*[batch_size,1,num_bboxes,[item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]]*/
        vector<int> label_shape(4,1);
        int num_bboxes = 0;
        for(int cls_idx=0;cls_idx<annotated_datum.annotation_group_size();cls_idx++){
            num_bboxes += annotated_datum.annotation_group(cls_idx).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        label_shape[2] = std::max(num_bboxes,1);
        label_shape[3] = 8;
        top[1]->Reshape(label_shape);
        for(int idx =0;idx<this->PREFETCH_COUNT;++idx){
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }
    template <typename Dtype>
    void ObjDetectDataLayer<Dtype>::load_batch(caffe::Batch<Dtype> *batch) {}

    template<typename Dtype>
    vector<int> ObjDetectDataLayer<Dtype>::InferBlobShape(const Datum &datum) {
        vector<int> shape(4);
        shape[1] = datum.channels();
        shape[2] = datum.height();
        shape[3] = datum.width();

        return shape;
    }


    INSTANTIATE_CLASS(ObjDetectDataLayer);
    REGISTER_LAYER_CLASS(ObjDetectData);

}