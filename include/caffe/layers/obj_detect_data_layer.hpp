#ifndef CAFFE_OBJ_DETECT_DATA_LAYER_HPP
#define CAFFE_OBJ_DETECT_DATA_LAYER_HPP
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe{
    template <typename Dtype>
    class ObjDetectDataLayer : public BasePrefetchingDataLayer<Dtype>{
    public:
        explicit ObjDetectDataLayer(const LayerParameter &param);
        virtual ~ObjDetectDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype> * > &bottom,
                                    const vector<Blob<Dtype> * > &top);
        virtual inline bool ShareInParallel() const{ return false;};
        virtual inline const char * type() {return "ObjDetectData";};
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 2; }
        vector<int> InferBlobShape(const Datum &datum);


    protected:
        virtual void load_batch(Batch<Dtype> * batch);
        AnnotatedDatum_AnnotationType anno_type_;
        string label_map_file;
        DataReader<AnnotatedDatum> reader_;

    };


}

#endif //CAFFE_OBJ_DETECT_DATA_LAYER_HPP
