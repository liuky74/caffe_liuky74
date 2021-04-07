//
// This script converts the CIFAR dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_cifar_data input_folder output_db_file
// The CIFAR dataset could be downloaded at
//    http://www.cs.toronto.edu/~kriz/cifar.html

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <iostream>
#include <caffe/util/io.hpp>

#include "boost/scoped_ptr.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include <cstdlib>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

using caffe::Datum;
using boost::scoped_ptr;
using std::string;
namespace db = caffe::db;

const int kCIFARSize = 32;
const int kCIFARImageNBytes = 3072;
const int kCIFARBatchSize = 10000;
const int kCIFARTrainBatches = 5;

std::map<string, int> ALL_CLASSES = {
        {"__background__", 0},
        {"marker", 1},/*必须是标注类*/
        {"smoke", 2},
        {"smoke_gray", 3},
        {"water_mist", 4},
        {"water", 5},
        {"vehicle", 6},
};


struct Det_S {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int cls_num;
    string cls_name;

};

void read_image(std::ifstream *file, int *label, char *buffer) {
    char label_char;
    file->read(&label_char, 1);
    *label = label_char;
    file->read(buffer, kCIFARImageNBytes);
    return;
}

void load_label(const string &label_path, std::vector<Det_S> &dets_vec) {
    std::ifstream label_file;
    label_file.open(label_path.c_str());
    char get_line[1024],get_line2[1024];
    char *split;
    while (!label_file.eof()) {
        label_file.getline(get_line, 1024);
        if(strlen(get_line)<=0) break;
        if (get_line[strlen(get_line) - 1] == '\r') {
            get_line[strlen(get_line) - 1] = 0;
        }
        Det_S det;
        split = std::strtok(get_line, ",");
        det.xmin = strtof(split, nullptr);
        split = std::strtok(nullptr, ",");
        det.ymin = strtof(split, nullptr);
        split = std::strtok(nullptr, ",");
        det.xmax = strtof(split, nullptr);
        split = std::strtok(nullptr, ",");
        det.ymax = strtof(split, nullptr);
        split = std::strtok(nullptr, ",");
        /*跳过标注类,最后统一添加*/
        if(strcmp(split,"marker") == 0){
            continue;
        }
        det.cls_name = split;
        det.cls_num = ALL_CLASSES[split];
        dets_vec.push_back(det);
    }
    /*统一添加标注类*/
    Det_S det={0.1,0.1,0.1,0.1,ALL_CLASSES["marker"],"marker"};
    dets_vec.push_back(det);
}

int read_video(const string &file_name, cv::Mat (&video_data)[12]) {

    cv::VideoCapture cap;
    bool is_open = cap.open(file_name);
    if (!is_open) {
        printf("|INFO: can't open %s|", file_name.c_str());
        return -1;
    }
    cv::Mat img;
    for (int idx=0; idx < 12; idx++) {
        cap.read(img);
        cv::resize(img,video_data[idx],cv::Size2i(300,300));
    }
    return 0;
}

std::vector<string> get_file_name(const string &file_path) {
    std::ifstream infile;
    std::vector<string> file_name_vec;
    char file_name[1024];
    if (access(file_path.c_str(), F_OK) == -1) {
        printf("|ERR|%s not exist|\n", file_path.c_str());
        exit(0);
    }
    infile.open(file_path.c_str());
    while (!infile.eof()) {
        infile.getline(file_name, 1024);
        std::flush(std::cout);
        if(strlen(file_name)>0) file_name_vec.push_back((string) (file_name));
        else printf("|WAR|getline for file name len <=0, print: %s.\n|",file_name);
    }
    return file_name_vec;
}


void img_concate(cv::Mat (&imgs)[12], cv::Mat &output){



    std::vector<cv::Mat> R_channels_vec;
    std::vector<cv::Mat> G_channels_vec;
    std::vector<cv::Mat> B_channels_vec;
    for(cv::Mat img:imgs){
        cv::imwrite("/befor_split.jpg",img);
        std::vector<cv::Mat> split_mats;
        cv::split(img,split_mats);
        R_channels_vec.push_back(split_mats[0]);
        G_channels_vec.push_back(split_mats[1]);
        B_channels_vec.push_back(split_mats[2]);
    }
    cv::Mat R_channels;
    cv::Mat G_channels;
    cv::Mat B_channels;
    std::vector<cv::Mat> output_vec;
    cv::merge(R_channels_vec,R_channels);
    output_vec.push_back(R_channels);
    cv::merge(G_channels_vec,G_channels);
    output_vec.push_back(G_channels);
    cv::merge(B_channels_vec,B_channels);
    output_vec.push_back(B_channels);
    cv::merge(output_vec,output);

//测试代码
/*    std::vector<cv::Mat> test_split_mats;
    std::vector<cv::Mat> test_mat_vec;
    cv::Mat test_mat;
    char save_path[200];
    cv::split(output,test_split_mats);
    for(int i=0;i<12;i++){
        test_mat_vec.clear();
        test_mat_vec.push_back(test_split_mats[i+0]);
        test_mat_vec.push_back(test_split_mats[i+12]);
        test_mat_vec.push_back(test_split_mats[i+24]);
        cv::merge(test_mat_vec,test_mat);
        sprintf(save_path,"/tmp_%d.jpg",i);
        cv::imwrite(save_path,test_mat);
    }*/
}

void convert_dataset(string *input_folders, int dir_size, const string &output_folder,
                     const string &db_type,bool test= false) {
    scoped_ptr<db::DB> train_db(db::GetDB(db_type));
    string lmdb_save_path;
    if(test){
        lmdb_save_path = output_folder + "/smoke_car_test_" + db_type;
    } else{
        lmdb_save_path = output_folder + "/smoke_car_train_" + db_type;
    }

    /*判断是否有旧文件,有则删除*/
    if (access(lmdb_save_path.c_str(), F_OK) != -1) {
        char command_line[200];
        sprintf(command_line, "rm -rf %s", lmdb_save_path.c_str());
        system(command_line);
    }
    /*新建lmdb文件*/
    train_db->Open(lmdb_save_path, db::NEW);
    /*获取lmdb文件的空间指针*/
    scoped_ptr<db::Transaction> txn(train_db->NewTransaction());
    int annotation_idx=0;

    LOG(INFO) << "Writing Training data";
    for (int idx = 0; idx < dir_size; idx++) {/*文件夹循环*/
        string input_folder = input_folders[idx];
        string input_data_folder = input_folder + "/data";
        string input_label_folder = input_folder + "/label";
        string data_name_file_path = input_folder + "/file_names.txt";
        std::vector<string> file_name_vec;
        file_name_vec = get_file_name(data_name_file_path);

        for (const string &file_name:file_name_vec) {/*文件循环*/
            string data_path = input_data_folder + "/" + file_name + ".mp4";
            string label_path = input_label_folder + "/" + file_name + ".txt";
            cv::Mat video_data[12];
            std::vector<Det_S> dets;
/*            auto a = dets.size();
            if(a>1){
                std::cout<<"---------------------------------------"<<std::endl;
                for(Det_S _det:dets){
                    printf("|%f,%f,%f,%f,%d,%s|\n",_det.xmin,_det.ymin,_det.xmax,_det.ymax,_det.cls_num,_det.cls_name.c_str());

                }
            }*/
            /*读取视频文件,存入video_data中*/
            if(read_video(data_path, video_data)==-1){
                exit(0);
            }
            /*读取label文件,存入dets中*/
            load_label(label_path, dets);
            /*新建annotated datum,这是SSD专用的datum格式*/
            caffe::AnnotatedDatum annotated_datum;
            annotated_datum.set_type(caffe::AnnotatedDatum_AnnotationType_BBOX);

            /*循环将dets中的数据存入datum中*/
            for(int cls_i=1;cls_i<ALL_CLASSES.size();cls_i++){/*从1开始是因为0是背景类*/
                int instance_id=0;
                /*每个类别会被分为一个group,声明一个指针用于获取datum中各个group的内存位置*/
                caffe::AnnotationGroup *annotation_group= nullptr;
                for(int i=0; i < dets.size(); i++) {
                    if(dets[i].cls_num!=cls_i){
                        continue;
                    }
                    /*当前训练数据并不是每个类都有的,只有在包含某个类的det时才会创建对应的group*/
                    if(annotation_group == nullptr){
                        annotation_group = annotated_datum.add_annotation_group();
                        annotation_group->set_group_label(cls_i);
                    }
                    /*声明一个annotation来保存box*/
                    caffe::Annotation *annotation = annotation_group->add_annotation();
                    /*当前分类下的第几个det*/
                    annotation->set_instance_id(instance_id);
                    instance_id++;
                    /*声明一个bbox类,用于保存box*/
                    caffe::NormalizedBBox *normalizedBBox = annotation->mutable_bbox();
                    normalizedBBox->set_xmin(dets[i].xmin);
                    normalizedBBox->set_ymin(dets[i].ymin);
                    normalizedBBox->set_xmax(dets[i].xmax);
                    normalizedBBox->set_ymax(dets[i].ymax);
                    /*默认不是难分对象*/
                    normalizedBBox->set_difficult(false);
                }
            }
            /*开始保存图像数据*/
            caffe::Datum *video_datum = annotated_datum.mutable_datum();
//            cv::Mat video_mat(300,300,CV_8UC(32));
            cv::Mat video_mat;
            /*将12帧图片转换为[300,300,36]的Mat对象,存入video_mat*/
            img_concate(video_data,video_mat);
            /*cv::img->datum数据转换*/
            caffe::CVMatToDatum(video_mat,video_datum);
            /*将数据指向进annotated中*/
//            annotated_datum.set_allocated_datum(&video_datum);
            string serialize_data;
            CHECK(annotated_datum.SerializeToString(&serialize_data));
            txn->Put(caffe::format_int(annotation_idx), serialize_data);
            annotation_idx++;
            if(annotation_idx%500==499){
                txn->Commit();
                txn.reset(train_db->NewTransaction());
                printf("|INFO|process %d|\n",annotation_idx);
            }
        }

    }
    txn->Commit();
    train_db->Close();
}


int main(int argc, char **argv) {
    const char env_name[20] = "LANG";
    setenv("LANG", "zh_CN_UTF-8", 1);


    FLAGS_alsologtostderr = 1;
    string output_dir = "/workspace/data/smokeCar_lmdb";
    string model = "lmdb";
    bool build_name_file = true;
    int dirs_count = 0;
    char command_line[1000];

    //生成训练样本文件列表
    string input_dirs[] = {
            "/workspace/data/smoke/train_data/rebuild_datas/AutoBuild_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/base_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/DeAn_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/GuRun_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/HeNeng_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/TongHua_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/WanZai_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/XinXiang_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/YiFeng_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/YiFeng_rain_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/YunJing_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/ZhangYe_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/ZhangYeROI_dataset",
            "/workspace/data/smoke/train_data/rebuild_datas/ZhangYe2_dataset",
    };

    //train lmdb数据生成
    for (string dir:input_dirs) {
        if (build_name_file) {
            sprintf(command_line, "/bin/bash /workspace/caffe/examples/custom_obj_detect/get_file_name.sh %s",
                    dir.c_str());
            system(command_line);
        }
        dirs_count++;
    }
    google::InitGoogleLogging(argv[0]);
    convert_dataset(input_dirs, dirs_count, output_dir, model);

    //test lmdb数据生成
    build_name_file= true;
    string test_dirs[] = {
            "/workspace/data/smoke/train_data/rebuild_datas/XinXiang_dataset",
    };
    dirs_count=0;
    for (string dir:test_dirs) {
        if (build_name_file) {
            sprintf(command_line, "/bin/bash /workspace/caffe/examples/custom_obj_detect/get_file_name.sh %s",
                    dir.c_str());
            system(command_line);
        }
        dirs_count++;
    }
    convert_dataset(test_dirs, dirs_count, output_dir, model,true);

    return 0;
}
