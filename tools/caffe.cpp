#ifdef WITH_PYTHON_LAYER

#include "boost/python.hpp"

namespace bp = boost::python;
#endif
/*google开源的命令行参数解析工具*/
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
/* caffe的头文件中会导入solver_factory以及layer_factory两个工厂类
 * 这两个工厂类会在加载头文件的过程中导入所有的layer(层)和solver(解算器)定义*/
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

/*gflags是google的一个开源的处理命令行参数的库。
  在使用命令行参数的文件文件中（源文件或头文件），首先使用一下定义语句进行变量的定义。
  DEFINE_int32，DEFINE_int64，DEFINE_bool，DEFINE_double，DEFINE_string等，
  语法为：DEFINE_int32(name, default_value, "description")。
  接着你就可以使用FLAGS_name变量了，这些变量的值则是由命令行参数传递，无则为默认值，
  在其他代码文件中若想用该命令参数，可以用DECLARE_int32(name)声明（name为int32类型，也可以使用其他支持的类型）。
  在caffe.cpp中有很多FLAGS_name定义，如DEFINE_string(gpu,"","some description"），则命令行后-gpu 0，表示FLAGS_gpu=0，默认值为空。
  */

DEFINE_string(gpu, "",
              "Optional; run in GPU mode on given device IDs separated by ','."
              "Use '-gpu all' to run on all available GPUs. The effective training "
              "batch size is multiplied by the number of devices.");
DEFINE_string(solver, "",
              "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
              "The model definition protocol buffer text file.");
DEFINE_string(phase, "",
              "Optional; network phase (TRAIN or TEST). Only used for 'time'.");
DEFINE_int32(level, 0,
             "Optional; network level.");
DEFINE_string(stage, "",
              "Optional; network stages (not to be confused with phase), "
              "separated by ','.");
DEFINE_string(snapshot, "",/*快照，便于恢复训练时使用*/
              "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",/*加载预训练模型权重参数，便于网络finetune，不可与snapshot参数共用*/
              "Optional; the pretrained weights to initialize finetuning, "
              "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
             "The number of iterations to run.");
DEFINE_string(sigint_effect, "stop",
              "Optional; action to take when a SIGINT signal is received: "
              "snapshot, stop or none.");
DEFINE_string(sighup_effect, "snapshot",
              "Optional; action to take when a SIGHUP signal is received: "
              "snapshot, stop or none.");

// A simple registry for caffe commands.
/*声明函数指针BrewFunction用于指向[train，test，device_query，time]这几个执行函数*/
typedef int (*BrewFunction)();

/*定义一个容器类,用于储存[train，test，device_query，time]这几个执行函数,以函数名为key,利用函数指针保存函数*/
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;/*声明一个上文的这个容器类*/



/* 用宏定义的方式定义了train()，test()，device_query()，time()四个函数的四个不同类,
 * 并将这4个函数注册到g_brew_map中.
 * 在定义完成后同时还声明了一个类对象
 *
 * 以train为例:
 * class __Registerer_train{
 * public:
 *     __Registerer_train() {
 *         g_brew_map["train"] = &train;
 *     }
 * }
 * __Registerer_train g_registerer_train;
 * */
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* 将定义的类注册到g_brew_map容器中 */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; /*定义完成后立刻声明一个类对象*/\
__Registerer_##func g_registerer_##func; \
}

/*在caffe.cpp 中 BrewFunction 作为GetBrewFunction()函数的返回类型，
  可以是 train()，test()，device_query()，time() 这四个函数指针的其中一个。
  在train()，test()，中可以调用solver类的函数，从而进入到net，进入到每一层，运行整个caffe程序。
  */
static BrewFunction GetBrewFunction(const caffe::string &name) {
    if (g_brew_map.count(name)) {/*判断输入的是不是g_brew_map中train，test，device_query，time中一个*/
        return g_brew_map[name];/*这是一个函数映射,会返回对应的函数*/
    } else {
        LOG(ERROR) << "Available caffe actions:";
        for (BrewMap::iterator it = g_brew_map.begin();
             it != g_brew_map.end(); ++it) {
            LOG(ERROR) << "\t" << it->first;/*LOG来源于google的glog库，控制程序的日志输出消息和测试消息*/
        }
        LOG(FATAL) << "Unknown action: " << name;
        return NULL;  // not reachable, just to suppress old compiler warnings.
    }
}

// Parse GPU ids or use all available devices
/*解析可用的GPU*/
static void get_gpus(vector<int> *gpus) {
    if (FLAGS_gpu == "all") {
        int count = 0;
#ifndef CPU_ONLY/*如果未定义了只用CPU的话，CUDA会寻找可用GPU*/
        CUDA_CHECK(cudaGetDeviceCount(&count));
#else
        NO_GPU;
#endif
        for (int i = 0; i < count; ++i) {
            gpus->push_back(i);
        }
    } else if (FLAGS_gpu.size()) {
        vector<string> strings;
        boost::split(strings, FLAGS_gpu, boost::is_any_of(","));
        for (int i = 0; i < strings.size(); ++i) {
            gpus->push_back(boost::lexical_cast<int>(strings[i]));/*将所有GPU的id存进gpus*/
        }
    } else {
        CHECK_EQ(gpus->size(), 0);
    }
}

// Parse phase from flags
/*判断是训练还是测试*/
caffe::Phase get_phase_from_flags(caffe::Phase default_value) {
    if (FLAGS_phase == "")
        return default_value;
    if (FLAGS_phase == "TRAIN")
        return caffe::TRAIN;
    if (FLAGS_phase == "TEST")
        return caffe::TEST;
    LOG(FATAL) << "phase must be \"TRAIN\" or \"TEST\"";
    return caffe::TRAIN;  // Avoid warning
}

// Parse stages from flags
vector<string> get_stages_from_flags() {
    vector<string> stages;
    boost::split(stages, FLAGS_stage, boost::is_any_of(","));
    return stages;
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
/*GPU诊断*/
int device_query() {
    LOG(INFO) << "Querying GPUs " << FLAGS_gpu;
    vector<int> gpus;
    get_gpus(&gpus);
    for (int i = 0; i < gpus.size(); ++i) {
        caffe::Caffe::SetDevice(gpus[i]);
        caffe::Caffe::DeviceQuery();
    }
    return 0;
}
/*通过宏声明device_query的注册类,并将该函数注册进入g_brew_map表中*/
RegisterBrewFunction(device_query);

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
/*加载训练的或者传入的模型*/
void CopyLayers(caffe::Solver<float> *solver, const std::string &model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(","));
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        solver->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < solver->test_nets().size(); ++j) {
            solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

// Translate the signal effect the user specified on the command-line to the
// corresponding enumeration.
/*将交互端传来的string类型的标志转成枚举类型的变量*/
caffe::SolverAction::Enum GetRequestedAction(
        const std::string &flag_value) {
    if (flag_value == "stop") {
        return caffe::SolverAction::STOP;
    }
    if (flag_value == "snapshot") {
        return caffe::SolverAction::SNAPSHOT;
    }
    if (flag_value == "none") {
        return caffe::SolverAction::NONE;
    }
    LOG(FATAL) << "Invalid signal effect \"" << flag_value << "\" was specified";
}

// Train / Finetune a model.
/*模型的训练函数,训练或者微调网络都是走这个分支*/
int train() {
    /*google的glog库，检查--solver、--snapshot和--weight并输出消息；必须有指定solver，并且snapshot和weight两者只需指定其一；*/
    CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
    CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())/*必须传入solver文件*/
                                                                  << "Give a snapshot to resume training or weights to finetune "
                                                                     "but not both.";
    vector<string> stages = get_stages_from_flags();
    /*声明一个solver解算器参数类*/
    /*SolverParameter是通过Google Protocol Buffer自动生成的一个类*/
    caffe::SolverParameter solver_param;/*定义SolverParameter的对象，该类保存solver参数和相应的方法*/
    caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);/*传入solver文件,解析参数传入solver_param对象*/

    /*此处定义了level和stage ，应该在caffe::Net函数中有具体定义*/
    solver_param.mutable_train_state()->set_level(FLAGS_level);
    for (int i = 0; i < stages.size(); i++) {
        solver_param.mutable_train_state()->add_stage(stages[i]);
    }

    // If the gpus flag is not provided, allow the mode and device to be set
    /*根据命令参数-gpu或者solver.prototxt提供的信息设置GPU*/
    if (FLAGS_gpu.size() == 0// in the solver prototxt.如果没有提供gpu参数,caffe会自己检测是否有可用的gpu
        && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
        if (solver_param.has_device_id()) {/*如果没有设置GPU device id,则默认设置使用设备id为0*/
            FLAGS_gpu = "" +
                        boost::lexical_cast<string>(solver_param.device_id());
        } else {  // Set default GPU if unspecified
            FLAGS_gpu = "" + boost::lexical_cast<string>(0);
        }
    }

    /*多GPU下，将GPU编号存入vector容器中（get_gpus()函数通过FLAGS_gpu获取）*/
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() == 0) {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    } else {
        ostringstream s;
        for (int i = 0; i < gpus.size(); ++i) {
            s << (i ? ", " : "") << gpus[i];
        }
        LOG(INFO) << "Using GPUs " << s.str();
#ifndef CPU_ONLY/*不是CPU模式下,输出GPU信息*/
        cudaDeviceProp device_prop;
        for (int i = 0; i < gpus.size(); ++i) {
            cudaGetDeviceProperties(&device_prop, gpus[i]);
            LOG(INFO) << "GPU " << gpus[i] << ": " << device_prop.name;
        }
#endif
        solver_param.set_device_id(gpus[0]);/*开始修正解算器的参数*/
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
        Caffe::set_solver_count(gpus.size());
    }
    /* 信号捕获器
     * 处理snapshot, stop or none信号，其声明在include/caffe/util/signal_Handler.h中
     * GetRequestedAction在caffe.cpp中，将‘stop’，‘snapshot’，‘none’转换为标准信号，即解析；
     */
    caffe::SignalHandler signal_handler(
            GetRequestedAction(FLAGS_sigint_effect),
            GetRequestedAction(FLAGS_sighup_effect));
    /* 将定义好的参数传入solver类,创建solver解算器对象并使用智能指针进行包装
     * 注意CreateSolver(solver_param)函数已经初始化了解算器
     * 初始化流程:
     *   所有解算器都会调用父类solver的构造函数,同时还会调用preSolver函数进行预处理
     *   父类solver的构造函数会加载solver文件中的参数进行solver初始化
     *   在solver初始化的过程中会调用Net的init函数对网络结构进行初始化
     * */
    shared_ptr<caffe::Solver<float>> solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    /* 通过GetActionFunction来处理获得的系统信号
     * 在SetActionFunction中将GetActionFunction函数地址传给参数action_request_function_
     * 在网络训练的过程中，在GetRequestedAction中来处理action_request_function_得到的函数指针
     * */
    solver->SetActionFunction(signal_handler.GetActionFunction());
    /* 判断了一下用户是否定义了snapshot或者weights这两个参数中的一个
     * 如果定义了则需要通过Solver提供的接口从snapshot或者weights文件
     * 中去读取已经训练好的网络的参数，来接着训练*/
    if (FLAGS_snapshot.size()) {
        LOG(INFO) << "Resuming from " << FLAGS_snapshot;
        solver->Restore(FLAGS_snapshot.c_str());
    } else if (FLAGS_weights.size()) {
        CopyLayers(solver.get(), FLAGS_weights);
    }

    /*运行解算器开始迭代训练/测试模型*/
    if (gpus.size() > 1) {/*针对多GPU进行优化*/
        caffe::P2PSync<float> sync(solver, NULL, solver->param());
        sync.Run(gpus);/*执行解算器(迭代训练/测试)*/
    } else {
        LOG(INFO) << "Starting Optimization";
        solver->Solve();/*Starting Optimization 执行解算器(迭代训练/测试)*/
    }
    LOG(INFO) << "Optimization Done.";
    return 0;
}
/*将train函数注册,并写入g_brew_map键值对中*/
RegisterBrewFunction(train);


// Test: score a model.
int test() {
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
    CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
    vector<string> stages = get_stages_from_flags();

    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() != 0) {
        LOG(INFO) << "Use GPU with device ID " << gpus[0];
#ifndef CPU_ONLY
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, gpus[0]);
        LOG(INFO) << "GPU device name: " << device_prop.name;
#endif
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }
    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, caffe::TEST, FLAGS_level, &stages);
    caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
    LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

    vector<int> test_score_output_id;
    vector<float> test_score;
    float loss = 0;
    for (int i = 0; i < FLAGS_iterations; ++i) {
        float iter_loss;
        const vector<Blob<float> *> &result =
                caffe_net.Forward(&iter_loss);
        loss += iter_loss;
        int idx = 0;
        for (int j = 0; j < result.size(); ++j) {
            const float *result_vec = result[j]->cpu_data();
            for (int k = 0; k < result[j]->count(); ++k, ++idx) {
                const float score = result_vec[k];
                if (i == 0) {
                    test_score.push_back(score);
                    test_score_output_id.push_back(j);
                } else {
                    test_score[idx] += score;
                }
                const std::string &output_name = caffe_net.blob_names()[
                        caffe_net.output_blob_indices()[j]];
                LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
            }
        }
    }
    loss /= FLAGS_iterations;
    LOG(INFO) << "Loss: " << loss;
    for (int i = 0; i < test_score.size(); ++i) {
        const std::string &output_name = caffe_net.blob_names()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        const float loss_weight = caffe_net.blob_loss_weights()[
                caffe_net.output_blob_indices()[test_score_output_id[i]]];
        std::ostringstream loss_msg_stream;
        const float mean_score = test_score[i] / FLAGS_iterations;
        if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * mean_score << " loss)";
        }
        LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    }

    return 0;
}

RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
    CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";
    caffe::Phase phase = get_phase_from_flags(caffe::TRAIN);
    vector<string> stages = get_stages_from_flags();

    // Set device id and mode
    vector<int> gpus;
    get_gpus(&gpus);
    if (gpus.size() != 0) {
        LOG(INFO) << "Use GPU with device ID " << gpus[0];
        Caffe::SetDevice(gpus[0]);
        Caffe::set_mode(Caffe::GPU);
    } else {
        LOG(INFO) << "Use CPU.";
        Caffe::set_mode(Caffe::CPU);
    }
    // Instantiate the caffe net.
    Net<float> caffe_net(FLAGS_model, phase, FLAGS_level, &stages);

    // Do a clean forward and backward pass, so that memory allocation are done
    // and future iterations will be more stable.
    LOG(INFO) << "Performing Forward";
    // Note that for the speed benchmark, we will assume that the network does
    // not take any input blobs.
    float initial_loss;
    caffe_net.Forward(&initial_loss);
    LOG(INFO) << "Initial loss: " << initial_loss;
    LOG(INFO) << "Performing Backward";
    caffe_net.Backward();

    const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();
    const vector<vector<Blob<float> *> > &bottom_vecs = caffe_net.bottom_vecs();
    const vector<vector<Blob<float> *> > &top_vecs = caffe_net.top_vecs();
    const vector<vector<bool> > &bottom_need_backward =
            caffe_net.bottom_need_backward();
    LOG(INFO) << "*** Benchmark begins ***";
    LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
    Timer total_timer;
    total_timer.Start();
    Timer forward_timer;
    Timer backward_timer;
    Timer timer;
    std::vector<double> forward_time_per_layer(layers.size(), 0.0);
    std::vector<double> backward_time_per_layer(layers.size(), 0.0);
    double forward_time = 0.0;
    double backward_time = 0.0;
    for (int j = 0; j < FLAGS_iterations; ++j) {
        Timer iter_timer;
        iter_timer.Start();
        forward_timer.Start();
        for (int i = 0; i < layers.size(); ++i) {
            timer.Start();
            layers[i]->Forward(bottom_vecs[i], top_vecs[i]);
            forward_time_per_layer[i] += timer.MicroSeconds();
        }
        forward_time += forward_timer.MicroSeconds();
        backward_timer.Start();
        for (int i = layers.size() - 1; i >= 0; --i) {
            timer.Start();
            layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                                bottom_vecs[i]);
            backward_time_per_layer[i] += timer.MicroSeconds();
        }
        backward_time += backward_timer.MicroSeconds();
        LOG(INFO) << "Iteration: " << j + 1 << " forward-backward time: "
                  << iter_timer.MilliSeconds() << " ms.";
    }
    LOG(INFO) << "Average time per layer: ";
    for (int i = 0; i < layers.size(); ++i) {
        const caffe::string &layername = layers[i]->layer_param().name();
        LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
                  "\tforward: " << forward_time_per_layer[i] / 1000 /
                                   FLAGS_iterations << " ms.";
        LOG(INFO) << std::setfill(' ') << std::setw(10) << layername <<
                  "\tbackward: " << backward_time_per_layer[i] / 1000 /
                                    FLAGS_iterations << " ms.";
    }
    total_timer.Stop();
    LOG(INFO) << "Average Forward pass: " << forward_time / 1000 /
                                             FLAGS_iterations << " ms.";
    LOG(INFO) << "Average Backward pass: " << backward_time / 1000 /
                                              FLAGS_iterations << " ms.";
    LOG(INFO) << "Average Forward-Backward: " << total_timer.MilliSeconds() /
                                                 FLAGS_iterations << " ms.";
    LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() << " ms.";
    LOG(INFO) << "*** Benchmark ends ***";
    return 0;
}

RegisterBrewFunction(time);

int main(int argc, char **argv) {
    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    // Set version
    gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
    // Usage message.
    gflags::SetUsageMessage("command line brew\n"
                            "usage: caffe <command> <args>\n\n"
                            "commands:\n"
                            "  train           train or finetune a model\n"
                            "  test            score a model\n"
                            "  device_query    show GPU diagnostic information\n"
                            "  time            benchmark model execution time");
    // Run tool or show usage.
    caffe::GlobalInit(&argc, &argv);
    if (argc == 2) {
#ifdef WITH_PYTHON_LAYER
        try {
#endif
            /*从g_brew_map数组中取出[train,test]函数并执行*/
            return GetBrewFunction(caffe::string(argv[1]))();
#ifdef WITH_PYTHON_LAYER
        } catch (bp::error_already_set) {
            PyErr_Print();
            return 1;
        }
#endif
    } else {
        gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
    }
}
