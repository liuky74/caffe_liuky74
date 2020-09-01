// caffe.hpp is the header file that you need to include in your code. It wraps
// all the internal caffe header files into one for simpler inclusion.

#ifndef CAFFE_CAFFE_HPP_
#define CAFFE_CAFFE_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
/*layer_factory维护一个静态类工厂类,用于注册层*/
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
/*solver_factory维护一个静态类工厂类,用于注册解算器*/
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

#endif  // CAFFE_CAFFE_HPP_
