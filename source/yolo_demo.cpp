#include <memory>
#include <stdlib.h>
#include <MNN/Interpreter.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace MNN;

int main(int argc, char** argv) {
    auto model_name = argv[1];
    auto img_name = argv[2];
    // model support 1x3x320x320
    auto size = 320;
    // read img
    cv::Mat resized_img;
    cv::Mat img;
    auto src_img = cv::imread(img_name, cv::IMREAD_ANYDEPTH);
    cv::resize(src_img, resized_img, cv::Size(size, size));
    resized_img.convertTo(img, CV_32F, 1.0/255.0);
    // pre deal
    // ...
    // create net
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(model_name));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    // create session
    auto session = net->createSession(config);
    // get input tensor
    auto input = net->getSessionInput(session, NULL);
    // reshape of input
    auto shape = input->shape();
    // NCHW: default is 1x3x320x320
    shape[2] = size;
    shape[3] = size;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    // get output tensor
    auto output_0 = net->getSessionOutput(session, "output");
    auto output_1 = net->getSessionOutput(session, "781");
    auto output_2 = net->getSessionOutput(session, "801");
    // create real input and outputo
    std::shared_ptr<Tensor> input0(new Tensor(input, Tensor::TENSORFLOW)); // img is NHWC = TENSORFLOW
    auto dimType = output_0->getDimensionType();
    std::shared_ptr<Tensor> output0(new Tensor(output_0, dimType)), 
                            output1(new Tensor(output_1, dimType)),
                            output2(new Tensor(output_2, dimType));
    // read data from imgae to input0
    ::memcpy(input0->host<float>(), img.ptr(0), img.channels() * img.rows * img.cols * sizeof(float));
    // copy data to input tensor
    input->copyFromHostTensor(input0.get());
    // inference
    net->runSession(session);
    // copy from output tensor
    output_0->copyToHostTensor(output0.get());
    output_1->copyToHostTensor(output1.get());
    output_2->copyToHostTensor(output2.get());
    // deal with output
    printf("outputs_first_ele: %f, %f, %f\n", output0->host<float>()[0],
                                              output1->host<float>()[0],
                                              output2->host<float>()[0]);
    // post process
    return 0;
}
