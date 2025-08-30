// cpp

#include <filesystem>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>

// ROS
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <signal.h>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/opencv.hpp>

// tensorrt
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <cv_bridge/cv_bridge.h>

namespace fs = std::filesystem;
using namespace std::chrono;

nvinfer1::ICudaEngine* engine_{nullptr};
nvinfer1::IExecutionContext* context_{nullptr};
cudaStream_t stream_;
void* buffers_[4]{nullptr, nullptr, nullptr, nullptr};

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

double computeEPE(const cv::Mat& disp_gt, const cv::Mat& disp_pred) {
    CV_Assert(disp_gt.type() == CV_32F && disp_pred.type() == CV_32F);
    CV_Assert(disp_gt.size() == disp_pred.size());

    cv::Mat valid_mask = (disp_gt > 0) & (disp_gt < 192);

    cv::Mat error;
    cv::absdiff(disp_pred, disp_gt, error);

    cv::Scalar mean_error = cv::mean(error, valid_mask);

    return mean_error[0];
}

cv::Mat gen_error_colormap() {
    float data[10][5] = {
        {0.0f / 3.0f,    0.1875f / 3.0f,  49,  54, 149},
        {0.1875f / 3.0f, 0.375f / 3.0f,   69, 117, 180},
        {0.375f / 3.0f,  0.75f / 3.0f,  116, 173, 209},
        {0.75f / 3.0f,   1.5f / 3.0f,  171, 217, 233},
        {1.5f / 3.0f,    3.0f / 3.0f,  224, 243, 248},
        {3.0f / 3.0f,    6.0f / 3.0f,  254, 224, 144},
        {6.0f / 3.0f,    12.0f / 3.0f, 253, 174,  97},
        {12.0f / 3.0f,   24.0f / 3.0f, 244, 109,  67},
        {24.0f / 3.0f,   48.0f / 3.0f, 215,  48,  39},
        {48.0f / 3.0f,   std::numeric_limits<float>::infinity(), 165, 0, 38}
    };

    cv::Mat cols(10, 5, CV_32F);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 5; ++j) {
            float val = data[i][j];
            if (j >= 2) val /= 255.0f;
            cols.at<float>(i, j) = val;
        }
    }
    return cols;
}

cv::Mat vis(const cv::Mat& D_est, const cv::Mat& D_gt) {
    CV_Assert(D_est.size() == D_gt.size() && D_est.type() == CV_32F && D_gt.type() == CV_32F);

    int H = D_gt.rows;
    int W = D_gt.cols;

    cv::Mat mask = (D_gt > 0);
    cv::Mat error = cv::abs(D_gt - D_est);

    error.setTo(0, ~mask);

    for (int y = 0; y < H; ++y) {
        float* err_ptr = error.ptr<float>(y);
        const float* gt_ptr = D_gt.ptr<float>(y);
        const uchar* mask_ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            if (mask_ptr[x]) {
                float abs_error_norm = err_ptr[x];
                float rel_error_norm = (err_ptr[x] / gt_ptr[x]);
                err_ptr[x] = std::min(abs_error_norm, rel_error_norm);
            }
        }
    }

    cv::Mat cols = gen_error_colormap();
    cv::Mat error_image(H, W, CV_32FC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < cols.rows; ++i) {
        float lower = cols.at<float>(i, 0);
        float upper = cols.at<float>(i, 1);
        cv::Vec3f color(cols.at<float>(i, 2), cols.at<float>(i, 3), cols.at<float>(i, 4));

        for (int y = 0; y < H; ++y) {
            float* err_ptr = error.ptr<float>(y);
            cv::Vec3f* out_ptr = error_image.ptr<cv::Vec3f>(y);
            const uchar* mask_ptr = mask.ptr<uchar>(y);
            for (int x = 0; x < W; ++x) {
                if (mask_ptr[x]) {
                    if (err_ptr[x] >= lower && err_ptr[x] < upper) {
                        out_ptr[x] = color;
                    }
                }
            }
        }
    }

    for (int y = 0; y < H; ++y) {
        cv::Vec3f* out_ptr = error_image.ptr<cv::Vec3f>(y);
        const uchar* mask_ptr = mask.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            if (!mask_ptr[x]) {
                out_ptr[x] = cv::Vec3f(0,0,0);
            }
        }
    }

    return error_image;
}

void visualize_and_record_disparity(
    const cv::Mat& disparity,
    const cv::Mat& disp_filtered_16,
    const cv::Mat& conf_map,
    const cv::Mat& gt_mat,
    const cv::Mat& left_img,
    const cv::Mat& valid_mask,
    bool record_video,
    double elapsed_ms,
    double fx,
    double baseline,
    double th,
    cv::VideoWriter& video_writer
) {

    int center_x = disparity.cols / 2;
    int center_y = disparity.rows / 2;

    float disp_val = disparity.at<float>(center_y, center_x);

    std::string depth_text;
    if (disp_val > 0.0) {
        double depth = (fx * baseline) / disp_val;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << depth << " m";
        depth_text = oss.str();
    } else {
        depth_text = "N/A";
    }

    double max_val = 30000, min_val = 400;
    cv::Mat disp_norm, disp_color;

    disp_filtered_16.convertTo(disp_norm, CV_8UC1, -255.0 / (max_val - min_val), 255.0 * max_val / (max_val - min_val));
    cv::applyColorMap(disp_norm, disp_color, cv::COLORMAP_MAGMA);


    cv::Mat left_color;
    if (left_img.channels() == 1) {
        cv::cvtColor(left_img, left_color, cv::COLOR_GRAY2BGR);
    } else {
        left_color = left_img.clone();
    }

    if (left_color.size() != disp_color.size()) {
        cv::resize(left_color, left_color, disp_color.size());
    }

    cv::Mat conf_map_color, conf_norm ;

    conf_map.convertTo(conf_norm, CV_8UC1, 256.0);
    cv::cvtColor(conf_norm, conf_map_color, cv::COLOR_GRAY2BGR);

    cv::Mat gt_mat_color, gt_norm ;
    gt_mat.convertTo(gt_norm, CV_8UC1, 256.0);
    cv::cvtColor(gt_norm, gt_mat_color, cv::COLOR_GRAY2BGR);

    cv::Mat gt_map_float;
    gt_mat.convertTo(gt_map_float, CV_32FC1, 1.0f / 256.0f);
    gt_map_float.setTo(0, ~valid_mask);

    cv::Mat D1_map = vis(disparity, gt_map_float);
    double epe = computeEPE(gt_map_float, disparity);

    cv::Mat error_map_color, error_norm ;
    D1_map.convertTo(error_norm, CV_8UC3, 255.0);
    cv::cvtColor(error_norm, error_map_color, cv::COLOR_RGB2BGR);

    cv::circle(disp_color, cv::Point(center_x, center_y), 5, cv::Scalar(255, 0, 0), -1);
    cv::putText(disp_color, depth_text, cv::Point(center_x + 10, center_y - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << 1000.0 / elapsed_ms << " HZ";
    std::string text = oss.str();

    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 4;
    cv::Scalar text_color(0, 255, 0);
    int baseline_2 = 0;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline_2);
    cv::Point text_org(disp_color.cols - text_size.width - 10, text_size.height + 10);
    cv::putText(disp_color, text, text_org, font_face, font_scale, text_color, thickness);

    std::ostringstream oss_th;
    oss_th << std::fixed << std::setprecision(2) << "Confidence Threshold: "  << th;
    std::string text_th_str = oss_th.str();
    cv::Point text_th(10, text_size.height + 10);
    cv::putText(conf_map_color, text_th_str, text_th, font_face, font_scale, text_color, thickness);

    std::ostringstream oss_epe;
    oss_epe << std::fixed << std::setprecision(2) << "End Point Error (EPE) [px]: "  << epe;
    std::string text_epe_str = oss_epe.str();
    cv::Point text_epe(10, text_size.height + 10);
    cv::putText(error_map_color, text_epe_str, text_epe, font_face, font_scale, text_color, thickness);

    cv::Mat combined_disp;
    cv::vconcat(left_color, disp_color, combined_disp);

    cv::Mat combined_conf;
    cv::vconcat(error_map_color, conf_map_color, combined_conf);

    cv::Mat combined;
    cv::hconcat(combined_disp, combined_conf, combined);

    cv::Mat combined_resized;
    cv::resize(combined, combined_resized, cv::Size(), 0.62, 0.62, cv::INTER_AREA);

    cv::imshow("Left + Disparity", combined_resized);
    cv::waitKey(1);

    if (record_video && !video_writer.isOpened()) {
        std::string output_path = "disparity_output.avi";
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        int fps = 30;
        cv::Size frame_size(combined.cols, combined.rows);
        video_writer.open(output_path, fourcc, fps, frame_size);
    }

    if (record_video && video_writer.isOpened()) {
        cv::Mat output_frame;
        if (combined.channels() == 1) {
            cv::cvtColor(combined, output_frame, cv::COLOR_GRAY2BGR);
        } else {
            output_frame = combined;
        }
        video_writer.write(output_frame);
    }

}


float* preprocess_image(const cv::Mat& img, const int net_input_width, const int net_input_height, int& pad_right, int& pad_bottom) {

    int w = img.cols;
    int h = img.rows;
    int m = 32;

    int wi = (w / m + 1) * m;
    int hi = (h / m + 1) * m;
    pad_right = wi - w;
    pad_bottom = hi - h;

    cv::Mat img_rgb;
    cv::copyMakeBorder(img, img_rgb, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_rgb.convertTo(img_rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img_rgb, channels);

    float mean_vals[3] = {0.485f, 0.456f, 0.406f};
    float std_vals[3]  = {0.229f, 0.224f, 0.225f};

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean_vals[c]) / std_vals[c];
    }

    int size = 3 * img_rgb.rows * img_rgb.cols;
    float* chw = new float[size];

    int idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < img_rgb.rows; ++h) {
            for (int w = 0; w < img_rgb.cols; ++w) {
                chw[idx++] = channels[c].at<float>(h, w);
            }
        }
    }

    return chw;
}

nvinfer1::ICudaEngine* loadEngine(const std::string& engineFile) {
    std::ifstream engineFileStream(engineFile, std::ios::binary);
    if (!engineFileStream) {
        std::cerr << "Error opening engine file: " << engineFile << std::endl;
        return nullptr;
    }

    engineFileStream.seekg(0, std::ios::end);
    size_t size = engineFileStream.tellg();
    engineFileStream.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    engineFileStream.read(engineData.data(), size);
    engineFileStream.close();

    static Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    if (!runtime) {
        std::cerr << "Error creating TensorRT runtime" << std::endl;
        return nullptr;
    }

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    delete runtime;

    if (!engine) {
        std::cerr << "Error deserializing engine" << std::endl;
        return nullptr;
    }

    return engine;
}

bool initializeTensorRT(std::string model_path, int net_input_height_, int net_input_width_,
                        int& leftIndex_, int& rightIndex_, int& outputIndex_, int& confidenceIndex_,
                        size_t& inputSize_, size_t& outputSize_) {

    engine_ = loadEngine(model_path);
    if (!engine_) {
         std::cerr << "Error loading engine" << std::endl;
    }

    context_ = engine_->createExecutionContext();

    cudaStreamCreate(&stream_);

    inputSize_ = 1 * 3 * net_input_height_ * net_input_width_ * sizeof(float);
    outputSize_ = 1 * net_input_height_ * net_input_width_ * sizeof(float);

    std::vector<std::string> leftNames  = {"input1", "input_left", "left", "input_left:0", "input_1"};
    std::vector<std::string> rightNames = {"input2", "input_right", "right", "input_right:0", "input_2"};

    std::vector<std::string> outputNames = {"output1", "disp", "output_0", "output:0"};
    std::vector<std::string> confidenceNames = {"output2", "confidence", "output_0", "output:0"};

    leftIndex_ = -1;
    rightIndex_ = -1;
    outputIndex_ = -1;
    confidenceIndex_ = -1;

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);

        for (const auto& leftName : leftNames)
            if (strcmp(name, leftName.c_str()) == 0) leftIndex_ = i;

        for (const auto& rightName : rightNames)
            if (strcmp(name, rightName.c_str()) == 0) rightIndex_ = i;

        for (const auto& outputName : outputNames)
            if (strcmp(name, outputName.c_str()) == 0) outputIndex_ = i;

        for (const auto& confidenceName : confidenceNames)
            if (strcmp(name, confidenceName.c_str()) == 0) confidenceIndex_ = i;
    }

    nvinfer1::Dims4 inputDims = {1, 3, net_input_height_, net_input_width_};
    context_->setInputShape(engine_->getIOTensorName(leftIndex_), inputDims);
    context_->setInputShape(engine_->getIOTensorName(rightIndex_), inputDims);

    cudaMalloc(&buffers_[leftIndex_], inputSize_);
    cudaMalloc(&buffers_[rightIndex_], inputSize_);
    cudaMalloc(&buffers_[outputIndex_], outputSize_);
    cudaMalloc(&buffers_[confidenceIndex_], outputSize_);

    return true;
}


class KittiImagePublisher : public rclcpp::Node {
public:
    KittiImagePublisher() : Node("kitti_image_publisher"), current_index(0), threshold(0.5) {
        RCLCPP_INFO(this->get_logger(), "Image Publisher Node Started!");

        // Parameters
        kitti_path = this->declare_parameter<std::string>("kitti_path", "./10");
        model_path = this->declare_parameter<std::string>("model_path", "/tmp/StereoModelCof.plan");
        record_video = this->declare_parameter<bool>("record_video", true);
        net_input_width_ = this->declare_parameter<int>("net_input_width", 384);
        net_input_height_ = this->declare_parameter<int>("net_input_height", 1248);
        fx = this->declare_parameter<double>("fx", 707.0912);
        baseline = this->declare_parameter<double>("baseline", 0.536);
        max_disp = this->declare_parameter<double>("max_disp", 192);

        fps = 150;

        left_dir_ = kitti_path + "/image_2";
        right_dir_ = kitti_path + "/image_3";
        gt_dir_ = kitti_path + "/disp_occ_0";

        if (!fs::exists(left_dir_) || !fs::exists(right_dir_)) {
            RCLCPP_ERROR(this->get_logger(), "Invalid KITTI dataset path: %s", kitti_path.c_str());
            throw std::runtime_error("KITTI dataset directories not found!");
        }

        for (const auto& entry : fs::directory_iterator(left_dir_)) {
            std::string filename = entry.path().filename().string();
            if (filename.size() >= 7 && filename.compare(filename.size() - 7, 7, "_10.png") == 0) {
                left_images_.push_back(entry.path().string());
            }
        }

        for (const auto& entry : fs::directory_iterator(right_dir_)) {
            std::string filename = entry.path().filename().string();
            if (filename.size() >= 7 && filename.compare(filename.size() - 7, 7, "_10.png") == 0) {
                right_images_.push_back(entry.path().string());
            }
        }

        for (const auto& entry : fs::directory_iterator(gt_dir_)) {
            gt_images_.push_back(entry.path().string());
        }

        std::sort(left_images_.begin(), left_images_.end());
        std::sort(right_images_.begin(), right_images_.end());
        std::sort(gt_images_.begin(), gt_images_.end());

        if (left_images_.size() != right_images_.size()) {
            RCLCPP_ERROR(this->get_logger(), "Mismatch in number of images between left and right cameras.");
            throw std::runtime_error("Left and right image counts do not match!");
        }

        cv::namedWindow("Left + Disparity", cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("Threshold", "Left + Disparity", &thresholdslider, 100.0);

        left_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/cam0/image_raw", 10);
        right_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/cam1/image_raw", 10);
        disparity_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/disparity/image_raw", 10);

        timer = this->create_wall_timer(
            std::chrono::milliseconds(1000 / fps),
            std::bind(&KittiImagePublisher::publishImages, this)
        );

        if (!initializeTensorRT(model_path, net_input_height_, net_input_width_,
                                leftIndex_, rightIndex_, outputIndex_, confidenceIndex_, inputSize_, outputSize_)) {

            RCLCPP_ERROR(this->get_logger(), "TensorRT initialization failed!");
            rclcpp::shutdown();
        }
    }

    ~KittiImagePublisher() {
        if (video_writer.isOpened()) {
            video_writer.release();
            RCLCPP_INFO(this->get_logger(), "Video writer released.");
        }
    }

private:
    void publishImages() {

        if (current_index >= left_images_.size()) {
            current_index = 0;
        }

        threshold = static_cast<double>(thresholdslider) / 100.0;

        left_img = cv::imread(left_images_[current_index], cv::IMREAD_COLOR);
        right_img = cv::imread(right_images_[current_index], cv::IMREAD_COLOR);
        gt_img = cv::imread(gt_images_[current_index], cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

        int original_height = left_img.rows;
        int original_width = left_img.cols;

        if (left_img.empty() || right_img.empty()) {
            RCLCPP_WARN(this->get_logger(), "Failed to read images at index %d", current_index);
            return;
        }

        rclcpp::Time current_time = this->get_clock()->now();

        // Convert to ROS Image messages
        auto left_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", left_img).toImageMsg();
        auto right_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", right_img).toImageMsg();

        left_msg->header.stamp = current_time;
        right_msg->header.stamp = current_time;

        left_msg->header.frame_id = "left_camera";
        right_msg->header.frame_id = "right_camera";

        left_pub_->publish(*left_msg);
        right_pub_->publish(*right_msg);

        // Run stereo inference
        float* outputData = new float[1 * net_input_height_ * net_input_width_];
        float* confidenceData = new float[1 * net_input_height_ * net_input_width_];

        float* inputLeft = preprocess_image(left_img, net_input_width_, net_input_height_, pad_right, pad_bottom);
        float* inputRight = preprocess_image(right_img, net_input_width_, net_input_height_, pad_right, pad_bottom);

        auto start = high_resolution_clock::now();

        // Copy input data to device
        cudaMemcpyAsync(buffers_[leftIndex_], inputLeft, inputSize_, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers_[rightIndex_], inputRight, inputSize_, cudaMemcpyHostToDevice, stream_);

        context_->setTensorAddress(engine_->getIOTensorName(leftIndex_), buffers_[leftIndex_]);
        context_->setTensorAddress(engine_->getIOTensorName(rightIndex_), buffers_[rightIndex_]);
        context_->setTensorAddress(engine_->getIOTensorName(outputIndex_), buffers_[outputIndex_]);
        context_->setTensorAddress(engine_->getIOTensorName(confidenceIndex_), buffers_[confidenceIndex_]);

        // Run inference
        if (!context_->enqueueV3(stream_)) {
            std::cerr << "Inference failed\n";
        }

        cudaStreamSynchronize(stream_);

        auto end = high_resolution_clock::now();
        double elapsed_ms = duration<double, std::milli>(end - start).count();
        std::cout << "Elapsed time =: " << elapsed_ms << " ms" << std::endl;

        // Copy output back to host
        cudaMemcpyAsync(outputData, buffers_[outputIndex_], outputSize_, cudaMemcpyDeviceToHost, stream_);
        cv::Mat disp_mat(net_input_height_, net_input_width_, CV_32FC1, outputData);

        cudaMemcpyAsync(confidenceData, buffers_[confidenceIndex_], outputSize_, cudaMemcpyDeviceToHost, stream_);
        cv::Mat conf_mat(net_input_height_, net_input_width_, CV_32FC1, confidenceData);

        // Crop the disparity cv::Mat to remove padding
        if (pad_bottom > 0 || pad_right > 0) {
            disp_mat = disp_mat(cv::Rect(0, 0, original_width, original_height));
            conf_mat = conf_mat(cv::Rect(0, 0, original_width, original_height));
        }

        // 1. Spatial smoothing
        cv::medianBlur(disp_mat, disp_filtered, 5);

        // 2. Temporal smoothing
        //float alpha = 0.4; // for temporal refinement
        //static cv::Mat prev_disp;
        //if (prev_disp.empty()) prev_disp = disp_filtered.clone();
        //cv::addWeighted(disp_filtered, alpha, prev_disp, 1.0 - alpha, 0, disp_filtered);
        //prev_disp = disp_filtered.clone();

        conf_mask = conf_mat >= threshold;
        range_mask = (disp_filtered > 0) & (disp_filtered < max_disp);
        valid_mask = range_mask & conf_mask;

        disp_filtered.setTo(0, ~valid_mask);
        disp_filtered.convertTo(disp_filtered_16, CV_16UC1, 256.0);
        visualize_and_record_disparity(
            disp_filtered,
            disp_filtered_16,
            conf_mat,
            gt_img,
            left_img,
            valid_mask,
            record_video,
            elapsed_ms,
            fx,
            baseline,
            threshold,
            video_writer
        );

        std::cout << "Original Image Size: " << left_img.cols << " x " << left_img.rows << std::endl;

        delete[] inputLeft;
        delete[] inputRight;
        delete[] outputData;
        delete[] confidenceData;


        auto disp_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "16UC1", disp_filtered_16).toImageMsg();

        disp_msg->header.stamp = current_time;
        disp_msg->header.frame_id = "left_camera";
        disparity_pub_->publish(*disp_msg);
        current_index++;
    }

    cv::Mat conf_mask, range_mask, valid_mask, left_img, right_img, gt_img;
    std::string kitti_path, model_path, left_dir_, right_dir_, gt_dir_;
    std::vector<std::string> left_images_, right_images_, gt_images_;
    cv::Mat disp_filtered, disp_filtered_16;
    double threshold, fx, baseline, max_disp;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr left_pub_, right_pub_, disparity_pub_;
    rclcpp::TimerBase::SharedPtr timer;
    bool record_video;
    int net_input_width_, net_input_height_, current_index, fps, thresholdslider, pad_right, pad_bottom;
    int leftIndex_, rightIndex_, outputIndex_, confidenceIndex_;
    cv::VideoWriter video_writer;
    size_t inputSize_,  outputSize_;

};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<KittiImagePublisher>();
    rclcpp::spin(node);

    if (context_) delete context_;
    if (engine_) delete engine_;
    for (int i = 0; i < 3; ++i) if (buffers_[i]) cudaFree(buffers_[i]);
    rclcpp::shutdown();
    return 0;
}



