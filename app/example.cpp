
//string main_path= "/Users/arthur/Documents/work/AL_work/ACAP/AL_CV/AL_CV/";
// github.com/doleron/yolov5-opencv-cpp-python
// medium.com/mlearning-ai/detecting-objects-with-yolov5-opencv-python-and-c-c7cf13d1483c
#include <fstream> //to navagate and find paths
#include <ctime> // for time 
#include <stdlib.h>
#include <syslog.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/aruco.hpp>
#include "imgprovider.h" // the provider
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


// this the pass for the files that we stored .onnx and .txt using Docker
std::string main_path= "/usr/local/packages/opencv_app/util/";

std::vector<std::string> load_class_list()

// reading the text for classes
{
    std::vector<std::string> class_list;
    std::ifstream ifs(main_path+"classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

// load the yolov5s
int load_net(cv::dnn::Net &net)
{   //reding the net
    cv::dnn::Net result = cv::dnn::readNet(main_path+"yolov5s.onnx");
    net = result;
    return 0;
}

// format to be in good shape for the yolov5s
cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


//some configuration for YOLO
const float INPUT_WIDTH = 640.0;//1024 x 864
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{//our detections
    int class_id;
    float confidence;
    cv::Rect box;
};

//better configuration
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    // using blob to prepare the frame to be modeled
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);// inject a frame as input for model
    std::vector<cv::Mat> outputs;
    //foward the the frame in the model and give an get an output 
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    const int rows = 25200;
    std::vector<int> class_ids;//classe id
    std::vector<float> confidences; // the confidence 
    std::vector<cv::Rect> boxes;// ancor boxes

    for (int i = 0; i < rows; ++i) {
        
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            //
            float * classes_scores = data + 5;
            cv::Mat scores(1,(int) className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                //the box
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}



int main(int argc, char **argv)
{   openlog("opencv_app", LOG_PID|LOG_CONS, LOG_USER);
    syslog(LOG_INFO, "Running OpenCV example with VDO as video source");
    ImgProvider_t* provider = NULL;

    // The desired width and height of the BGR frame
    unsigned int width = 1024;
    unsigned int height = 576;

    // chooseStreamResolution gets the least resource intensive stream
    // that exceeds or equals the desired resolution specified above
    unsigned int streamWidth = 0;
    unsigned int streamHeight = 0;
    if (!chooseStreamResolution(width, height, &streamWidth,
                                &streamHeight)) {
        syslog(LOG_ERR, "%s: Failed choosing stream resolution", __func__);
        exit(1);
    }

    syslog(LOG_INFO, "Creating VDO image provider and creating stream %d x %d",
            streamWidth, streamHeight);
    provider = createImgProvider(streamWidth, streamHeight, 2, VDO_FORMAT_YUV);
    if (!provider) {
      syslog(LOG_ERR, "%s: Failed to create ImgProvider", __func__);
      exit(2);
    }

    syslog(LOG_INFO, "Start fetching video frames from VDO");
    if (!startFrameFetch(provider)) {
      syslog(LOG_ERR, "%s: Failed to fetch frames from VDO", __func__);
      exit(3);
    }

    
    std::vector<std::string> class_list = load_class_list(); // all the classes
    cv::Mat frame= cv::Mat(height * 3 / 2
                           , width, CV_8UC1);; //N2cv3MatE
    cv::Mat dst;


    cv::dnn::Net net;
    load_net(net);
    //aruco
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
    int count_frame=0;
    while (true)
    {   count_frame++;
        //aurco
        //std::vector <int> ids;
        //std::vector <std::vector<cv::Point2f> > corners;
        VdoBuffer* buf = getLastFrameBlocking(provider);
        if (!buf) {
          syslog(LOG_INFO, "No more frames available, exiting");
          exit(0);
        }//[1024 x 864]
        frame.data = static_cast<uint8_t*>(vdo_buffer_get_data(buf)); //N2cv3MatE to CV_8UC3
        cv::cvtColor(frame, dst, cv::COLOR_GRAY2RGB);
        //std::cout<<frame.size() << " " <<" "<< typeid(frame).name() << std::endl;
       // cv::aruco::detectMarkers(dst,dictionary,corners,ids);
        std::vector<Detection> output;// for coco
        //if (ids.size() > 0){//ides of aurco
         //   for (const int &i:ids){
                //res= "Id detected: " + std::to_string(i);
         //       syslog(LOG_INFO, "Id detected : %d",i);}
            
            // 80 coco
            std::time_t now = std::time(nullptr);
            
            detect(dst, net, output, class_list);//for coco
            std::time_t now_end = std::time(nullptr) - now;
            std::cout << "Current time: " << std::ctime(&now) << std::endl;
            int detections = output.size();
            //std::cout << "- name: " << std::endl;
            for (int i = 0; i < detections; ++i)
            {
                
                auto detection = output[i];
                auto box = detection.box;
                auto classId = detection.class_id;
                
              //  std::cout << "- name: " <<class_list[classId].c_str()<<". - x: "<<box.x<<". - y: "<<box.y<<". - width: "<<box.width<<". -height: "<< box.height<<". -confidence: "<<detection.confidence <<std::endl;
            }
            //syslog(LOG_INFO, "It has been detected : %d",detections);
        
            
        //}
        

        // Release the VDO frame buffer
        returnFrame(provider, buf);
        std::cout<< "we are at : "<<count_frame <<std::endl;
    }

    return EXIT_SUCCESS;
}
