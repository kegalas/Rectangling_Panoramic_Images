#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <chrono>
#include "RPI.hpp"

int main(int argc, char* argv[]) {
    if(argc!=2 && argc!=3){
        std::cout<<"usage: RPI.exe image_filename";
        return 0;
    }

    std::string filename = argv[1];
    cv::Mat img = cv::imread(filename, 1);
    if(img.empty()){
        std::cout<<"img invalid";
        return 0;
    }

    bool extra = 0;
    if(argc==3 && std::string(argv[2])=="-extra") extra = 1;

    cv::Mat output;
    RPI::warpImage(img, output, 1, extra);
//    cv::imshow("result", output);
//    cv::waitKey(0);
    cv::imwrite("result.jpg", output);

    return 0;
}
