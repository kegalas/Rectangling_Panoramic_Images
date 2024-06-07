#ifndef _RPI_H_
#define _RPI_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <thread>

namespace RPI{

enum{
    INVALID_PIXEL = 0,
    VALID_PIXEL = 1
};

enum{
    TOP_SIDE = 0,
    BOTTOM_SIDE = 1,
    LEFT_SIDE = 2,
    RIGHT_SIDE = 3,
    INVALID_SIDE = 4
};

void getMask(cv::Mat const & img, cv::Mat & output){
    output = cv::Mat::ones(img.rows, img.cols, CV_8UC1);

    for(size_t i=0;i<img.rows;i++){
        for(size_t j=0;j<img.cols;j++){
            uint8_t r, g, b;
            b = img.at<cv::Vec3b>(i,j).val[0];
            g = img.at<cv::Vec3b>(i,j).val[1];
            r = img.at<cv::Vec3b>(i,j).val[2];
            if(g>=250&&b<=5&&r<=5) output.at<uint8_t>(i,j) = INVALID_PIXEL;
        }
    }
}

void getEnergy(cv::Mat const & img, cv::Mat & output, cv::Mat const & mask){
    cv::Mat dx, dy;
    cv::Mat img_gray;

    cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    cv::Sobel(img_gray, dx, CV_16S, 1, 0, 3);
    cv::Sobel(img_gray, dy, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);

    output = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);

    for(size_t i=0;i<img.rows;i++){
        for(size_t j=0;j<img.cols;j++){
            output.at<double>(i, j) = dx.at<uint8_t>(i,j)/2.0+dy.at<uint8_t>(i,j)/2.0;
            if(mask.at<uint8_t>(i,j)==INVALID_PIXEL) output.at<double>(i, j) = 1e8;
        }
    }
}

uint8_t getBoundarySegment(cv::Mat const & mask, std::vector<cv::Point>& output){
    output.clear();
    uint8_t seg_type = INVALID_SIDE;
    size_t max_len = 0;
    size_t cols = mask.cols, rows = mask.rows;

    {
        size_t len = 0;
        std::vector<cv::Point> pts;
        for(size_t j=0;j<cols;j++){
            cv::Point p = cv::Point(j, 0);
            if(mask.at<uint8_t>(p)!=INVALID_PIXEL){
                if(len>max_len) {
                    output = std::move(pts);
                    max_len = len;
                    seg_type = TOP_SIDE;
                }
                pts.clear();
                len = 0;
            }
            else{
                if(pts.empty()){
                    pts.push_back(p);
                    pts.push_back(p);
                }
                else{
                    pts[1] = p;
                }
                len++;
            }
        }
        if(len>max_len) {
            output = std::move(pts);
            max_len = len;
            seg_type = TOP_SIDE;
        }
    }

    {
        size_t len = 0;
        std::vector<cv::Point> pts;
        for(size_t j=0;j<cols;j++){
            cv::Point p = cv::Point(j, rows-1);
            if(mask.at<uint8_t>(p)!=INVALID_PIXEL){
                if(len>max_len) {
                    output = std::move(pts);
                    max_len = len;
                    seg_type = BOTTOM_SIDE;
                }
                pts.clear();
                len = 0;
            }
            else{
                if(pts.empty()){
                    pts.push_back(p);
                    pts.push_back(p);
                }
                else{
                    pts[1] = p;
                }
                len++;
            }
        }
        if(len>max_len) {
            output = std::move(pts);
            max_len = len;
            seg_type = BOTTOM_SIDE;
        }
    }

    {
        size_t len = 0;
        std::vector<cv::Point> pts;
        for(size_t i=0;i<rows;i++){
            cv::Point p = cv::Point(0, i);
            if(mask.at<uint8_t>(p)!=INVALID_PIXEL){
                if(len>max_len) {
                    output = std::move(pts);
                    max_len = len;
                    seg_type = LEFT_SIDE;
                }
                pts.clear();
                len = 0;
            }
            else{
                if(pts.empty()){
                    pts.push_back(p);
                    pts.push_back(p);
                }
                else{
                    pts[1] = p;
                }
                len++;
            }
        }
        if(len>max_len) {
            output = std::move(pts);
            max_len = len;
            seg_type = LEFT_SIDE;
        }
    }

    {
        size_t len = 0;
        std::vector<cv::Point> pts;
        for(size_t i=0;i<rows;i++){
            cv::Point p = cv::Point(cols-1, i);
            if(mask.at<uint8_t>(p)!=INVALID_PIXEL){
                if(len>max_len) {
                    output = std::move(pts);
                    max_len = len;
                    seg_type = RIGHT_SIDE;
                }
                pts.clear();
                len = 0;
            }
            else{
                if(pts.empty()){
                    pts.push_back(p);
                    pts.push_back(p);
                }
                else{
                    pts[1] = p;
                }
                len++;
            }
        }
        if(len>max_len) {
            output = std::move(pts);
            max_len = len;
            seg_type = RIGHT_SIDE;
        }
    }

    return seg_type;
}

void getSubImageVertex(cv::Mat const & img, uint8_t seg_type,
                       std::vector<cv::Point> const & bound_seg,
                       std::vector<cv::Point>& output){
    output.clear();
    if(bound_seg.size()!=2||seg_type==INVALID_SIDE) return;

    size_t w = img.cols, h = img.rows;
    if(seg_type==TOP_SIDE){
        cv::Point p1 = bound_seg[0];
        cv::Point p2 = bound_seg[1];
        if(p1.x>p2.x) std::swap(p1,p2);
        p2.y = h-1;
        output.push_back(p1);
        output.push_back(p2);
    }
    else if(seg_type==BOTTOM_SIDE){
        cv::Point p1 = bound_seg[0];
        cv::Point p2 = bound_seg[1];
        if(p1.x>p2.x) std::swap(p1,p2);
        p1.y = 0;
        output.push_back(p1);
        output.push_back(p2);
    }
    else if(seg_type==LEFT_SIDE){
        cv::Point p1 = bound_seg[0];
        cv::Point p2 = bound_seg[1];
        if(p1.y>p2.y) std::swap(p1,p2);
        p2.x = w-1;
        output.push_back(p1);
        output.push_back(p2);
    }
    else if(seg_type==RIGHT_SIDE){
        cv::Point p1 = bound_seg[0];
        cv::Point p2 = bound_seg[1];
        if(p1.y>p2.y) std::swap(p1,p2);
        p1.x = 0;
        output.push_back(p1);
        output.push_back(p2);
    }
}

void addSeam(cv::Mat& img, cv::Mat const & energy, uint8_t seg_type,
             std::vector<cv::Point> const & sub_img_vertex){
    cv::Mat dp = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    std::vector<cv::Point> seam;

    cv::Point const & p1 = sub_img_vertex[0];
    cv::Point const & p2 = sub_img_vertex[1];

    auto isValid = [&p1, &p2](cv::Point p){
        return p.x>=p1.x && p.y>=p1.y && p.x<=p2.x && p.y<=p2.y;
    };
    if(seg_type==TOP_SIDE || seg_type==BOTTOM_SIDE){
        for(size_t x=p1.x;x<=p2.x;x++){
            for(size_t y=p1.y;y<=p2.y;y++){
                cv::Point p;
                double dis = 1e15;
                bool check = 0;
                dp.at<double>(y, x) = energy.at<double>(y, x);

                p.y = y, p.x = x-1;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;
                p.y = y-1, p.x = x-1;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;
                p.y = y+1, p.x = x-1;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;

                if(!check) dis = 0;
                dp.at<double>(y, x) += dis;
            }
        }

        cv::Point min_p(p2.x, p1.y);
        {
            double mindis = dp.at<double>(p1.y, p2.x);
            for(size_t y=p1.y;y<=p2.y;y++){
                double dis = dp.at<double>(y, p2.x);
                if(dis<mindis){
                    mindis = dis;
                    min_p.y = y;
                }
            }
        }
        while(min_p.x>=p1.x){
            seam.push_back(min_p);
            if(min_p.x==p1.x) break;

            double mindis = dp.at<double>(min_p.y, min_p.x-1);
            int miny = min_p.y;
            if(isValid(cv::Point(min_p.x-1, min_p.y-1))){
                double dis = dp.at<double>(min_p.y-1, min_p.x-1);
                if(dis<mindis){
                    miny = min_p.y-1;
                    mindis = dis;
                }
            }
            if(isValid(cv::Point(min_p.x-1, min_p.y+1))){
                double dis = dp.at<double>(min_p.y+1, min_p.x-1);
                if(dis<mindis){
                    miny = min_p.y+1;
                    mindis = dis;
                }
            }
            min_p.y = miny;
            min_p.x = min_p.x-1;
        }

//        for(auto const & p:seam){
//            img.at<cv::Vec3b>(p) = cv::Vec3b(255, 0, 255);
//        }
    }
    else if((seg_type==LEFT_SIDE || seg_type==RIGHT_SIDE)){
        for(size_t y=p1.y;y<=p2.y;y++){
            for(size_t x=p1.x;x<=p2.x;x++){
                cv::Point p;
                double dis = 1e15;
                bool check = 0;
                dp.at<double>(y, x) = energy.at<double>(y, x);

                p.y = y-1, p.x = x;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;
                p.y = y-1, p.x = x-1;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;
                p.y = y-1, p.x = x+1;
                if(isValid(p)) dis = std::min(dis, dp.at<double>(p)), check=1;

                if(!check) dis = 0;
                dp.at<double>(y, x) += dis;
            }
        }

        cv::Point min_p(p1.x, p2.y);
        {
            double mindis = dp.at<double>(p2.y, p1.x);
            for(size_t x=p1.x;x<=p2.x;x++){
                double dis = dp.at<double>(p2.y, x);
                if(dis<mindis){
                    mindis = dis;
                    min_p.x = x;
                }
            }
        }
        while(min_p.y>=p1.y){
            seam.push_back(min_p);
            if(min_p.y==p1.y) break;

            double mindis = dp.at<double>(min_p.y-1, min_p.x);
            int minx = min_p.x;
            if(isValid(cv::Point(min_p.x-1, min_p.y-1))){
                double dis = dp.at<double>(min_p.y-1, min_p.x-1);
                if(dis<mindis){
                    minx = min_p.x-1;
                    mindis = dis;
                }
            }
            if(isValid(cv::Point(min_p.x+1, min_p.y-1))){
                double dis = dp.at<double>(min_p.y-1, min_p.x+1);
                if(dis<mindis){
                    minx = min_p.x+1;
                    mindis = dis;
                }
            }
            min_p.y = min_p.y-1;
            min_p.x = minx;
        }

//        for(auto const & p:seam){
//            img.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
//        }
    }

    if(seg_type==TOP_SIDE){
        for(auto const & p:seam){
            for(size_t y=p1.y;y<p.y;y++){
                img.at<cv::Vec3b>(y, p.x) = img.at<cv::Vec3b>(y+1, p.x);
            }
            if(p.y<img.rows){
                cv::Vec3d a = img.at<cv::Vec3b>(p.y, p.x);
                cv::Vec3d b = img.at<cv::Vec3b>(p.y+1, p.x);
                b = (a+b)/2;
                b += cv::Vec3d(0.5, 0.5, 0.5);
                img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(cv::Vec3i(b));
            }
        }
    }
    else if(seg_type==BOTTOM_SIDE){
        for(auto const & p:seam){
            for(size_t y=p2.y;y>p.y;y--){
                img.at<cv::Vec3b>(y, p.x) = img.at<cv::Vec3b>(y-1, p.x);
            }
            if(p.y>0){
                cv::Vec3d a = img.at<cv::Vec3b>(p.y, p.x);
                cv::Vec3d b = img.at<cv::Vec3b>(p.y-1, p.x);
                b = (a+b)/2;
                b += cv::Vec3d(0.5, 0.5, 0.5);
                img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(cv::Vec3i(b));
            }
        }
    }
    else if(seg_type==LEFT_SIDE){
        for(auto const & p:seam){
            for(size_t x=p1.x;x<p.x;x++){
                img.at<cv::Vec3b>(p.y, x) = img.at<cv::Vec3b>(p.y, x+1);
            }
            if(p.x<img.cols){
                cv::Vec3d a = img.at<cv::Vec3b>(p.y, p.x);
                cv::Vec3d b = img.at<cv::Vec3b>(p.y, p.x+1);
                b = (a+b)/2;
                b += cv::Vec3d(0.5, 0.5, 0.5);
                img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(cv::Vec3i(b));
            }
        }
    }
    else if(seg_type==RIGHT_SIDE){
        for(auto const & p:seam){
            for(size_t x=p2.x;x>p.x;x--){
                img.at<cv::Vec3b>(p.y, x) = img.at<cv::Vec3b>(p.y, x-1);
            }
            if(p.x>0){
                cv::Vec3d a = img.at<cv::Vec3b>(p.y, p.x);
                cv::Vec3d b = img.at<cv::Vec3b>(p.y, p.x-1);
                b = (a+b)/2;
                b += cv::Vec3d(0.5, 0.5, 0.5);
                img.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(cv::Vec3i(b));
            }
        }
    }
}

void localWarp(cv::Mat const & img){
    cv::Mat energy, mask;
    cv::Mat img_out = img.clone();

    auto start = std::chrono::high_resolution_clock::now();


    while(true){
        getMask(img_out, mask);
        getEnergy(img_out, energy, mask);
        std::vector<cv::Point> bound_seg;
        uint8_t seg_type = getBoundarySegment(mask, bound_seg);
        if(seg_type==INVALID_SIDE || bound_seg.empty()) break;
        std::vector<cv::Point> sub_img_vertex;
        getSubImageVertex(img, seg_type, bound_seg, sub_img_vertex);
        addSeam(img_out, energy, seg_type, sub_img_vertex);

//        cv::rectangle(img_out, sub_img_vertex[0], sub_img_vertex[1], cv::Scalar(255, 0, 0), 2);
//        cv::imshow("image", img_out);
//        cv::waitKey(1);
    }
//    cv::waitKey(0);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << duration << "ms" << std::endl;
    cv::imwrite("result.jpg", img_out);
}

}; // namespace RPI

#endif
