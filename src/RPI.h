#ifndef _RPI_H_
#define _RPI_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Eigen>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <cassert>
#include <fstream>

namespace RPI{

namespace LocalWarp{

enum{
    INVALID_PIXEL = 0,
    VALID_PIXEL = 1,
    SEAM_PIXEL = 2
};

enum{
    TOP_SIDE = 0,
    BOTTOM_SIDE = 1,
    LEFT_SIDE = 2,
    RIGHT_SIDE = 3,
    INVALID_SIDE = 4
};

void getMask(cv::Mat const & img, cv::Mat & output){
    cv::Mat ret = cv::Mat::ones(img.rows, img.cols, CV_8UC1);

    for(size_t i=0;i<img.rows;i++){
        for(size_t j=0;j<img.cols;j++){
            uint8_t r, g, b;
            b = img.at<cv::Vec3b>(i,j).val[0];
            g = img.at<cv::Vec3b>(i,j).val[1];
            r = img.at<cv::Vec3b>(i,j).val[2];
            if(g>=250&&b<=50&&r<=50) ret.at<uint8_t>(i,j) = INVALID_PIXEL;
        }
    }

    output = std::move(ret);
}

void getEnergy(cv::Mat const & img, cv::Mat & output, cv::Mat const & mask){
    cv::Mat dx, dy;
    cv::Mat img_gray;

    cv::cvtColor(img, img_gray, CV_BGR2GRAY);
    cv::Sobel(img_gray, dx, CV_16S, 1, 0, 3);
    cv::Sobel(img_gray, dy, CV_16S, 0, 1, 3);
    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);

    cv::Mat ret = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);

    for(size_t i=0;i<img.rows;i++){
        for(size_t j=0;j<img.cols;j++){
            ret.at<double>(i, j) = dx.at<uint8_t>(i,j)/2.0+dy.at<uint8_t>(i,j)/2.0;
            if(mask.at<uint8_t>(i,j)==INVALID_PIXEL) ret.at<double>(i, j) = 1e8;
            else if(mask.at<uint8_t>(i,j)==SEAM_PIXEL) ret.at<double>(i, j) = 1e5;
        }
    }

    output = std::move(ret);
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

void addSeam(cv::Mat& img, cv::Mat& mask, cv::Mat const & energy, cv::Mat& dis_delta,
             uint8_t seg_type, std::vector<cv::Point> const & sub_img_vertex){
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
    }
    if(seg_type==TOP_SIDE){
        for(auto const & p:seam){
            mask.at<uint8_t>(p) = SEAM_PIXEL;
            for(size_t y=p1.y;y<p.y;y++){
                img.at<cv::Vec3b>(y, p.x) = img.at<cv::Vec3b>(y+1, p.x);
                mask.at<uint8_t>(y, p.x) = mask.at<uint8_t>(y+1, p.x);
                dis_delta.at<cv::Vec2i>(y, p.x) = dis_delta.at<cv::Vec2i>(y+1, p.x) + cv::Vec2i(0, 1);
            }
            if(p.y<img.rows-1){
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
            mask.at<uint8_t>(p) = SEAM_PIXEL;
            for(size_t y=p2.y;y>p.y;y--){
                img.at<cv::Vec3b>(y, p.x) = img.at<cv::Vec3b>(y-1, p.x);
                mask.at<uint8_t>(y, p.x) = mask.at<uint8_t>(y-1, p.x);
                dis_delta.at<cv::Vec2i>(y, p.x) = dis_delta.at<cv::Vec2i>(y-1, p.x) + cv::Vec2i(0, -1);
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
            mask.at<uint8_t>(p) = SEAM_PIXEL;
            for(size_t x=p1.x;x<p.x;x++){
                img.at<cv::Vec3b>(p.y, x) = img.at<cv::Vec3b>(p.y, x+1);
                mask.at<uint8_t>(p.y, x) = mask.at<uint8_t>(p.y, x+1);
                dis_delta.at<cv::Vec2i>(p.y, x) = dis_delta.at<cv::Vec2i>(p.y, x+1) + cv::Vec2i(1, 0);
            }
            if(p.x<img.cols-1){
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
            mask.at<uint8_t>(p) = SEAM_PIXEL;
            for(size_t x=p2.x;x>p.x;x--){
                img.at<cv::Vec3b>(p.y, x) = img.at<cv::Vec3b>(p.y, x-1);
                mask.at<uint8_t>(p.y, x) = mask.at<uint8_t>(p.y, x-1);
                dis_delta.at<cv::Vec2i>(p.y, x) = dis_delta.at<cv::Vec2i>(p.y, x-1) + cv::Vec2i(-1, 0);
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

void localWarp(cv::Mat const & img, cv::Mat& output, cv::Mat& dis_delta){
    cv::Mat energy, mask;
    cv::Mat ret = img.clone();
    dis_delta = cv::Mat::zeros(img.rows, img.cols, CV_32SC2);

    getMask(ret, mask);
    while(true){
        getEnergy(ret, energy, mask);
        std::vector<cv::Point> bound_seg;
        uint8_t seg_type = getBoundarySegment(mask, bound_seg);
        if(seg_type==INVALID_SIDE || bound_seg.empty()) break;
        std::vector<cv::Point> sub_img_vertex;
        getSubImageVertex(img, seg_type, bound_seg, sub_img_vertex);
        addSeam(ret, mask, energy, dis_delta, seg_type, sub_img_vertex);
    }

    output = std::move(ret);
}

} // namespace LocalWarp

namespace GlobalWarp{

int const GRID_ROW_CNT = 10;
int const GRID_COL_CNT = 40;
//int const GRID_ROW_CNT = 3;
//int const GRID_COL_CNT = 3;

void getGrid(cv::Mat const & img, cv::Mat const & dis_delta, cv::Mat& output){
    cv::Mat ret = cv::Mat::zeros(GRID_ROW_CNT, GRID_COL_CNT, CV_32SC2);
    int w = img.cols, h = img.rows;
    int w_delta = (w+GRID_COL_CNT-2)/(GRID_COL_CNT-1);
    int h_delta = (h+GRID_ROW_CNT-2)/(GRID_ROW_CNT-1);

    for(int i=0;i<GRID_COL_CNT;i++){
        for(int j=0;j<GRID_ROW_CNT;j++){
            ret.at<cv::Vec2i>(j, i) = cv::Vec2i(std::min(w_delta*i, w-1), std::min(h_delta*j, h-1));
        }
    }

    for(int i=0;i<GRID_COL_CNT;i++){
        for(int j=0;j<GRID_ROW_CNT;j++){
            cv::Point p = ret.at<cv::Vec2i>(j, i);
            ret.at<cv::Vec2i>(j, i) += dis_delta.at<cv::Vec2i>(p);
        }
    }

    output = std::move(ret);
}

void drawGrid(cv::Mat& img, cv::Mat const & grid){
    for(int i=0;i<GRID_COL_CNT;i++){
        for(int j=0;j<GRID_ROW_CNT;j++){
            cv::Point p1 = grid.at<cv::Vec2i>(j, i);
            if(i<GRID_COL_CNT-1){
                cv::Point p2 = grid.at<cv::Vec2i>(j, i+1);
                cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
            if(j<GRID_ROW_CNT-1){
                cv::Point p2 = grid.at<cv::Vec2i>(j+1, i);
                cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
        }
    }
}

void getEsMat(cv::Mat const & grid, Eigen::SparseMatrix<double>& output, double lambda=1.0){
    size_t rows = grid.rows, cols = grid.cols;
    Eigen::SparseMatrix<double> ret((rows-1)*(cols-1)*8, rows*cols*2);

    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            Eigen::MatrixXd aq(8, 4);

            cv::Vec2i v = grid.at<cv::Vec2i>(i, j);
            aq(0, 0) = v[0]; aq(0, 1) = -v[1]; aq(0, 2) = 1.0; aq(0, 3) = 0.0;
            aq(1, 0) = v[1]; aq(1, 1) = v[0];  aq(1, 2) = 0.0; aq(1, 3) = 1.0;

            v = grid.at<cv::Vec2i>(i, j+1);
            aq(2, 0) = v[0]; aq(2, 1) = -v[1]; aq(2, 2) = 1.0; aq(2, 3) = 0.0;
            aq(3, 0) = v[1]; aq(3, 1) = v[0];  aq(3, 2) = 0.0; aq(3, 3) = 1.0;

            v = grid.at<cv::Vec2i>(i+1, j);
            aq(4, 0) = v[0]; aq(4, 1) = -v[1]; aq(4, 2) = 1.0; aq(4, 3) = 0.0;
            aq(5, 0) = v[1]; aq(5, 1) = v[0];  aq(5, 2) = 0.0; aq(5, 3) = 1.0;

            v = grid.at<cv::Vec2i>(i+1, j+1);
            aq(6, 0) = v[0]; aq(6, 1) = -v[1]; aq(6, 2) = 1.0; aq(6, 3) = 0.0;
            aq(7, 0) = v[1]; aq(7, 1) = v[0];  aq(7, 2) = 0.0; aq(7, 3) = 1.0;

            aq = (aq * (aq.transpose() * aq).inverse() * aq.transpose() - Eigen::MatrixXd::Identity(8,8));

            for(int k=0;k<8;k++){
                int x = (i*(cols-1)+j)*8+k;
                int y = (i*cols+j)*2;
                ret.insert(x, y) = aq(k, 0);
                ret.insert(x, y+1) = aq(k, 1);
                ret.insert(x, y+2) = aq(k, 2);
                ret.insert(x, y+3) = aq(k, 3);
                y = (((i+1)*cols)+j)*2;
                ret.insert(x, y) = aq(k, 4);
                ret.insert(x, y+1) = aq(k, 5);
                ret.insert(x, y+2) = aq(k, 6);
                ret.insert(x, y+3) = aq(k, 7);
            }
        }
    }

    ret.makeCompressed();
    ret = ret * lambda / (cols-1) / (rows-1);
    output = std::move(ret);
}

void getEbMat(cv::Mat const & img, cv::Mat const & grid, Eigen::SparseMatrix<double>& outputMat, Eigen::VectorXd& outputVec, double lambda=1e8){
    size_t rows = grid.rows, cols = grid.cols;
    size_t boundVertexCnt = 2*rows+2*cols-4;
    Eigen::SparseMatrix<double> retM(boundVertexCnt*2, rows*cols*2);
    Eigen::VectorXd retV(boundVertexCnt*2);
    size_t cnt = 0;

    for(size_t i=1;i<cols-1;i++){
        retM.insert(cnt, i*2) = 0;
        retV(cnt) = 0;
        cnt++;
        retM.insert(cnt, i*2+1) = 1;
        retV(cnt) = 0;
        cnt++;

        retM.insert(cnt, ((rows-1)*cols+i)*2) = 0;
        retV(cnt) = 0;
        cnt++;
        retM.insert(cnt, ((rows-1)*cols+i)*2+1) = 1;
        retV(cnt) = img.rows-1;
        cnt++;
    }

    for(size_t j=1;j<rows-1;j++){
        retM.insert(cnt, j*cols*2) = 1;
        retV(cnt) = 0;
        cnt++;
        retM.insert(cnt, j*cols*2+1) = 0;
        retV(cnt) = 0;
        cnt++;

        retM.insert(cnt, (j*cols+cols-1)*2) = 1;
        retV(cnt) = img.cols-1;
        cnt++;
        retM.insert(cnt, (j*cols+cols-1)*2+1) = 0;
        retV(cnt) = 0;
        cnt++;
    }

    retM.insert(cnt, 0*2) = 1;
    retV(cnt) = 0;
    cnt++;
    retM.insert(cnt, 0*2+1) = 1;
    retV(cnt) = 0;
    cnt++;

    retM.insert(cnt, (cols-1)*2) = 1;
    retV(cnt) = img.cols-1;
    cnt++;
    retM.insert(cnt, (cols-1)*2+1) = 1;
    retV(cnt) = 0;
    cnt++;

    retM.insert(cnt, (rows-1)*cols*2) = 1;
    retV(cnt) = 0;
    cnt++;
    retM.insert(cnt, (rows-1)*cols*2+1) = 1;
    retV(cnt) = img.rows-1;
    cnt++;

    retM.insert(cnt, ((rows-1)*cols+cols-1)*2) = 1;
    retV(cnt) = img.cols-1;
    cnt++;
    retM.insert(cnt, ((rows-1)*cols+cols-1)*2+1) = 1;
    retV(cnt) = img.rows-1;
    cnt++;

    retV = retV * lambda;
    outputVec = std::move(retV);
    retM = retM * lambda;
    outputMat = std::move(retM);
}

void vec2Grid(cv::Mat const & img, cv::Mat const & grid, Eigen::VectorXd const & V, cv::Mat& output){
    size_t rows = grid.rows, cols = grid.cols;
    cv::Mat ret = grid.clone();
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x = V((i*cols+j)*2);
            int y = V((i*cols+j)*2+1);
            x = std::min(std::max(0, x), img.cols-1);
            y = std::min(std::max(0, y), img.rows-1);
            ret.at<cv::Vec2i>(i, j) = cv::Vec2i(x, y);
        }
    }
    output = std::move(ret);
}

Eigen::SparseMatrix<double> vconcat(Eigen::SparseMatrix<double> const & lhs, Eigen::SparseMatrix<double> const & rhs){
    Eigen::SparseMatrix<double> ret(lhs.rows()+rhs.rows(), lhs.cols());

    for(size_t k=0;k<lhs.outerSize();k++){
        for(Eigen::SparseMatrix<double>::InnerIterator it(lhs, k);it;++it){
            ret.insert(it.row(), it.col()) = it.value();
        }
    }

    for(size_t k=0;k<rhs.outerSize();k++){
        for(Eigen::SparseMatrix<double>::InnerIterator it(rhs, k);it;++it){
            ret.insert(it.row()+lhs.rows(), it.col()) = it.value();
        }
    }
    ret.makeCompressed();
    return ret;
}

Eigen::VectorXd vconcat(Eigen::VectorXd const & lhs, Eigen::VectorXd const & rhs){
    Eigen::VectorXd ret(lhs.rows()+rhs.rows());
    ret << lhs, rhs;
    return ret;
}

void globalWarp(cv::Mat const & img_, cv::Mat const & grid_, cv::Mat& output, int iterCnt=10){
    cv::Mat ret = grid_.clone();
    cv::Mat grid = grid_.clone();

    Eigen::SparseMatrix<double> esM;
    getEsMat(grid, esM);
    Eigen::SparseMatrix<double> ebM;
    Eigen::VectorXd ebV;
    getEbMat(img_, grid, ebM, ebV);

    for(int T=0;T<iterCnt;T++){
        Eigen::VectorXd B1 = Eigen::VectorXd::Zero(esM.rows());
        Eigen::VectorXd B = vconcat(B1, ebV);

        Eigen::SparseMatrix<double> A = vconcat(esM, ebM);

        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
        solver.compute(A);
        Eigen::VectorXd V = solver.solve(B);
        vec2Grid(img_, grid, V, ret);
    }

    output = std::move(ret);
}

} // namespace GlobalWarp

namespace GL{

class Shader{
    // from https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/shader_s.h
public:
    unsigned int ID;
    Shader(const char* vertexPath, const char* fragmentPath){
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;

        vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        try{
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;

            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();

            vShaderFile.close();
            fShaderFile.close();

            vertexCode   = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        }
        catch (std::ifstream::failure& e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }
        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();

        unsigned int vertex, fragment;

        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");

        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    // activate the shader
    // ------------------------------------------------------------------------
    void use()
    {
        glUseProgram(ID);
    }
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const
    {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type)
    {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void glShow(cv::Mat const & img, cv::Mat const & gridInit, cv::Mat const & gridAfter){
    size_t width = img.cols, height = img.rows;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "Final result", NULL, NULL);
    if (window == NULL){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        std::terminate();
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to initialize GLAD" << std::endl;
        std::terminate();
    }

    Shader shader("./img.vs","./img.fs");
    size_t cols = gridInit.cols, rows = gridInit.rows;
    auto vertices = std::unique_ptr<float>(new float[cols*rows*5]);

    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x = gridAfter.at<cv::Vec2i>(i, j)[0], y = gridAfter.at<cv::Vec2i>(i, j)[1];
            vertices.get()[(i*cols+j)*5] = float(2.0*x/width-1);;
            vertices.get()[(i*cols+j)*5+1] = -float(2.0*y/height-1);
            vertices.get()[(i*cols+j)*5+2] = 0.f;

            x = gridInit.at<cv::Vec2i>(i, j)[0], y = gridInit.at<cv::Vec2i>(i, j)[1];
            vertices.get()[(i*cols+j)*5+3] = float(1.0*x/width);
            vertices.get()[(i*cols+j)*5+4] = float(1-1.0*y/height);
        }
    }

    auto indices = std::unique_ptr<unsigned int>(new unsigned int[(cols-1)*(rows-1)*6]);

    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            indices.get()[(i*(cols-1)+j)*6] = i*cols+j;
            indices.get()[(i*(cols-1)+j)*6+1] = i*cols+j+1;
            indices.get()[(i*(cols-1)+j)*6+2] = (i+1)*cols+j;
            indices.get()[(i*(cols-1)+j)*6+3] = i*cols+j+1;
            indices.get()[(i*(cols-1)+j)*6+4] = (i+1)*cols+j+1;
            indices.get()[(i*(cols-1)+j)*6+5] = (i+1)*cols+j;
        }
    }

    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*cols*rows*5, vertices.get(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(float)*(cols-1)*(rows-1)*6, indices.get(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint texture;
    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cv::Mat tempMat = img.clone();
    cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2RGB);
    cv::flip(tempMat, tempMat, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tempMat.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    while(!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, texture);

        shader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6*(rows-1)*(cols-1) , GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
}

} // namespace GL

void warpImage(cv::Mat const & img, cv::Mat& output){
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat dis_delta;
    cv::Mat img_out = img.clone();
    LocalWarp::localWarp(img, img_out, dis_delta);
    cv::imwrite("local_warp.jpg", img_out);

    cv::Mat gridInit;
    GlobalWarp::getGrid(img, dis_delta, gridInit);

    img_out = img.clone();
    GlobalWarp::drawGrid(img_out, gridInit);
    cv::imwrite("grid_init.jpg", img_out);

    cv::Mat gridAfter;
    GlobalWarp::globalWarp(img, gridInit, gridAfter, 1);

    img_out = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    GlobalWarp::drawGrid(img_out, gridAfter);
    cv::imwrite("grid_after.jpg", img_out);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << duration << "ms" << std::endl;
    GL::glShow(img, gridInit, gridAfter);

//    cv::Mat img_gray;
//    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
//    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
//    std::vector<cv::Vec4f> lines;
//    lsd->detect(img_gray, lines);
//    cv::Mat img_lines = img.clone();
//    img_lines = cv::Scalar(0,0,0);
//    lsd->drawSegments(img_lines, lines);
//    cv::imshow("img lines", img_lines);
//    cv::waitKey(0);

    output = std::move(img_out);
}

} // namespace RPI

#endif
