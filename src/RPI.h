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
#include <array>
#include <optional>
#include <cmath>

namespace RPI{

namespace Geometry{
// 来自于我的ACM计算几何板子，这里只是用来求线段交点的。
typedef double db;
struct Point{
    db x,y;
    Point(double x_, double y_):x(x_),y(y_){}
};
typedef Point Vec;
struct Line{
    Point p; Vec v;
    Line(Point p_, Vec v_):p(p_.x, p_.y),v(v_.x, v_.y){}
};//点向式直线，不保证方向向量为单位向量
struct Seg{
    Point a,b;
    Seg(Point a_, Point b_):a(a_.x, a_.y),b(b_.x, b_.y){}
};//线段

db const EPS = 1e-9;
db const PI = acos(-1);

bool eq(db a, db b)  {return std::abs(a - b)< EPS;}//等于
bool ge(db a, db b)  {return a - b          > EPS;}//大于
bool le(db a, db b)  {return a - b          < -EPS;}//小于
bool geq(db a, db b) {return a - b          > -EPS;}//大于等于
bool leq(db a, db b) {return a - b          < EPS;}//小于等于
int sgn(db x) {
    if (std::abs(x) < EPS) return 0;
    if (x < 0) return -1;
    return 1;
} // 符号，等于零返回0，大于零返回1，小于零返回-1

Vec operator+(Vec a, Vec b){return {a.x+b.x, a.y+b.y};}
Vec operator-(Vec a, Vec b){return {a.x-b.x, a.y-b.y};}
Vec operator*(db k, Vec v){return {k*v.x, k*v.y};}
Vec operator*(Vec v, db k){return {v.x*k, v.y*k};}
db operator*(Vec a, Vec b){return a.x*b.x+a.y*b.y;}
db operator^(Vec a, Vec b){return a.x*b.y-a.y*b.x;}//叉积
db len2(Vec v){return v.x*v.x+v.y*v.y;}//长度平方
db len(Vec v){return std::sqrt(len2(v));}//向量长度

Line line(Point a, Point b){return {a,b-a};}//两点式直线
Line line(db k, db b){return {{0,b},{1,k}};}//斜截式直线y=kx+b
Line line(Point p, db k){return {p,{1,k}};}//点斜式直线
Line line(Seg l){return {l.a, l.b-l.a};}//线段所在直线

bool on(Point p, Line l){return eq((p.x-l.p.x)*l.v.y, (p.y-l.p.y)*l.v.x);}//点是否在直线上
bool on(Point p, Seg l){return eq(len(p-l.a)+len(p-l.b),len(l.a-l.b));}//点是否在线段上

bool operator==(Point a, Point b){return eq(a.x,b.x)&&eq(a.y,b.y);}//点重合
bool operator==(Line a, Line b){return on(a.p,b)&&on(a.p+a.v,b);}//直线重合
bool operator==(Seg a, Seg b){return ((a.a==b.a&&a.b==b.b)||(a.a==b.b&&a.b==b.a));}//线段（完全）重合

Point rotate(Point p, db rad){return {cos(rad)*p.x-sin(rad)*p.y,sin(rad)*p.x+cos(rad)*p.y};}

std::vector<Point> inter(Line a, Line b){
    //两直线的交点，没有交点返回空vector，否则返回一个大小为1的vector
    // 不能重叠
    db c = a.v^b.v;
    std::vector<Point> ret;
    if(eq(c,0.0)) return ret;
    Vec v = 1/c*Vec{a.p^(a.p+a.v), b.p^(b.p+b.v)};
    ret.push_back({v*Vec{-b.v.x, a.v.x},v*Vec{-b.v.y, a.v.y}});
    return ret;
}

std::vector<Point> inter(Seg s1, Seg s2) {
    // 两线段的交点，没有交点返回空vector，否则返回一个大小为1的vector
    // 这里特别规定，如果两条线段有重叠线段，会返回第一条线段的两个端点
    std::vector<Point> ret;
    using std::max;
    using std::min;
    bool check = true;
    check = check && geq(max(s1.a.x, s1.b.x), min(s2.a.x, s2.b.x));
    check = check && geq(max(s2.a.x, s2.b.x), min(s1.a.x, s1.b.x));
    check = check && geq(max(s1.a.y, s1.b.y), min(s2.a.y, s2.b.y));
    check = check && geq(max(s2.a.y, s2.b.y), min(s1.a.y, s1.b.y));
    if (!check) return ret;

    db pd1 = (s2.a - s1.a) ^ (s1.b - s1.a);
    db pd2 = (s2.b - s1.a) ^ (s1.b - s1.a);
    if (sgn(pd1 * pd2) == 1) return ret;
    std::swap(s1, s2);  // 双方都要跨立实验
    pd1 = (s2.a - s1.a) ^ (s1.b - s1.a);
    pd2 = (s2.b - s1.a) ^ (s1.b - s1.a);
    if (sgn(pd1 * pd2) == 1) return ret;

    if (sgn(pd1) == 0 && sgn(pd2) == 0) {
        ret.push_back(s2.a);
        ret.push_back(s2.a);
        return ret;
    }
    return inter(line(s2), line(s1));
}

int inpoly(std::vector<Point> const & poly, Point p){
    // 一个点是否在多边形内？
    // 0外部，1内部，2边上，3顶点上
    int n=poly.size();
    for(int i=0;i<n;i++){
        if(poly[i]==p) return 3;
    }
    for(int i=0;i<n;i++){
        if(on(p, Seg{poly[(i+1)%n],poly[i]})) return 2;
    }
    int cnt = 0;
    for(int i=0;i<n;i++){
        int j = (i+1)%n;
        int k = sgn((p-poly[j])^(poly[i]-poly[j]));
        int u = sgn(poly[i].y-p.y);
        int v = sgn(poly[j].y-p.y);
        if(k>0 && u<0 && v>=0) cnt++;
        if(k<0 && v<0 && u>=0) cnt--;
    }
    return cnt != 0;
}
}

namespace LocalWarp{

enum{
    INVALID_PIXEL = 0,
    VALID_PIXEL = 1,
    SEAM_PIXEL = 2 // seam上的像素，能量会被赋值为1e5，这是我添加的东西，而非原论文的
};

enum{
    TOP_SIDE = 0,
    BOTTOM_SIDE = 1,
    LEFT_SIDE = 2,
    RIGHT_SIDE = 3,
    INVALID_SIDE = 4
};

void getMask(cv::Mat const & img, cv::Mat & output){
    // 从原图中得到无效像素的mask，这里我的判断标准是是否接近绿色
    // 因为绿色比较少见
    // 显然有更好的方法去做这件事，但是我这里没有什么原始的全景照片信息，只能找论文中的图用PS加上绿色边界了
    // 而且加的不是很好，不是硬边缘背景
    cv::Mat ret = cv::Mat::ones(img.rows, img.cols, CV_8UC1);

    for(size_t i=0;i<img.rows;i++){
        for(size_t j=0;j<img.cols;j++){
            uint8_t r, g, b;
            b = img.at<cv::Vec3b>(i,j).val[0];
            g = img.at<cv::Vec3b>(i,j).val[1];
            r = img.at<cv::Vec3b>(i,j).val[2];
            if(g>=240&&b<=150&&r<=150) ret.at<uint8_t>(i,j) = INVALID_PIXEL;
        }
    }

    output = std::move(ret);
}

void getEnergy(cv::Mat const & img, cv::Mat & output, cv::Mat const & mask){
    // 获取图像像素的能量信息，用于求解seam。使用sobel算子计算
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
            else if(mask.at<uint8_t>(i,j)==SEAM_PIXEL) ret.at<double>(i, j) = 1e5; // 非原论文内容
        }
    }

    output = std::move(ret);
}

uint8_t getBoundarySegment(cv::Mat const & mask, std::vector<cv::Point>& output){
    // 即获取原文中提到的最长的边界段
    // 这里的实现比较暴力，纯纯写了四种情况
    // 想要优雅可以尝试旋转图片，用同一段代码检测

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
    // 通过bound_seg获取其对应的sub image的左上右下两个顶点的坐标

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
    // 动态规划求解seam
    // 代码很暴力，写了四种情况，可以尝试旋转图片来优雅地实现
    // dis_delta指的是像素距离原图对应像素的距离，在获取grid的时候有用

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
    // 输入图像，输出localwarp后的图片，还有每个像素相对于原图对应像素的移动距离
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
int const BIN_NUM = 50;

class Segment{
    // 存储线段，方便进行保线运算
public:
    cv::Vec2d a, b;
    double rad;
    int id;

    Segment(cv::Vec2d const & p1, cv::Vec2d const & p2):a(p1), b(p2){
        if(a[0]>b[0] || (a[0]==b[0]&&a[1]<b[1])) std::swap(a, b);

        cv::Vec2d v = b-a;
        rad = v[0]/std::sqrt(v[0]*v[0]+v[1]*v[1]);
        if(rad>1) rad = 1-1e-5;
        if(rad<-1) rad = -1+1e-5; // 防止出现NaN
        rad = acos(rad);
        id = -1;
    }

    Segment(double x1, double y1, double x2, double y2):Segment(cv::Vec2d(x1,y1), cv::Vec2d(x2,y2)){}
};

using LinesInQuadType = std::array<std::array<std::vector<Segment>, GRID_COL_CNT-1>, GRID_ROW_CNT-1>;
// 存储每个quad里面的线段

void getGrid(cv::Mat const & img, cv::Mat const & dis_delta, cv::Mat& output){
    // 获取初始的grid
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
    // 在图像img上绘制grid，主要是用于检查实现正确性
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

void drawLines(cv::Mat& img, LinesInQuadType const & lines){
    // 在图像上绘制各个quad里的线段，主要是用于检查实现正确性
    for(size_t i=0;i<GRID_ROW_CNT-1;i++){
        for(size_t j=0;j<GRID_COL_CNT-1;j++){
            for(auto const & l:lines[i][j]){
                cv::Point p1(l.a[0], l.a[1]);
                cv::Point p2(l.b[0], l.b[1]);
                cv::line(img, p1, p2, cv::Scalar(0, 255, 0));
            }
        }
    }
}

void drawLines(cv::Mat& img, std::vector<cv::Vec4f> const & lines){
    // 在图像上绘制初始的检测出来的线段，主要是用于检查实现正确性
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    lsd->drawSegments(img, lines);
}

void getLines(cv::Mat const & img, cv::Mat const & gridInit, LinesInQuadType& output){
    // 获取图像上的线段信息，切割后放入各个quad中
    double const mindis = 1;
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    lsd->detect(img_gray, lines); // opencv自带的图像检测算法

    auto quadCutSeg = [&mindis](std::array<cv::Vec2f, 4> const & quad, cv::Vec4f const & seg)->std::pair<bool, cv::Vec4f>{
        // 切割一条线段，获取属于quad内的部分
        // 返回的first是代表是否有在quad的部分，second则是这部分线段
        std::vector<Geometry::Point> poly;
        poly.emplace_back(quad[0][0], quad[0][1]);
        poly.emplace_back(quad[2][0], quad[2][1]);
        poly.emplace_back(quad[3][0], quad[3][1]);
        poly.emplace_back(quad[1][0], quad[1][1]);//按顶点逆时针给出的多边形

        Geometry::Point p1{seg[0], seg[1]};
        Geometry::Point p2{seg[2], seg[3]};
        if(Geometry::len(p1-p2)<mindis){ // 线段太小抛弃，否则之后有概率出现NaN
            return {false, seg};
        }
        if(Geometry::inpoly(poly, p1) && Geometry::inpoly(poly, p2)){ // 线段完全在quad内
            return {true, seg};
        }

        std::vector<Geometry::Point> interPoints;
        std::vector<Geometry::Seg> segs;
        segs.emplace_back(Geometry::Point(quad[0][0], quad[0][1]), Geometry::Point(quad[1][0], quad[1][1]));
        segs.emplace_back(Geometry::Point(quad[1][0], quad[1][1]), Geometry::Point(quad[3][0], quad[3][1]));
        segs.emplace_back(Geometry::Point(quad[2][0], quad[2][1]), Geometry::Point(quad[3][0], quad[3][1]));
        segs.emplace_back(Geometry::Point(quad[0][0], quad[0][1]), Geometry::Point(quad[2][0], quad[2][1]));
        Geometry::Seg seg2{Geometry::Point(seg[0], seg[1]), Geometry::Point(seg[2], seg[3])};

        for(auto const & seg1:segs){ // 暴力和quad的四条边求交点，可知最多有两个交点（除去四个顶点的情况）
            for(auto const &p:Geometry::inter(seg1, seg2)){
                interPoints.push_back(p);
            }
        }
        if(interPoints.size()==0) return {false, cv::Vec4f()}; // 完全不相交

        if(interPoints.size()==1){ // 线段的一个点在quad内，而一个在外，需要把在内的顶点放入
            if(Geometry::inpoly(poly, p1)) interPoints.push_back(p1);
            else interPoints.push_back(p2);
        }

        cv::Vec4f ret;

        if(Geometry::len(interPoints[0]-interPoints[1])<mindis){
            return {false, ret};
        }

        ret[0] = interPoints[0].x;
        ret[1] = interPoints[0].y;
        ret[2] = interPoints[1].x;
        ret[3] = interPoints[1].y;
        return {true, ret};
    };

    for(size_t i=0;i<GRID_ROW_CNT-1;i++){
        for(size_t j=0;j<GRID_COL_CNT-1;j++){
            output[i][j].clear();
            std::array<cv::Vec2f, 4> quad;
            quad[0] = cv::Vec2f(gridInit.at<cv::Vec2i>(i, j));
            quad[1] = cv::Vec2f(gridInit.at<cv::Vec2i>(i, j+1));
            quad[2] = cv::Vec2f(gridInit.at<cv::Vec2i>(i+1, j));
            quad[3] = cv::Vec2f(gridInit.at<cv::Vec2i>(i+1, j+1));

            for(auto const & line:lines){ // 暴力检测所有线段和所有quad的交集，应该会有更好的算法
                auto ans = quadCutSeg(quad, line);
                if(ans.first) output[i][j].emplace_back(ans.second[0], ans.second[1], ans.second[2], ans.second[3]);
            }
        }
    }
}

Eigen::MatrixXd getBilinearInterpolationMat(std::vector<cv::Vec2i> const & quad, Segment const & l){
    // 获取双线性插值矩阵，具体可见博客的数学详解
    Eigen::MatrixXd m1(4, 8);
    m1 << l.a[0], l.a[1], l.a[0]*l.a[1], 1, 0, 0, 0, 0,
        0, 0, 0, 0, l.a[0], l.a[1], l.a[0]*l.a[1], 1,
        l.b[0], l.b[1], l.b[0]*l.b[1], 1, 0, 0, 0, 0,
        0, 0, 0, 0, l.b[0], l.b[1], l.b[0]*l.b[1], 1;

    Eigen::MatrixXd m2 = Eigen::MatrixXd::Zero(8, 8);
    for(int t=0;t<4;t++){
        m2(t*2, 0) = quad[t][0];
        m2(t*2, 1) = quad[t][1];
        m2(t*2, 2) = quad[t][0]*quad[t][1];
        m2(t*2, 3) = 1;

        m2(t*2+1, 4) = quad[t][0];
        m2(t*2+1, 5) = quad[t][1];
        m2(t*2+1, 6) = quad[t][0]*quad[t][1];
        m2(t*2+1, 7) = 1;
    }

    Eigen::MatrixXd ret = m1*(m2.inverse());
    return ret;
}

void initBins(LinesInQuadType& lines, std::vector<int>& binsCntOutput, std::vector<double>& binsRadOutput, int M = BIN_NUM){
    // 初始化各个bin
    // 这里的theta_m实际上是个旋转的相对值，我一度以为是绝对值
    // 所以binsRadOutput初始化为0
    // 并不需要把初始的线段也全部离散化为50个角度
    binsCntOutput.clear();
    binsCntOutput.resize(M);
    binsRadOutput.clear();
    binsRadOutput.resize(M);

    for(int i=0;i<M;i++){
        binsCntOutput[i] = 0;
        binsRadOutput[i] = 0;
    }

    for(size_t i=0;i<GRID_ROW_CNT-1;i++){
        for(size_t j=0;j<GRID_COL_CNT-1;j++){
            for(auto& l:lines[i][j]){
                Geometry::Point p1(l.a[0], l.a[1]);
                Geometry::Point p2(l.b[0], l.b[1]);

                Geometry::Vec v = p2-p1;
                double const pi = Geometry::PI;

                double rad = v.x/std::sqrt(v.x*v.x+v.y*v.y);
                if(rad>1) rad = 1-1e-5;
                if(rad<-1) rad = -1+1e-5; // 防止NaN
                rad = acos(rad);

                int id = int((rad+pi/2.0)/pi*M);
                id = std::min(std::max(0, id), M-1);
                binsCntOutput[id]++;

                l.id = id;
            }
        }
    }
}

void updateBins(LinesInQuadType const & lines, cv::Mat const & gridInit, cv::Mat const & gridAfter,
                std::vector<int> const & binsCnt, std::vector<double>& binsRad, int M = BIN_NUM){
    // 根据第一部分更新的grid结果，来更新theta_m，见论文
    for(size_t i=0;i<M;i++){
        binsRad[i] = 0.0;
    }
    size_t rows = gridInit.rows, cols = gridInit.cols;

    std::vector<cv::Vec2i> quad;
    std::vector<cv::Vec2i> quadAfter;
    size_t cnt = 0;

    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            quad.clear();
            quad.push_back(gridInit.at<cv::Vec2i>(i, j));
            quad.push_back(gridInit.at<cv::Vec2i>(i, j+1));
            quad.push_back(gridInit.at<cv::Vec2i>(i+1, j));
            quad.push_back(gridInit.at<cv::Vec2i>(i+1, j+1));

            quadAfter.clear();
            quadAfter.push_back(gridAfter.at<cv::Vec2i>(i, j));
            quadAfter.push_back(gridAfter.at<cv::Vec2i>(i, j+1));
            quadAfter.push_back(gridAfter.at<cv::Vec2i>(i+1, j));
            quadAfter.push_back(gridAfter.at<cv::Vec2i>(i+1, j+1));

            for(auto const & l:lines[i][j]){
                Eigen::MatrixXd K = getBilinearInterpolationMat(quad, l);
                Eigen::MatrixXd V(8, 1);
                V<<quadAfter[0][0], quadAfter[0][1], quadAfter[1][0], quadAfter[1][1],
                    quadAfter[2][0], quadAfter[2][1], quadAfter[3][0], quadAfter[3][1];
                auto e = K*V;
                Segment l2(e(0, 0), e(1, 0), e(2, 0), e(3, 0));
                // l是原图中的线段，l2则是我们经过插值得到的目标线段
                binsRad[l.id] += l2.rad-l.rad;
            }
        }
    }

    for(size_t i=0;i<M;i++){
        if(binsCnt[i]) binsRad[i] /= binsCnt[i];
    }
}

void getEsMat(cv::Mat const & grid, Eigen::SparseMatrix<double>& output, double lambda=1.0){
    // 获取Es的Mat，具体可见博客详解
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

void getElMat(cv::Mat const & grid, Eigen::SparseMatrix<double>& output,
              LinesInQuadType const & lines, std::vector<int>& binsCnt,
              std::vector<double>& binsRad, double lambda=100.0){
    // 获取El的Mat，具体可见博客详解
    size_t lineCnt = 0;
    size_t rows = grid.rows, cols = grid.cols;

    for(auto const & bin: binsCnt){
        lineCnt += bin;
    }

    Eigen::SparseMatrix<double> ret(lineCnt*2, rows*cols*2);
    std::vector<cv::Vec2i> quad;
    size_t cnt = 0;

    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            quad.clear();
            quad.push_back(grid.at<cv::Vec2i>(i, j));
            quad.push_back(grid.at<cv::Vec2i>(i, j+1));
            quad.push_back(grid.at<cv::Vec2i>(i+1, j));
            quad.push_back(grid.at<cv::Vec2i>(i+1, j+1));

            for(auto const & l:lines[i][j]){
                Eigen::MatrixXd R(2,2);
                double theta = binsRad[l.id];
                R << cos(theta), -sin(theta),
                    sin(theta), cos(theta);
                Eigen::MatrixXd e(2,1);
                e<<l.b[0]-l.a[0], l.b[1]-l.a[1];
                Eigen::MatrixXd C = R*e*(e.transpose()*e).inverse()*e.transpose()*R.transpose()-Eigen::MatrixXd::Identity(2, 2);
                Eigen::MatrixXd K = getBilinearInterpolationMat(quad, l);
                Eigen::MatrixXd D(2, 4);
                D << -1, 0, 1, 0,
                    0, -1, 0, 1;
                Eigen::MatrixXd final = C * D * K;

                for(int t=0;t<2;t++){
                    int y = (i*cols+j)*2;
                    ret.insert(cnt, y) = final(t, 0);
                    ret.insert(cnt, y+1) = final(t, 1);
                    ret.insert(cnt, y+2) = final(t, 2);
                    ret.insert(cnt, y+3) = final(t, 3);

                    y = (((i+1)*cols)+j)*2;
                    ret.insert(cnt, y) = final(t, 4);
                    ret.insert(cnt, y+1) = final(t, 5);
                    ret.insert(cnt, y+2) = final(t, 6);
                    ret.insert(cnt, y+3) = final(t, 7);
                    cnt++;
                }
            }
        }
    }

    ret.makeCompressed();
    ret = ret * lambda / lineCnt;
    output = std::move(ret);
}

void getEbMat(cv::Size const & rectSz, cv::Mat const & grid, Eigen::SparseMatrix<double>& outputMat, Eigen::VectorXd& outputVec, double lambda=1e8){
    // 获取Eb的Mat，具体可见博客详解
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
        retV(cnt) = rectSz.height-1;
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
        retV(cnt) = rectSz.width-1;
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
    retV(cnt) = rectSz.width-1;
    cnt++;
    retM.insert(cnt, (cols-1)*2+1) = 1;
    retV(cnt) = 0;
    cnt++;

    retM.insert(cnt, (rows-1)*cols*2) = 1;
    retV(cnt) = 0;
    cnt++;
    retM.insert(cnt, (rows-1)*cols*2+1) = 1;
    retV(cnt) = rectSz.height-1;
    cnt++;

    retM.insert(cnt, ((rows-1)*cols+cols-1)*2) = 1;
    retV(cnt) = rectSz.width-1;
    cnt++;
    retM.insert(cnt, ((rows-1)*cols+cols-1)*2+1) = 1;
    retV(cnt) = rectSz.height-1;
    cnt++;

    retV = retV * lambda;
    outputVec = std::move(retV);
    retM = retM * lambda;
    outputMat = std::move(retM);
}

void vec2Grid(cv::Size const & rectSz, cv::Mat const & grid, Eigen::VectorXd const & V, cv::Mat& output){
    // 把我们求出的V转换成一个grid
    size_t rows = grid.rows, cols = grid.cols;
    cv::Mat ret = grid.clone();
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x = V((i*cols+j)*2);
            int y = V((i*cols+j)*2+1);
            x = std::min(std::max(0, x), rectSz.width-1);
            y = std::min(std::max(0, y), rectSz.height-1);
            ret.at<cv::Vec2i>(i, j) = cv::Vec2i(x, y);
        }
    }
    output = std::move(ret);
}

Eigen::SparseMatrix<double> vconcat(Eigen::SparseMatrix<double> const & lhs, Eigen::SparseMatrix<double> const & rhs){
    // 竖直方向拼接稀疏矩阵
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
    // 竖直方向拼接向量
    Eigen::VectorXd ret(lhs.rows()+rhs.rows());
    ret << lhs, rhs;
    return ret;
}

void resizeGrid(cv::Mat& grid, cv::Size const & srcSz, cv::Size const & dstSz){
    // 用于在缩小的图片上生成最终的grid后，将其缩放回原图大小
    for(size_t i=0;i<grid.rows;i++){
        for(size_t j=0;j<grid.cols;j++){
            cv::Vec2i v = grid.at<cv::Vec2i>(i,j);
            v[0] = int(1.0*v[0]*dstSz.width/srcSz.width+0.5);
            v[1] = int(1.0*v[1]*dstSz.height/srcSz.height+0.5);
            grid.at<cv::Vec2i>(i,j) = v;
        }
    }
}

void gridStretchingReduction(cv::Mat const & gridInit, cv::Mat const & gridAfter, cv::Size& rectSz){
    // 文章中提到的减小图片拉伸的办法
    double avgX=0.0, avgY = 0.0;
    for(size_t i=0;i<GRID_ROW_CNT-1;i++){
        for(size_t j=0;j<GRID_COL_CNT-1;j++){
            int xmin0=1e9, xmax0=0, ymin0=1e9, ymax0=0;
            cv::Vec2i v = gridInit.at<cv::Vec2i>(i,j);
            xmin0 = std::min(xmin0, v[0]); ymin0 = std::min(ymin0, v[1]);
            xmax0 = std::max(xmax0, v[0]); ymax0 = std::max(ymax0, v[1]);
            v = gridInit.at<cv::Vec2i>(i,j+1);
            xmin0 = std::min(xmin0, v[0]); ymin0 = std::min(ymin0, v[1]);
            xmax0 = std::max(xmax0, v[0]); ymax0 = std::max(ymax0, v[1]);
            v = gridInit.at<cv::Vec2i>(i+1,j+1);
            xmin0 = std::min(xmin0, v[0]); ymin0 = std::min(ymin0, v[1]);
            xmax0 = std::max(xmax0, v[0]); ymax0 = std::max(ymax0, v[1]);
            v = gridInit.at<cv::Vec2i>(i+1,j+1);
            xmin0 = std::min(xmin0, v[0]); ymin0 = std::min(ymin0, v[1]);
            xmax0 = std::max(xmax0, v[0]); ymax0 = std::max(ymax0, v[1]);

            int xmin1=1e9, xmax1=0, ymin1=1e9, ymax1=0;
            v = gridAfter.at<cv::Vec2i>(i,j);
            xmin1 = std::min(xmin1, v[0]); ymin1 = std::min(ymin1, v[1]);
            xmax1 = std::max(xmax1, v[0]); ymax1 = std::max(ymax1, v[1]);
            v = gridAfter.at<cv::Vec2i>(i,j+1);
            xmin1 = std::min(xmin1, v[0]); ymin1 = std::min(ymin1, v[1]);
            xmax1 = std::max(xmax1, v[0]); ymax1 = std::max(ymax1, v[1]);
            v = gridAfter.at<cv::Vec2i>(i+1,j+1);
            xmin1 = std::min(xmin1, v[0]); ymin1 = std::min(ymin1, v[1]);
            xmax1 = std::max(xmax1, v[0]); ymax1 = std::max(ymax1, v[1]);
            v = gridAfter.at<cv::Vec2i>(i+1,j+1);
            xmin1 = std::min(xmin1, v[0]); ymin1 = std::min(ymin1, v[1]);
            xmax1 = std::max(xmax1, v[0]); ymax1 = std::max(ymax1, v[1]);

            if(xmax0-xmin0>3) avgX += 1.0*(xmax1-xmin1)/(xmax0-xmin0);
            if(ymax0-ymin0>3) avgY += 1.0*(ymax1-ymin1)/(ymax0-ymin0);
        }
    }

    avgX /= (GRID_COL_CNT-1)*(GRID_ROW_CNT-1);
    avgY /= (GRID_COL_CNT-1)*(GRID_ROW_CNT-1);

    rectSz.width /= avgX;
    rectSz.height /= avgY;
}

void globalWarp(cv::Mat const & img_, cv::Size const & rectSz, cv::Mat const & grid_, cv::Mat& output, int iterCnt=10){
    // 输入图像，目标矩形框，初始grid
    // 输出目标grid
    cv::Mat ret = grid_.clone();
    cv::Mat const grid = grid_.clone();

    Eigen::SparseMatrix<double> esM;
    getEsMat(grid, esM, 2);

    GlobalWarp::LinesInQuadType lines;
    GlobalWarp::getLines(img_, grid, lines);

    std::vector<int> binsCnt;
    std::vector<double> binsRad;
    GlobalWarp::initBins(lines, binsCnt, binsRad);

    Eigen::SparseMatrix<double> ebM;
    Eigen::VectorXd ebV;
    getEbMat(rectSz, grid, ebM, ebV);

    for(int T=0;T<iterCnt;T++){
        Eigen::SparseMatrix<double> elM;
        getElMat(grid, elM, lines, binsCnt, binsRad);

        Eigen::VectorXd B1 = Eigen::VectorXd::Zero(esM.rows()+elM.rows());
        Eigen::VectorXd B = vconcat(B1, ebV);

        Eigen::SparseMatrix<double> A = vconcat(vconcat(esM, elM), ebM);

        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
        solver.compute(A);
        Eigen::VectorXd V = solver.solve(B);
        vec2Grid(rectSz, grid, V, ret);

        // 之上是更新V的部分，之下是更新theta_m的部分
        updateBins(lines, grid, ret, binsCnt, binsRad);
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

void glShow(cv::Mat const & img, cv::Mat const & gridInit, cv::Mat const & gridAfter, size_t width, size_t height){
    // 在opengl上进行纹理采样，显示图像，教程可见learnopengl网站
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
    auto vertices = std::make_unique<float[]>(cols*rows*5);

    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x = gridAfter.at<cv::Vec2i>(i, j)[0], y = gridAfter.at<cv::Vec2i>(i, j)[1];
            vertices[(i*cols+j)*5] = std::min(std::max(float(2.0*x/width-1), -1.f), 1.f);
            vertices[(i*cols+j)*5+1] = std::min(std::max(-float(2.0*y/height-1), -1.f), 1.f);
            vertices[(i*cols+j)*5+2] = 0.f;

            x = gridInit.at<cv::Vec2i>(i, j)[0], y = gridInit.at<cv::Vec2i>(i, j)[1];
            vertices[(i*cols+j)*5+3] = std::min(std::max(float(1.0*x/img.cols), 0.f), 1.f);
            vertices[(i*cols+j)*5+4] = std::min(std::max(float(1-1.0*y/img.rows), 0.f), 1.f);
        }
    }

    auto indices = std::make_unique<unsigned int[]>((cols-1)*(rows-1)*6);

    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            indices[(i*(cols-1)+j)*6] = i*cols+j;
            indices[(i*(cols-1)+j)*6+1] = i*cols+j+1;
            indices[(i*(cols-1)+j)*6+2] = (i+1)*cols+j;
            indices[(i*(cols-1)+j)*6+3] = i*cols+j+1;
            indices[(i*(cols-1)+j)*6+4] = (i+1)*cols+j+1;
            indices[(i*(cols-1)+j)*6+5] = (i+1)*cols+j;
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

//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cv::Mat tempMat = img.clone();
    cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2RGB);
    cv::flip(tempMat, tempMat, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tempMat.data);
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

void warpImage(cv::Mat const & img, cv::Mat& output, int gridDiv, bool extra=false){
    // 输入图像，输出warp的最终结果，只是我暂时还没有完成这个output，我还没有把图像从opengl里抠出来
    // gridDiv指，在1/gridDiv的比例上进行grid的计算
    // 虽然原文指出可以在小图上进行grid的计算，之后再放大回去。但我的实验结果表明，还是原图更好
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat dis_delta;
    cv::Mat img_out = img.clone();
    cv::Mat img_small;
    cv::Size sz;
    sz.width = img.cols/gridDiv;
    sz.height = img.rows/gridDiv;
    cv::resize(img, img_small, sz);

    LocalWarp::localWarp(img_small, img_out, dis_delta);
    if(extra) cv::imwrite("local_warp.jpg", img_out);

    cv::Mat gridInit;
    GlobalWarp::getGrid(img_small, dis_delta, gridInit);

    if(extra){
        img_out = img_small.clone();
        GlobalWarp::drawGrid(img_out, gridInit);
        cv::imwrite("grid_init.jpg", img_out);
    }

    cv::Mat gridAfter;
    cv::Size rectSz;
    rectSz.width = img_small.cols;
    rectSz.height = img_small.rows;
    GlobalWarp::globalWarp(img_small, rectSz, gridInit, gridAfter, 1);
    GlobalWarp::gridStretchingReduction(gridInit, gridAfter, rectSz);
    GlobalWarp::globalWarp(img_small, rectSz, gridInit, gridAfter, 3);

    if(extra){
        img_out = cv::Mat::zeros(rectSz.height, rectSz.width, CV_8UC3);
        GlobalWarp::drawGrid(img_out, gridAfter);
        cv::imwrite("grid_after.jpg", img_out);
    }

    GlobalWarp::resizeGrid(gridInit, sz, cv::Size(img.cols, img.rows));
    GlobalWarp::resizeGrid(gridAfter, sz, cv::Size(img.cols, img.rows));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << duration << "ms" << std::endl;

    cv::Mat mask;
    LocalWarp::getMask(img, mask);
    cv::Mat img_dilated = img.clone();

    for(int i=0;i<img_dilated.rows;i++){
        for(int j=0;j<img_dilated.cols;j++){
            if(mask.at<uint8_t>(i, j)==LocalWarp::INVALID_PIXEL)img_dilated.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        }
    }

    cv::Mat struct1 = getStructuringElement(0, cv::Size(3, 3));//矩形结构元素
    cv::dilate(img_dilated, img_dilated, struct1, cv::Point(-1, -1), 20);
    cv::Mat texture = img.clone();

    for(int i=0;i<img_dilated.rows;i++){
        for(int j=0;j<img_dilated.cols;j++){
            if(mask.at<uint8_t>(i, j)==LocalWarp::INVALID_PIXEL) texture.at<cv::Vec3b>(i, j) = img_dilated.at<cv::Vec3b>(i, j);
        }
    }

    GL::glShow(texture, gridInit, gridAfter, rectSz.width*gridDiv, rectSz.height*gridDiv);

    output = std::move(img_out);
}

} // namespace RPI

#endif
