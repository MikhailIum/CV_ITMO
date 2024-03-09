#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sciplot/sciplot.hpp>

void plotting_OX(cv::Mat proj){
    // setting for a plot
    sciplot::Plot2D plot;
    plot.xlabel("col");
    plot.ylabel("intensity");
    plot.legend().hide();
    plot.ytics().hide();
    plot.xtics().hide();

    
    // adding data
    sciplot::Vec x = sciplot::linspace(0, proj.rows, proj.rows);
    std::vector<int> y;
    for (int i = 0; i < proj.rows; ++i){
        y.push_back(proj.at<float>(i));
    }

    // drawing a plot
    plot.drawCurve(x, y);

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    canvas.save("result_OX.pdf");
}


void plotting_OY(cv::Mat proj){
    // setting for a plot
    sciplot::Plot2D plot;
    plot.xlabel("intensity");
    plot.ylabel("row");
    plot.legend().hide();
    plot.ytics().hide();
    plot.xtics().hide();

    
    // adding data
    sciplot::Vec x = sciplot::linspace(0, proj.rows, proj.rows);
    std::vector<int> y;
    for (int i = 0; i < proj.rows; ++i){
        y.push_back(proj.at<float>(i));
    }

    // drawing a plot
    plot.drawCurve(y, x);

    sciplot::Figure fig = {{plot}};
    sciplot::Canvas canvas = {{fig}};

    canvas.save("result_OY.pdf");
}

int main(){

    cv::Mat image = cv::imread("cv.png", cv::IMREAD_COLOR);

    // An array to store projection
    cv::Mat proj(image.rows, 1, CV_32F);
    
    for (int i = 0; i < image.rows; ++i){
        double sum = 0;
        for (int j = 0; j < image.cols; ++j){
            if (image.channels() == 1){
                sum += image.at<uchar>(i , j);
            } else {
                const cv::Vec3b & pix = image.at<cv::Vec3b>(i, j);
                sum += pix[0] + pix[1] + pix[2];
            }
        }
        proj.at<float>(i) = float(sum);
    }
    
    proj /= 256 * image.channels();
    
    
    plotting_OY(proj);

    cv::Mat proj_x(image.cols, 1, CV_32F);
    
    for (int i = 0; i < image.cols; ++i){
        double sum = 0;
        for (int j = 0; j < image.rows; ++j){
            if (image.channels() == 1){
                sum += image.at<uchar>(j, i);
            } else {
                const cv::Vec3b & pix = image.at<cv::Vec3b>(j, i);
                sum += pix[0] + pix[1] + pix[2];
            }
        }
        proj_x.at<float>(i) = float(sum);
    }

    proj_x /= 256 * image.channels();
    plotting_OX(proj_x);
    

    return 0;
}