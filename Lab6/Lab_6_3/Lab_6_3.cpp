#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

void bwareaopen(const cv::Mat &A, cv::Mat &C, int dim, int conn = 8){
    if (A.channels() != 1 && A.type() != CV_8U && A.type() != CV_32F)
        return;
    
    // Find all connected components
    cv::Mat labels, stats, centers;
    int num = cv::connectedComponentsWithStats(A, labels, stats, centers, conn);

    // Clone image
    C = A.clone();

    // Check size of all connected components
    std::vector<int> td;
    for (int i = 0; i < num; ++i){
        if (stats.at<int>(i, cv::CC_STAT_AREA) < dim){
            td.push_back(i);
        }
    }

    // Remove small areas
    if (td.size() > 0){
        if (A.type() == CV_8U){
            for (int i = 0; i < C.rows; ++i){
                for (int j = 0; j < C.cols; ++j){
                    for (int k = 0; k < td.size(); ++k){
                        if (labels.at<int>(i, j) == td[k]){
                            C.at<uchar>(i, j) = 0;
                            continue;
                        }
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < C.rows; ++i){
            for (int j = 0; j < C.cols; ++j){
                for (int k = 0; k < td.size(); ++k){
                    if (labels.at<int>(i, j) == td[k]){
                        C.at<float>(i, j) = 0;
                        continue;
                    }
                }
            }
        }
    }
}

int main(){
    // loading the image
    cv::Mat image, image_gray, image_bw;
    image = cv::imread("original.jpg", cv::IMREAD_COLOR);

    // converting to grayscale and creating a binary mask
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    cv::imwrite("gray.jpg", image_gray);
    cv::threshold(image_gray, image_bw, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    bwareaopen(image_bw, image_bw, 20, 4);

    // closing operation
    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(image_bw, image_bw, cv::MORPH_CLOSE, B);
    cv::imwrite("mask.jpg", image_bw);

    // calculating distance
    cv::Mat image_fg;
    double image_fg_min, image_fg_max;
    cv::distanceTransform(image_bw, image_fg, cv::DIST_L2, 5);

    // front markers
    cv::minMaxLoc(image_fg, &image_fg_min, &image_fg_max);
    cv::threshold(image_fg, image_fg, 0.6 * image_fg_max, 255, 0);
    image_fg.convertTo(image_fg, CV_8U, 255.0 / image_fg_max);
    cv::Mat markers;
    int num = cv::connectedComponents(image_fg, markers);
    cv::imwrite("forground_markers.jpg", image_fg);
    
    // back markers
    cv::Mat image_bg = cv::Mat::zeros(image_bw.rows, image_bw.cols, image_bw.type());
    cv::Mat markers_bg = markers.clone();
    cv::watershed(image, markers_bg);
    image_bg.setTo(cv::Scalar(255), markers_bg == -1);
    cv::imwrite("background_markers.jpg", image_bg);
    
    // unknown areas
    cv::Mat image_unk;
    cv::bitwise_not(image_bg, image_unk);
    cv::subtract(image_unk, image_fg, image_unk);
    cv::imwrite("unknown_areas.jpg", image_unk);

    // union all the markers
    markers += 1;
    markers.setTo(cv::Scalar(0), image_unk == 255);

    cv::watershed(image, markers);

    // adding markers and borders to the original image 
    cv::Mat markers_jet;
    markers.convertTo(markers_jet, CV_8U, 255.0 / (num + 1));
    cv::applyColorMap(markers_jet, markers_jet, cv::COLORMAP_JET);
    image.setTo(cv::Scalar(0, 255, 0), markers == -1);
    cv::imwrite("markers_jet.jpg", markers_jet);
    cv::imwrite("segmentation.jpg", image);

    return 0;
}