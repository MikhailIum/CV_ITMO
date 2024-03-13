#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

int main(){
    // loading the image
    std::string filename = "original.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // creating a binary mask
    cv::Mat mask;
    cv::threshold(image, mask, 100, 255, cv::THRESH_BINARY_INV);

    // fixing mask using closing
    cv::Mat fixed_mask;
    cv::morphologyEx(mask, fixed_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)), cv::Point(-1, -1), 2);
    cv::imwrite("mask.jpg", fixed_mask);

    cv::Mat B = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));

    // Erosion
    cv::Mat BW2;
    cv::morphologyEx(fixed_mask, BW2, cv::MORPH_ERODE, B, cv::Point(-1, -1), 13, cv::BORDER_CONSTANT, cv::Scalar(0));


    // Dilation
    cv::Mat D, C, S;
    cv::Mat T = cv::Mat::zeros(fixed_mask.rows, fixed_mask.cols, fixed_mask.type());
    int pix_num = fixed_mask.rows * fixed_mask.cols;
    while (cv::countNonZero(BW2) < pix_num){
        cv::morphologyEx(BW2, D, cv::MORPH_DILATE, B, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::morphologyEx(D, C, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        S = C - D;
        cv::bitwise_or(S, T, T);
        BW2 = D;
    }


    // Closing for borders
    cv::morphologyEx(T, T, cv::MORPH_CLOSE, B, cv::Point(-1, -1), 17, cv::BORDER_CONSTANT, cv::Scalar(255));

    // Remove borders from an image
    cv::bitwise_and(~T, fixed_mask, fixed_mask);

    // saving splitted mask
    cv::imwrite("splitted.jpg", fixed_mask);

    // TODO: Большие пальцы разделились. А вот верхние - нет. Вероятно, пересечение сликшом велико.

    // saving an image with borders
    cv::Mat borders;
    cv::morphologyEx(fixed_mask, borders, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4)));
    cv::bitwise_and(borders, ~fixed_mask, borders);
    cv::imwrite("borders.jpg", borders); 
    
    return 0;
}