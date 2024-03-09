#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


// finding edges with Roberts filter 
cv::Mat Roberts_filter(cv::Mat image){
    int image_depth = image.depth();
    if (image_depth == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);

    cv::Mat G_x = (cv::Mat_<double>(2, 2) <<
                                    1, -1, 
                                    0, 0);

    cv::Mat G_y = (cv::Mat_<double>(2, 2) <<
                                    1, 0, 
                                    -1, 0);

    cv::Mat image_x, image_y;
    cv::filter2D(image, image_x, -1, G_x);
    cv::filter2D(image, image_y, -1, G_y);
    cv::magnitude(image_x, image_y, image);
    
    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);

    return image;
}

// finding edges with Prewitt filter 
cv::Mat Prewitt_filter(cv::Mat image){
    int image_depth = image.depth();
    if (image_depth == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);

    cv::Mat G_x = (cv::Mat_<double>(3, 3) <<
                                    -1, 0, 1, 
                                    -1, 0, 1,
                                    -1, 0, 1);

    cv::Mat G_y = (cv::Mat_<double>(3, 3) <<
                                    -1, -1, -1, 
                                    0, 0, 0,
                                    1, 1, 1);

    cv::Mat image_x, image_y;
    cv::filter2D(image, image_x, -1, G_x);
    cv::filter2D(image, image_y, -1, G_y);
    cv::magnitude(image_x, image_y, image);
    
    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);

    return image;
}

// finding edges with Laplace filter 
cv::Mat Laplace_filter(cv::Mat image){
    int image_depth = image.depth();
    if (image_depth == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);

    cv::Mat ker = (cv::Mat_<double>(3, 3) <<
                                    0, -1, 0, 
                                    -1, 4, -1,
                                    0, -1, 0);


    cv::filter2D(image, image, -1, ker);
    
    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);

    return image;
}


int main(){
    // loading the image
    cv::Mat image = cv::imread("original.jpg", cv::IMREAD_GRAYSCALE);

    // blurring using gaussian blur
    cv::Mat image_blur;
    cv::GaussianBlur(image, image_blur, cv::Size(3, 3), 0);

    // saving blurred image
    cv::imwrite("image_blurred.jpg", image_blur);

    cv::Mat working_image;

    // saving Roberts filtered image
    image_blur.copyTo(working_image);
    cv::imwrite("Roberts_filter.jpg", Roberts_filter(working_image));

    // saving Prewitt filtered image
    image_blur.copyTo(working_image);
    cv::imwrite("Prewitt_filter.jpg", Prewitt_filter(working_image));

    // saving Sobel filtered image
    cv::Sobel(image_blur, working_image, -1, 1, 1);
    cv::imwrite("Sobel_filter.jpg", working_image);

    // saving Laplace filtered image
    image_blur.copyTo(working_image);
    cv::imwrite("Laplace_filter.jpg", Laplace_filter(working_image));

    // saving Canny filtered image
    cv::Canny(image, working_image, 300, 600);
    cv::imwrite("Canny_filter.jpg", working_image);

    return 0;
}