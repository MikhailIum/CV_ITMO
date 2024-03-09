#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>

// image stitching
cv::Mat stitcher(cv::Mat left_image, cv::Mat right_image, int templ_size){
    cv::Mat templ = left_image(cv::Rect(left_image.cols - templ_size - 1, 0, templ_size, left_image.rows));

    cv::Mat res;
    cv::matchTemplate(right_image, templ, res, cv::TM_CCOEFF);

    double min_val, max_val;
    cv::Point2i min_loc, max_loc;
    minMaxLoc(res, &min_val, &max_val, &min_loc, &max_loc);
    

    cv::Mat result_img = cv::Mat::zeros(left_image.rows, left_image.cols + right_image.cols - max_loc.x - templ_size, left_image.type());

    left_image.copyTo(result_img(cv::Rect(0, 0, left_image.cols, left_image.rows)));


    right_image(cv::Rect(max_loc.x + templ_size, 0, right_image.cols - max_loc.x - templ_size, right_image.rows)).
            copyTo(result_img(cv::Rect(left_image.cols, 0, right_image.cols - max_loc.x - templ_size, right_image.rows)));

    return result_img;
}

// image stitching with the inline method
cv::Mat auto_stitcher(cv::Mat left_image, cv::Mat right_image){
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);

    std::vector<cv::Mat> images;
    images.push_back(right_image);
    images.push_back(left_image);

    cv::Mat result;
    cv::Stitcher::Status status = stitcher -> stitch(images, result);

    std::cout << status << std::endl;

    return result;
}



int main(){
    // loading the images
    cv::Mat right_image = cv::imread("Mishka.png", cv::IMREAD_COLOR);
    cv::Mat left_image = cv::imread("Vlada.png", cv::IMREAD_COLOR);


    // saving image stitching
    cv::imwrite("stitched.png", stitcher(left_image, right_image, 2));

    // saving image stitching with the inline method
    cv::imwrite("auto_stitched.png", auto_stitcher(left_image, right_image));
    // TODO: встроенный метод оказался глупым. Он не может найти область пересечения изображений и выкидывает статус 1 (not enough images).

    return 0;
}