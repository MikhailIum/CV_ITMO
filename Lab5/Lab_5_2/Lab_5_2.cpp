#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// drawing hough circles
cv::Mat hough_circles(cv::Mat image_input, int threshold, int j, int R1, int R2, bool use_canny = 0, int thresh_canny_1 = 0, int thresh_canny_2 = 0){
    cv::Mat image;
    image_input.copyTo(image);
    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    if (use_canny){
        cv::Mat canny;
        cv::Canny(image_gray, canny, thresh_canny_1, thresh_canny_2);
        canny.copyTo(image_gray);
        cv::imwrite("canny_" + std::to_string(j) + ".jpg", canny);
    } 

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1, 20, 100, 30, R1, R2);


    for( size_t i = 0; i < circles.size(); ++i){
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        circle(image, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
        int radius = c[2];
        circle(image, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
    }

    return image;

}



int main(){
    std::string filename = "original_";
    std::string filename_ext = ".jpg";

    std::vector<int> threshold {400, 2, 20};
    std::vector<int> thresh_canny_1{250, 250, 230};
    std::vector<int> thresh_canny_2 {255, 255, 255};

    // saving images
    for (int i = 1; i < 4; ++i){
        cv::Mat image = cv::imread(filename + std::to_string(i) + filename_ext);
        // drawing circles with R = 103
        cv::imwrite("circles_R=103_" + std::to_string(i) + ".jpg", hough_circles(image, threshold[i-1], i, 100, 105));
        // drawing circles with R in range(100, 120)
        cv::imwrite("circles_R=100-120_" + std::to_string(i) + ".jpg", hough_circles(image, threshold[i-1], i, 100, 120));
        // drawing circles with differential operator with R = 103
        cv::imwrite("circles_canny_R=103_" + std::to_string(i) + ".jpg", 
            hough_circles(image, threshold[i-1], i, 100, 105, true, thresh_canny_1[i-1], thresh_canny_2[i-1]));
        // drawing circles with differential operator with R in range (100, 120)
        cv::imwrite("circles_canny_R=100-120_" + std::to_string(i) + ".jpg", 
            hough_circles(image, threshold[i-1], i, 100, 120, true, thresh_canny_1[i-1], thresh_canny_2[i-1]));
    
    }

    // TODO: А вот в этом случае, кажется, результат на обычном изображении лучше. Мб это можно как-то объяснить, подумай. Возможно, просто парметры для алгоритма Кэнни не очень и края определились плохо

    return 0;
}