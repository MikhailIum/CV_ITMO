#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// drawing hough lines
cv::Mat hough_lines(cv::Mat image_input, int threshold, int j, bool use_canny = 0, int thresh_canny_1 = 0, int thresh_canny_2 = 0){
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

    std::vector<cv::Vec4i> linesP; 
    HoughLinesP(image_gray, linesP, 1, CV_PI/180, threshold, 130, 50);
    
    std::vector<int> dist;
    for(size_t i = 0; i < linesP.size(); i++ ){
        cv::Vec4i l = linesP[i];
        cv::Point pt1 = cv::Point(l[0], l[1]);
        cv::Point pt2 = cv::Point(l[2], l[3]); 
        line(image, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
        dist.push_back(sqrt(pow(pt1.x - pt2.x, 2) + pow(pt1.y - pt2.y, 2)));
        cv::circle(image, pt1, 10, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::circle(image, pt2, 10, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    int max = 0;
    int min = INT_MAX;
    int cnt = 0;
    for (int i = 0; i < dist.size(); ++i){
        max = std::max(max, dist[i]);
        min = std::min(min, dist[i]);
        cnt++;
    }

    if (use_canny){
        std::cout << "Длина самого длинного отрезка " << j << "-й картинки с использованием алгоритма Кэнни: " << max << std::endl;
        std::cout << "Длина самого короткого отрезка " << j << "-й картинки с использованием алгоритма Кэнни: " << min << std::endl;
        std::cout << "Количество найденных прямых на " << j << "-й картинке с использованием алгоритма Кэнни: " << cnt << std::endl;
    } else {
        std::cout << "Длина самого длинного отрезка " << j << "-й картинки: " << max << std::endl;
        std::cout << "Длина самого короткого отрезка " << j << "-й картинки: " << min << std::endl;
        std::cout << "Количество найденный прямых на " << j << "-й картинке: " << cnt << std::endl;
    }    
    std::cout << std::endl;

    return image;

}



int main(){
    std::string filename = "original_";
    std::string filename_ext = ".jpg";

    std::vector<int> threshold {160, 20, 70};
    std::vector<int> thresh_canny_1{250, 250, 200};
    std::vector<int> thresh_canny_2 {255, 255, 255};

    // saving images
    for (int i = 1; i < 4; ++i){
        cv::Mat image = cv::imread(filename + std::to_string(i) + filename_ext);
        // drawing lines
        cv::imwrite("lines_" + std::to_string(i) + ".jpg", hough_lines(image, threshold[i-1], i));
        // drawing lines with differential operator
        cv::imwrite("lines_canny_" + std::to_string(i) + ".jpg", 
            hough_lines(image, threshold[i-1], i, true, thresh_canny_1[i-1], thresh_canny_2[i-1]));
    }

    // TODO: К 3 картинке подпиши, что робот теперь еще лучше понимает реальность
    // TODO: Напиши еще, что засовывать обычную картинку без использования дифференциального оператора бессмысленно

    return 0;
}