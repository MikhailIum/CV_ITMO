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

    // threshold determination "by eye"
    cv::Mat mask; 
    cv::threshold(image, mask, 140, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("mask_binary.jpg", mask);
    // TODO: это подбор порога "на глаз"

    // finding minimum and maximum for a mean threshold 
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    std::cout << min_val << ' ' << max_val << std::endl;
    // min = 3, max = 255. Не очень репрезентативно для бинаризации. Просто есть и белые и черные пиксели, поэтому вряд ли будет результат лучше, чем ручками

    // threshold determination by mean value
    cv::threshold(image, mask, (max_val - min_val) / 2, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("mask_binary_mean.jpg", mask);
    // TODO: ну да, результат хуже

    // threshold determination by Otsu algorithm
    cv::threshold(image, mask, 100, 255, cv::THRESH_OTSU);
    cv::imwrite("mask_binary_otsu.jpg", ~mask);
    // TODO: а вот это уже ого-го. Мощный алгоитм даёт хорошие результаты

    // adaptive thresholding by mean
    cv::adaptiveThreshold(image, mask, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 12);
    cv::imwrite("mask_binary_adaptive_mean.jpg", mask);
    // TODO: удалось хорошо выделить границы

    // adaptive gaussian thresholding
    cv::adaptiveThreshold(image, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 13);
    cv::imwrite("mask_binary_adaptive_gaussian.jpg", mask);
    // TODO: Получили результат, похожий на предыдущий. Возможно, это можно как-то улучшить при правильном подборе параметров.

    return 0;
}