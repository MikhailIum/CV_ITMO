#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


// creating a binary mask for the road
cv::Mat make_mask(cv::Mat image_HSV){
    cv::Mat mask;
    // choosing white pixels by HSV-range
    cv::inRange(image_HSV, cv::Scalar(53, 0, 38), cv::Scalar(158, 37, 217), mask);
    cv::inRange(image_HSV, cv::Scalar(97, 0, 0), cv::Scalar(176, 255, 255), mask);

    // removing 'sky' pixels
    for (int x = 0; x < mask.cols; ++x){
        for (int y = 0; y < mask.cols / 3; ++y){
            mask.at<uchar>(y, x) = 0;
        }
    }

    return mask;
}

// fixing mask
cv::Mat fix_mask(cv::Mat mask){
    cv::Mat fixed_mask;
    cv::morphologyEx(mask, fixed_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 5)), cv::Point(-1, -1), 3);
    cv::morphologyEx(mask, fixed_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)), cv::Point(-1, -1), 2);

    return fixed_mask;
}

// adding mask to the original image
cv::Mat add_mask(cv::Mat mask, cv::Mat original){
    cv::Mat masked_road;
    original.copyTo(masked_road);
    for (int x = 0; x < masked_road.cols; ++x){
        for (int y = 0; y < masked_road.rows; ++y){
            if (mask.at<uchar>(y, x) == 255){
                masked_road.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0) * (70.0 / 255) + masked_road.at<cv::Vec3b>(y, x) * ((255 - 70.0) / 255);
            }
        }
    }

    return masked_road;
}

int main(){
    // loading the image
    std::string filename = "road.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    
    // TODO: Мы робототехники, поэтому давайте рассмотрим околоробототехничсекую задачу. Необходимо определить дорожное полотно на изображении.
    // Сперва нам необходимо получить бинарную маску дороги. 
    // Давайте сначала преобразуем наше изображение в HSV формат (так нам легче будет потом отделять определенный цвет)

    // converting to HSV
    cv::Mat image_HSV;
    cv::cvtColor(image, image_HSV, cv::COLOR_BGR2HSV);
    cv::imwrite("road_hsv.jpg", image_HSV);

    // TODO: итак, получим маску, выбирая необходимые диапазоны значений H, S, V. Также нужно убрать из маски небо, так как оно похоже на дорогу.
    // Для этого просто заменим всю верхнюю часть черными пикселями (мы же знаем, что дорога в нижней части).
    
    // creating a mask 
    cv::Mat mask = make_mask(image_HSV);
    cv::imwrite("mask.jpg", mask);

    // TODO: получилась маска, но она не очень хороша. Попробуем залатать дыры, используя морфологию.

    // fixing mask using morphology
    cv::Mat fixed = fix_mask(mask);
    cv::imwrite("fixed_mask.jpg", fixed);

    // TODO: Ну, стало слегка лучше. Это уже неплохой результат
    

    // saving original image with the mask and the fixed mask
    cv::imwrite("road_masked.jpg", add_mask(mask, image));
    cv::imwrite("road_masked_fixed.jpg", add_mask(fixed, image));

    // TODO: Ура, роботу теперь проще будет ехать по дорожке!

    return 0;
} 