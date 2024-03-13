#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// calculating entropy
float entropy(cv::Mat hist, cv::Mat image, int x, int y){
    float e = 0;
    for (int i = x-4; i < x+5; ++i){
        for (int j = y-4; j < y+5; ++j){
            int intensity = image.at<uchar>(j, i);
            e -= hist.at<float>(intensity) * log2(hist.at<float>(intensity));
        }
    }

    return e;
}


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

void print_params(cv::Mat image){
    int histSize = 256;
    float range [] = {0, 256};
    const float * histRange[] = {range};
    cv::Mat hist;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);    
    float sum = cv::sum(hist)[0];
    for (int i = 0; i < hist.rows; ++i){
        hist.at<float>(i) /= sum;
    }

    double m = 0;
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            int intensity = image.at<uchar>(y, x);
            m += intensity * hist.at<float>(intensity) / image.rows / image.cols;
        }
    }
    std::cout << "Среднее значение случайной величины: " << m << std::endl;

    
    double mu = 0;
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            int intensity = image.at<uchar>(y, x);
            mu += pow(intensity - m, 2) * hist.at<float>(intensity) / image.rows / image.cols;
        }
    }
    std::cout << "Дисперсия случайной величины: " << mu << std::endl;
    std::cout << "Стандартное отклонение: " << sqrt(mu) << std::endl;

    mu = 0;
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            int intensity = image.at<uchar>(y, x);
            mu += pow(intensity - m, 3) * hist.at<float>(intensity) / image.rows / image.cols;
        }
    }
    std::cout << "Характеристика симметрии гистограммы: " << mu << std::endl;

    double e = 0;
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            int intensity = image.at<uchar>(y, x);
            e -= hist.at<float>(intensity) * log2(hist.at<float>(intensity));
        }
    }
    std::cout << "Значение энтропии: " << e / image.rows / image.cols << std::endl;

}



int main(){
    // loading the image
    std::string filename = "original.jpg";
    cv::Mat image_rgb = cv::imread(filename, cv::IMREAD_COLOR);
    cv::Mat image;
    cv::cvtColor(image_rgb, image, cv::COLOR_BGR2GRAY);

    // saving grayscaled image
    cv::imwrite("image_grayscale.jpg", image);

    // histogram calculation 
    int histSize = 256;
    float range [] = {0, 256};
    const float * histRange[] = {range};
    cv::Mat hist;
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);    
    float sum = cv::sum(hist)[0];
    for (int i = 0; i < hist.rows; ++i){
        hist.at<float>(i) /= sum;
    }

    // calculating entropy matrix
    cv::Mat image_entropy(image.size(), image.type());
    for (int x = 4; x < image.cols - 4; ++x){
        for (int y = 4; y < image.rows - 4; ++y){
            image_entropy.at<uchar>(y, x) = entropy(hist, image, x, y);
        }
    }

    // normalization
    double min, max;
    cv::minMaxLoc(image_entropy, &min, &max);
    for (int x = 0; x < image_entropy.cols; ++x){
        for (int y = 0; y < image_entropy.rows; ++y){
            image_entropy.at<uchar>(y, x) *= 255 / (max - min);
        }
    }

    // binarization
    cv::GaussianBlur(image_entropy, image_entropy, cv::Size(17, 17), 19);
    cv::imwrite("image_for_binarization.jpg", image_entropy);
    cv::Mat mask;
    cv::threshold(image_entropy, mask, 100, 255, cv::THRESH_OTSU);
    cv::imwrite("mask_binary_otsu.jpg", ~mask);

    // fixing with morphology
    cv::Mat fixed_mask;
    bwareaopen(~mask, fixed_mask, 1000);
    cv::morphologyEx(fixed_mask, fixed_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)), cv::Point(-1, -1), 7);
    cv::morphologyEx(fixed_mask, fixed_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)), cv::Point(-1, -1), 7);
    cv::morphologyEx(fixed_mask, fixed_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)), cv::Point(-1, -1), 4);
    cv::morphologyEx(fixed_mask, fixed_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)), cv::Point(-1, -1), 4);
    cv::imwrite("mask_fixed.jpg", fixed_mask);


    // saving land and water separetly
    cv::Mat water(image_rgb.size(), image_rgb.type());
    for (int x = 0; x < image_rgb.cols; ++x){
        for (int y = 0; y < image_rgb.rows; ++y){
            if (fixed_mask.at<uchar>(y, x) == 0){
                water.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else {
                water.at<cv::Vec3b>(y, x) = image_rgb.at<cv::Vec3b>(y, x);
            }
        }
    }


    cv::Mat land(image_rgb.size(), image_rgb.type());
    fixed_mask = ~fixed_mask;
    for (int x = 0; x < image_rgb.cols; ++x){
        for (int y = 0; y < image_rgb.rows; ++y){
            if (fixed_mask.at<uchar>(y, x) == 0){
                land.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else {
                land.at<cv::Vec3b>(y, x) = image_rgb.at<cv::Vec3b>(y, x);
            }
        }
    }
    

    cv::imwrite("water.jpg", water);
    cv::imwrite("land.jpg", land);

    // TODO: Волны сильно помешали эксперименту, но результат всё равно не так плох

    cv::Mat water_gray;
    cv::bitwise_and(image, ~fixed_mask, water_gray);
    cv::Mat land_gray;
    cv::bitwise_and(image, fixed_mask, land_gray);

    std::cout << "Вода: " << std::endl;
    print_params(water_gray);
    std::cout << std::endl;
    std::cout << "Земля: " << std::endl;
    print_params(land_gray);

    // TODO: Не добавляй скрин значений, заполни табличку, как в методичке

    return 0;
}