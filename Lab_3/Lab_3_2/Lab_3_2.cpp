#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

// counterharmonic mean filter
cv::Mat counterharmonic_blur(cv::Mat image, double Q, int ker_size = 3){
    int image_depth = image.depth();
    if (image.depth() == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);
    int pad = ker_size / 2;
    for (int x = pad; x < image.cols - pad; ++x){
        for (int y = pad; y < image.rows - pad; ++y){
            double numerator_sum = 0;
            double denominator_sum = 0;
            for (int i = x-pad; i <= x+pad; ++i){
                for (int j = y-pad; j <= y+pad; ++j){
                    numerator_sum += pow(image.at<float>(j, i), Q+1);
                    if (image.at<float> (j, i) != 0)
                        denominator_sum += pow(image.at<float> (j, i), Q);

                }
            }
            if (denominator_sum == 0)
                image.at<float> (y, x) = 0;
            else {
                image.at<float> (y, x) = numerator_sum / denominator_sum;
            }
            
        }
    }

    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);
    
    
    return image;
}

// saving gaussian blur
void gauss(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image){
    cv::Mat working_image;

    cv::GaussianBlur(gaussian_noise_image, working_image, cv::Size(5, 5), 0);
    cv::imwrite("gaussian_blur_gaussian_noise.jpg", working_image);

    cv::GaussianBlur(impulse_noise_image, working_image, cv::Size(5, 5), 0);
    cv::imwrite("gaussian_blur_impulse_noise.jpg", working_image);

    cv::GaussianBlur(poisson_noise_image, working_image, cv::Size(5, 5), 0);
    cv::imwrite("gaussian_blur_poisson_noise.jpg", working_image);

    cv::GaussianBlur(speckle_noise_image, working_image, cv::Size(5, 5), 0);
    cv::imwrite("gaussian_blur_speckle_noise.jpg", working_image);

}

// saving counterharmonic blur
void counterharmonic(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image, int Q){
    cv::Mat working_image;

    gaussian_noise_image.copyTo(working_image);
    cv::imwrite("counterharmonic_blur_gaussian_noise_Q=" + std::to_string(Q) + ".jpg", counterharmonic_blur(working_image, Q));
    
    impulse_noise_image.copyTo(working_image);
    cv::imwrite("counterharmonic_blur_impulse_noise_Q=" + std::to_string(Q) + ".jpg", counterharmonic_blur(working_image, Q));

    poisson_noise_image.copyTo(working_image);
    cv::imwrite("counterharmonic_blur_poisson_noise_Q=" + std::to_string(Q) + ".jpg", counterharmonic_blur(working_image, Q));

    speckle_noise_image.copyTo(working_image);
    cv::imwrite("counterharmonic_blur_speckle_noise_Q=" + std::to_string(Q) + ".jpg", counterharmonic_blur(working_image, Q));
}


int main(){
    // loading the images
    cv::Mat gaussian_noise_image = cv::imread("gaussian_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat impulse_noise_image = cv::imread("impulse_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat poisson_noise_image = cv::imread("poisson_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat speckle_noise_image = cv::imread("speckle_noise.jpg", cv::IMREAD_GRAYSCALE);


    // saving gaussian blurred images
    gauss(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image);
    
    // saving counterharmonic blurred images with Q = 0
    counterharmonic(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 0);
    // TODO: При Q = 0 - арифметический усредняющий фильтр. Эффективен для слабо зашумленных изображений
    // Это можно заметить при фильтрации шума квантования

    // saving counterharmonic blurred images with Q = -1
    counterharmonic(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, -1);
    // TODO: При Q = -1 - гармонический усредняющий фильтр. Хорошо подавляет шумы типа "соль" и не работает с шумами типа "перец" (впрочем, как и все фильтры при Q < 0).
    // Это хорошо видно при фильтрации импульсного шума.

    // saving counterharmonic blurred images with Q = 1
    counterharmonic(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 0.05);
    // TODO: При Q > 0 - подавляются шумы типа "перец".
    // Это опять же хорошо видно при фильтрации импульсного шума.
    // Но для устранения импульсных помех лучше подойдет нелиинейная фильтрация. О ней дальше...


    return 0;
}