#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


int main(){
    // loading the image
    std::string filename = "original.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    // splitting the image by channels
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);
    cv::Mat image_B = image_BGR[0];
    cv::Mat image_G = image_BGR[1];
    cv::Mat image_R = image_BGR[2];

    // TODO: Используем сегментацию по цвету кожи

    // skin color segmentation
    for (int x = 0; x < image.cols; ++x){
        for (int y = 0; y < image.rows; ++y){
            uchar r = image_R.at<uchar>(y, x);
            uchar g = image_G.at<uchar>(y, x);
            uchar b = image_B.at<uchar>(y, x);

            uchar max_rgb = std::max(r, g);
            max_rgb = std::max(max_rgb, b);
            uchar min_rgb = std::min(r, g);
            min_rgb = std::min(min_rgb, b);

            double R = (r * 1.0 / (r + g + b));
            double G = (g * 1.0 / (r + g + b));
            double B = (b * 1.0 / (r + g + b));

            if (r > 95 && g > 40 && b < 20 && max_rgb - min_rgb > 15 
            && std::abs(r - g) > 15 && r > g && r > b 
            || r > 220 && g > 210 && b > 170
            && std::abs(r - g) <= 15 && g > b && r > b 
            || r*1.0 / g > 1.185 && r*1.0*b/(r + g + b)/(r + g + b) > 0.107
            && r*1.0*g/(r + g + b)/(r + g + b) > 0.112){
                image.at<cv::Vec3b>(y, x) = 0.7 * image.at<cv::Vec3b>(y, x) +
                        cv::Vec3b(0, 255, 0) * 0.3;
            }
        }
    }

    // TODO: Получилось достаточно точно, но некоторые части лица не определились,
    // а галстук - наоборот

    // saving segmentation image
    cv::imwrite("segmentation.jpg", image);


    return 0;
}