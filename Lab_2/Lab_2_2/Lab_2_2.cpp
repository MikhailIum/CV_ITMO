#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


// barrel distortion 
cv::Mat barrel_distortion(cv::Mat image){
    std::vector<float> t_x, t_y;

    for (int i = 0; i < image.cols; ++i){
        t_x.push_back(float(i));
    }

    for (int i = 0; i < image.rows; ++i){
        t_y.push_back(float(i));
    }

    cv::Mat x_i, y_i;
    cv::repeat(cv::Mat(t_x).reshape(1, 1), image.rows, 1, x_i);
    cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, image.cols, y_i);

    double x_mid = image.cols / 2.0;
    double y_mid = image.rows / 2.0;

    x_i = (x_i - x_mid) / x_mid;
    y_i = (y_i - y_mid) / y_mid;

    cv::Mat r, theta;
    cv::cartToPolar(x_i, y_i, r, theta);
    double F3(0.05), F5(0.01);
    cv::Mat r3, r5;
    pow(r, 3, r3);
    pow(r, 5, r5);

    r += r3 * F3;
    r += r5 * F5;

    cv::Mat u, v;
    cv::polarToCart(r, theta, u, v);
    u = u * x_mid + x_mid;
    v = v * y_mid + y_mid;

    cv::remap(image, image, u, v, cv::INTER_LINEAR);

    return image;
}

// pincushion distortion
cv::Mat pincushion_distortion(cv::Mat image){
    std::vector<float> t_x, t_y;

    for (int i = 0; i < image.cols; ++i){
        t_x.push_back(float(i));
    }

    for (int i = 0; i < image.rows; ++i){
        t_y.push_back(float(i));
    }

    cv::Mat x_i, y_i;
    cv::repeat(cv::Mat(t_x).reshape(1, 1), image.rows, 1, x_i);
    cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, image.cols, y_i);

    double x_mid = image.cols / 2.0;
    double y_mid = image.rows / 2.0;

    x_i = (x_i - x_mid) / x_mid;
    y_i = (y_i - y_mid) / y_mid;

    cv::Mat r, theta;
    cv::cartToPolar(x_i, y_i, r, theta);
    double F3(-0.2);
    cv::Mat r3;
    pow(r, 3, r3);

    r += r3 * F3;

    cv::Mat u, v;
    cv::polarToCart(r, theta, u, v);
    u = u * x_mid + x_mid;
    v = v * y_mid + y_mid;

    cv::remap(image, image, u, v, cv::INTER_LINEAR);

    return image;
}




int main(){

    // loading the image with barrel distortion
    std::string filename = "eiffel_barrel.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    // saving fixed image (barrel distortion)
    cv::imwrite("fixed_barrel.jpg", pincushion_distortion(image));

    // loading the image with pincushion distortion
    filename = "house_pincusion.jpg";
    image = cv::imread(filename, cv::IMREAD_COLOR);

    // saving fixed image (pincushion distortion)
    cv::imwrite("fixed_pincusion.jpg", barrel_distortion(image));

    return 0;
}