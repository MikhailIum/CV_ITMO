#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

// adding impulse noise
cv::Mat impulse_noise(cv::Mat image, double d){
    double salt_vs_pepper = 0.5;

    cv::Mat vals(image.size(), CV_32F);
    cv::randu(vals, cv::Scalar(0), cv::Scalar(1));
    
    if (image.depth() == CV_8U)
        image.setTo(cv::Scalar(255), vals < d * salt_vs_pepper);
    else 
        image.setTo(cv::Scalar(1), vals < d * salt_vs_pepper);

    image.setTo(cv::Scalar(0), (vals >= d * salt_vs_pepper) & (vals < d));

    return image;
}

// adding speckle noise
cv::Mat speckle_noise(cv::Mat image, double var){
    cv::Mat speckle(image.size(), CV_32F);
    cv::randn(speckle, cv::Scalar(0), cv::Scalar(sqrt(var)));
    if (image.depth() == CV_8U){
        cv::Mat image_f;
        image.convertTo(image_f, CV_32F);
        image_f += image_f.mul(speckle);
        image_f.convertTo(image, image.type());
    } else 
        image += image.mul(speckle);

    return image;
}

// adding gaussian noise
cv::Mat gaussian_noise(cv::Mat image, double var){
    double mean = 0;

    cv::Mat gauss(image.size(), CV_32F);
    cv::randn(gauss, cv::Scalar(mean), cv::Scalar(sqrt(var)));

    if (image.depth() == CV_8U){
        cv::Mat image_f;
        image.convertTo(image_f, CV_32F);
        image_f += gauss * 255;
        image_f.convertTo(image, image.type());
    } else 
        image += gauss;

    return image;
}

// helper method for poisson noise
std::vector<float> unique(const cv::Mat &image, bool sort = false){
    if (image.depth() != CV_32F){
        std::cerr << "unique() method only works with CV_32F Mat"
        << std::endl;
        return std::vector<float>();
    }

    std::vector<float> out;
    int rows = image.rows;
    int cols = image.cols * image.channels();
    if (image.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    for (int y = 0; y < rows; ++y){
        const float *row_ptr = image.ptr<float>(y);
        for (int x = 0; x < cols; ++x){
            float value = row_ptr[x];
            if (std::find(out.begin(), out.end(), value) == out.end()){
                out.push_back(value);
            }
        }
    }

    if (sort)
        std::sort(out.begin(), out.end());
    
    return out;
}


// adding poisson noise
cv::Mat poisson_noise(cv::Mat image){
    int image_depth = image.depth();
    if (image.depth() == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);
    
    size_t vals = unique(image).size();
    vals = (size_t)pow(2, ceil(log2(vals)));
    int rows = image.rows;
    int cols = image.cols;

    if (image.isContinuous()){
        cols *= rows;
        rows = 1;
    }

    using param_t = std::poisson_distribution<int>::param_type;
    std::default_random_engine engine;
    std::poisson_distribution<> poisson;

    for (int i = 0; i < rows; ++i){
        float *ptr = image.ptr<float>(i);
        for (int j = 0; j < cols; ++j){
            ptr[j] = float(poisson(engine, param_t({ptr[j] * vals}))) / vals;
        }
    }

    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);

    return image;
} 


int main(){

    // loading the image
    std::string filename = "original.jpg";
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // checking if the image was properly loaded
    if (image.empty()){
        std::cout << "Couldn't read the image " << filename << std::endl;
        exit(0);
    }

    // saving grayscale image
    cv::imwrite("original_gray.jpg", image);

    cv::Mat working_image;

    // saving image with impulse noise
    image.copyTo(working_image);
    cv::imwrite("impulse_noise.jpg", impulse_noise(working_image, 0.05));

    // saving image with speckle noise
    image.copyTo(working_image);
    cv::imwrite("speckle_noise.jpg", speckle_noise(working_image, 0.55));

    // saving image with gaussian noise
    image.copyTo(working_image);
    cv::imwrite("gaussian_noise.jpg", gaussian_noise(working_image, 0.2));

    // saving image with poisson noise
    image.copyTo(working_image);
    cv::imwrite("poisson_noise.jpg", poisson_noise(working_image));


    return 0;
}