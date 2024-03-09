#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

// saving median blur
void median_save(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image, int ker_size){
    cv::Mat working_image;

    cv::medianBlur(gaussian_noise_image, working_image, ker_size);
    cv::imwrite("median_blur_gaussian_noise.jpg", working_image);

    cv::medianBlur(impulse_noise_image, working_image, ker_size);
    cv::imwrite("median_blur_impulse_noise.jpg", working_image);

    cv::medianBlur(poisson_noise_image, working_image, ker_size);
    cv::imwrite("median_blur_poisson_noise.jpg", working_image);

    cv::medianBlur(speckle_noise_image, working_image, ker_size);
    cv::imwrite("median_blur_speckle_noise.jpg", working_image);      
}

// rank blur
cv::Mat rank_blur(cv::Mat image, int rank){
    int k_size[] = {3, 3};
    cv::Mat kernel = cv::Mat::ones(k_size[0], k_size[1], CV_64F);

    cv::Mat image_copy;
    if (image.depth() == CV_8U)
        image.convertTo(image_copy, CV_32F, 1.0 / 255);
    else image_copy = image;

    cv::copyMakeBorder(image_copy, image_copy, 
    int((k_size[0] - 1) / 2),
    int(k_size[0] / 2), 
    int((k_size[1] - 1) / 2),
    int(k_size[1] / 2), cv::BORDER_REPLICATE);

    cv::Mat image_tmp = cv::Mat::zeros(image.size(), image_copy.type());
    std::vector<double> c;
    c.reserve(k_size[0] * k_size[1]);

    for (int i = 0; i < image.rows; ++i){
        for (int j = 0; j < image.cols; ++j){
            c.clear();

            for (int a = 0; a < k_size[0]; ++a){
                for (int b = 0; b < k_size[1]; ++b){
                    for (int k = 0; k < kernel.at<double>(a, b); ++k){
                        c.push_back(image_copy.at<float>(i + a, j + b));
                    }
                }
            }
            
            std::sort(c.begin(), c.end());

            image_tmp.at<float>(i, j) = float(c[rank-1]);
        }
    }

    image_copy = image_tmp;

    if (image.depth() == CV_8U)
        image_copy.convertTo(image_copy, CV_8U, 255);


    return image_copy;
}


// median weight blur
cv::Mat weight_blur(cv::Mat image, int rank){
    int k_size[] = {3, 3};
    cv::Mat kernel = (cv::Mat_<double>(3, 3) <<
                                1, 2, 1,
                                2, 3, 2,
                                1, 2, 1); 

    cv::Mat image_copy;
    if (image.depth() == CV_8U)
        image.convertTo(image_copy, CV_32F, 1.0 / 255);
    else image_copy = image;

    cv::copyMakeBorder(image_copy, image_copy, 
    int((k_size[0] - 1) / 2),
    int(k_size[0] / 2), 
    int((k_size[1] - 1) / 2),
    int(k_size[1] / 2), cv::BORDER_REPLICATE);

    cv::Mat image_tmp = cv::Mat::zeros(image.size(), image_copy.type());
    std::vector<double> c;
    c.reserve(k_size[0] * k_size[1]);

    for (int i = 0; i < image.rows; ++i){
        for (int j = 0; j < image.cols; ++j){
            c.clear();

            for (int a = 0; a < k_size[0]; ++a){
                for (int b = 0; b < k_size[1]; ++b){
                    for (int k = 0; k < kernel.at<double>(a, b); ++k){
                        c.push_back(image_copy.at<float>(i + a, j + b));
                    }
                }
            }
            
            std::sort(c.begin(), c.end());

            image_tmp.at<float>(i, j) = float(c[rank-1]);
        }
    }

    image_copy = image_tmp;

    if (image.depth() == CV_8U)
        image_copy.convertTo(image_copy, CV_8U, 255);


    return image_copy;
}

// saving weight blur
void weight_save(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image, int rank){
    cv::Mat working_image;

    gaussian_noise_image.copyTo(working_image);
    cv::imwrite("weight_blur_gaussian_noise.jpg", weight_blur(working_image, rank));
    
    impulse_noise_image.copyTo(working_image);
    cv::imwrite("weight_blur_impulse_noise.jpg", weight_blur(working_image, rank));

    poisson_noise_image.copyTo(working_image);
    cv::imwrite("weight_blur_poisson_noise.jpg", weight_blur(working_image, rank));

    speckle_noise_image.copyTo(working_image);
    cv::imwrite("weight_blur_speckle_noise.jpg", weight_blur(working_image, rank));
}

// saving rank blur
void rank_save(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image, int rank){
    cv::Mat working_image;

    gaussian_noise_image.copyTo(working_image);
    cv::imwrite("rank_blur_gaussian_noise_rank=" + std::to_string(rank) + ".jpg", rank_blur(working_image, rank));
    
    impulse_noise_image.copyTo(working_image);
    cv::imwrite("rank_blur_impulse_noise_rank=" + std::to_string(rank) + ".jpg", rank_blur(working_image, rank));

    poisson_noise_image.copyTo(working_image);
    cv::imwrite("rank_blur_poisson_noise_rank=" + std::to_string(rank) + ".jpg", rank_blur(working_image, rank));

    speckle_noise_image.copyTo(working_image);
    cv::imwrite("rank_blur_speckle_noise_rank=" + std::to_string(rank) + ".jpg", rank_blur(working_image, rank));
}

// wiener blur
cv::Mat wiener_blur(cv::Mat I){

    int k_size [] = {5, 5};
    cv::Mat kernel = cv::Mat::ones(k_size[0], k_size[1], CV_64F);
    double k_sum = cv::sum(kernel)[0];
    
    cv::Mat I_copy;
    if (I.depth() == CV_8U)
        I.convertTo(I_copy, CV_32F, 1.0 / 255);
    else
    I_copy = I;
    
    cv::copyMakeBorder(I_copy, I_copy,
        int((k_size[0] - 1) / 2),
        int(k_size[0] / 2),
        int((k_size[1] - 1) / 2),
        int(k_size[1] / 2), cv::BORDER_REPLICATE);

    cv::Mat I_tmp = cv::Mat::zeros(I.size(), I_copy.type());
    double v(0);
    
    for (int i = 0; i < I.rows; ++i)
        for (int j = 0; j < I.cols; ++j){

            double m(0), q(0);
            for (int a = 0; a < k_size[0]; ++a)
                for (int b = 0; b < k_size[1]; ++b){
                    double t = I_copy.at<float>(i + a, j + b) * kernel.at<double>(a, b);
                    m += t;
                    q += t * t;
                }
            m /= k_sum;
            q /= k_sum;
            q -= m * m;
            v += q;
        }

    v /= I.cols * I.rows;

    for (int i = 0; i < I.rows; ++i)
        for (int j = 0; j < I.cols; ++j){
            double m(0) , q(0);
            for (int a = 0; a < k_size[0]; ++a)
                for (int b = 0; b < k_size[1]; ++b){
                    double t = I_copy.at<float>(i + a, j + b) * kernel.at<double>(a, b);
                    m += t ;
                    q += t * t ;
                }
            m /= k_sum;
            q /= k_sum;
            q -= m * m;

            double im = I_copy.at<float>(i + (k_size[0] - 1) / 2, j + (k_size[1] - 1) / 2);
            if ( q < v )
                I_tmp.at<float>(i, j) = float(m);
            else
                I_tmp.at<float>(i, j) = float((im - m) * (1 - v / q) + m);
        }

    I_copy = I_tmp;

    if (I.depth() == CV_8U)
        I_copy.convertTo(I_copy, CV_8U, 255);

    return I_copy;
}

// saving wiener blur
void wiener_save(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image){
    cv::Mat working_image;


    gaussian_noise_image.copyTo(working_image);
    cv::imwrite("wiener_blur_gaussian_noise.jpg", wiener_blur(working_image));
    
    impulse_noise_image.copyTo(working_image);
    cv::imwrite("wiener_blur_impulse_noise.jpg", wiener_blur(working_image));

    poisson_noise_image.copyTo(working_image);
    cv::imwrite("wiener_blur_poisson_noise.jpg", wiener_blur(working_image));

    speckle_noise_image.copyTo(working_image);
    cv::imwrite("wiener_blur_speckle_noise.jpg", wiener_blur(working_image));
}

// helper method for adaptive median blur
float adaptation(const cv::Mat &image, int row, int col, int ker_size, int s_max){
    std::vector<float> c;

    int pad = ker_size / 2;
    for (int i = -pad; i <= pad; ++i){
        for (int j = -pad; j <= pad; ++j){
            c.push_back(image.at<float>(row + j, col + i));
        }
    }

    std::sort(c.begin(), c.end());
    float z_max = c[c.size() - 1];
    float z_min = c[0];
    float z_med = c[c.size() / 2];

    float ans = image.at<float>(row, col);
    if (z_med > z_min && z_med < z_max){
        if (ans > z_min && ans < z_max){
            return ans;
        } else {
            return z_med;
        }
    } else {
        ker_size += 2;
        if (ker_size <= s_max)
            return adaptation(image, row, col, ker_size, s_max);
        else return z_med;
    }
}

// adaptive median blur
cv::Mat adative_blur(cv::Mat image){
    int image_depth = image.depth();
    if (image_depth == CV_8U)
        image.convertTo(image, CV_32F, 1.0 / 255);

    int ker_size = 2;
    int s_max = 7;
    int max_pad = s_max / 2;
    cv::copyMakeBorder(image, image, max_pad, max_pad, max_pad, max_pad, cv::BORDER_REFLECT);
    
    for (int x = max_pad; x < image.cols - max_pad; ++x){
        for (int y = max_pad; y < image.rows - max_pad; ++y){
            image.at<float>(y, x) = adaptation(image, y, x, ker_size, s_max);
        }
    } 

    if (image_depth == CV_8U)
        image.convertTo(image, CV_8U, 255);

    return image;
}

// saving adaprive median blur
void adaptive_save(cv::Mat gaussian_noise_image, cv::Mat impulse_noise_image, cv::Mat poisson_noise_image, cv::Mat speckle_noise_image){
    cv::Mat working_image;


    gaussian_noise_image.copyTo(working_image);
    cv::imwrite("adaptive_blur_gaussian_noise.jpg", adative_blur(working_image));
    
    impulse_noise_image.copyTo(working_image);
    cv::imwrite("adaptive_blur_impulse_noise.jpg", adative_blur(working_image));

    poisson_noise_image.copyTo(working_image);
    cv::imwrite("adaptive_blur_poisson_noise.jpg", adative_blur(working_image));

    speckle_noise_image.copyTo(working_image);
    cv::imwrite("adaptive_blur_speckle_noise.jpg", adative_blur(working_image));
}

int main(){
    // loading the images
    cv::Mat gaussian_noise_image = cv::imread("gaussian_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat impulse_noise_image = cv::imread("impulse_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat poisson_noise_image = cv::imread("poisson_noise.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat speckle_noise_image = cv::imread("speckle_noise.jpg", cv::IMREAD_GRAYSCALE);

    // saving median blurred image
    median_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 3);
    // TODO: можно заметить, что такая фильтрация действительно очень хорошо устраняет импульсный шум

    // saving median weight blurred image
    weight_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 8);
    // TODO: rank = 8 - середина вектора, значит, это медианная взвешенная фильтрация

    // saving min-blurred image
    rank_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 1);
    // TODO: rank = 1. Берем наименьшее значение в окне, значит, min-фильтр

    // saving max-blurred
    rank_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image, 9);
    // TODO: rank = 9. Берем наибольшее значение в окне, значит, max-фильтр

    // TODO: Очевидно, что в последних двух фильтрах в одном случае получились тёмные изображения, а в другом - светлые

    // saving wiener blurred image
    wiener_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image);

    // saving adaptive median blur
    adaptive_save(gaussian_noise_image, impulse_noise_image, poisson_noise_image, speckle_noise_image);
    return 0;
}