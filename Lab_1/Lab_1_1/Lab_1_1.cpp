#include <iostream>
#include <string>
#include <math.h>
#include <vector>

#include<opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat make_hist(cv::Mat image){

    // splitting channels (Blue, Green, Red)
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);
    cv::Mat image_B = image_BGR[0];
    cv::Mat image_G = image_BGR[1];
    cv::Mat image_R = image_BGR[2];


    // histogram calculation for each channel
    int histSize = 256;
    float range [] = {0, 256};
    const float * histRange[] = {range};
    cv::Mat bHist, gHist, rHist;
    cv::calcHist(&image_B, 1, 0, cv::Mat(), bHist, 1, &histSize, histRange);
    cv::calcHist(&image_G, 1, 0, cv::Mat(), gHist, 1, &histSize, histRange);
    cv::calcHist(&image_R, 1, 0, cv::Mat(), rHist, 1, &histSize, histRange);


    // making a histogram image
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    cv::normalize(bHist, bHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(gHist, gHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    cv::normalize(rHist, rHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(bHist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(bHist.at<float>(i)) ),
              cv::Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(gHist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(gHist.at<float>(i)) ),
              cv::Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(rHist.at<float>(i-1)) ),
              cv::Point( bin_w*(i), hist_h - cvRound(rHist.at<float>(i)) ),
              cv::Scalar( 0, 0, 255), 2, 8, 0  );
    }


    return histImage;
}

cv::Mat hist_shift(cv::Mat image, int shiftDist = 50){
    // Adding shift to intensity
    for (int row = 0; row < image.rows; ++row){
        for (int col = 0; col < image.cols; ++col){
            cv::Vec3b color = image.at<cv::Vec3b>(row, col);
            image.at<cv::Vec3b>(row, col) = cv::Vec3b(color[0] + shiftDist, color[1] + shiftDist, color[2] + shiftDist);
        }
    }

    return image;
}

cv::Mat extension_lut(int Imin, int Imax, float alpha = 0.5){
    // lut for extension transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = (i - Imin)*1.0 / (Imax - Imin);
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(255 * pow(var, alpha));
    }
    return lut;
}

cv::Mat hist_extension(cv::Mat image, float alpha = 0.5){
    // extending a histogram using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);


    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        double Imin , Imax;
        cv::minMaxLoc(image_BGR[channel], &Imin, &Imax);
        cv::LUT(image_BGR[channel], extension_lut(Imin, Imax, alpha), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;
}

cv::Mat calculate_prob(cv::Mat channel){
    // calculating cumulative histogram
    int histSize = 256;
    float range [] = {0, 256};
    const float * histRange[] = {range};
    cv::Mat hist;
    cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, histRange);

    for (int i = 0; i < hist.size[0]; ++i){
        hist.at<float>(i) /= channel.size[0] * channel.size[1];
    }

    for (int i = 1; i < hist.size[0]; ++i)
        hist.at<float>(i) += hist.at<float>(i - 1);


    return hist;
}

cv::Mat uniform_transformation_lut(int Imin, int Imax, cv::Mat hist){
    // lut for uniform transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = (Imax - Imin) * hist.at<float>(i) + Imin;
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
    }
    return lut;
}

cv::Mat uniform_transformation(cv::Mat image){
    // uniform transformation using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat hist = calculate_prob(image_BGR[channel]);
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        double Imin , Imax;
        cv::minMaxLoc(image_BGR[channel], &Imin, &Imax);
        cv::LUT(image_BGR[channel], uniform_transformation_lut(Imin, Imax, hist), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

cv::Mat exp_transformation_lut(int Imin, int Imax, cv::Mat hist, float alpha){
    // lut for expanencial transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = Imin - (1 / alpha)*std::log(1 - hist.at<float>(i));
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
    }
    return lut;
}



cv::Mat exp_transform(cv::Mat image, float alpha = 0.5){
    // exponencial transformation using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat hist = calculate_prob(image_BGR[channel]);
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        double Imin , Imax;
        cv::minMaxLoc(image_BGR[channel], &Imin, &Imax);
        cv::LUT(image_BGR[channel], exp_transformation_lut(Imin, Imax, hist, alpha), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}


cv::Mat Relay_transformation_lut(int Imin, int Imax, cv::Mat hist, float alpha){
    // lut for Relay transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = Imin + std::sqrt(2*alpha*alpha*std::log(1.0 / (1 - hist.at<float>(i))));
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
    }
    return lut;
}


cv::Mat Relay_transform(cv::Mat image, float alpha = 0.5){
    // Relay transformation using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat hist = calculate_prob(image_BGR[channel]);
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        double Imin , Imax;
        cv::minMaxLoc(image_BGR[channel], &Imin, &Imax);
        cv::LUT(image_BGR[channel], Relay_transformation_lut(Imin, Imax, hist, alpha), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

cv::Mat pow_transformation_lut(cv::Mat hist){
    // lut for power 2/3 transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = pow(hist.at<float>(i), 2.0/3.0) * 255;
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
    }
    return lut;
}


cv::Mat pow_transform(cv::Mat image){
    // pow 2/3 transformation using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat hist = calculate_prob(image_BGR[channel]);
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        cv::LUT(image_BGR[channel], pow_transformation_lut(hist), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

cv::Mat hyp_transformation_lut(cv::Mat hist, float alpha){
    // lut for hyperbolic transformation
    cv::Mat lut(1, 256, CV_8U);
    uchar *lut_ptr = lut.ptr();
    for (int i = 0; i < 256; i++){
        double var = pow(alpha, hist.at<float>(i));
        if (var < 0)
            lut_ptr[i] = 0;
        else
            lut_ptr[i] = cv::saturate_cast<uchar>(var);
    }
    return lut;
}


cv::Mat hyp_transform(cv::Mat image, float alpha = 0.05){
    // hyperbolic transformation using LUT
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat hist = calculate_prob(image_BGR[channel]);
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        cv::LUT(image_BGR[channel], hyp_transformation_lut(hist, alpha), image_new);
        image_BGR[channel] = image_new;
    }


    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

cv::Mat equalize_BGR(cv::Mat image){
    // built-in cpp function for each channel
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        cv::Mat image_new = cv::Mat(image_BGR[channel].rows, image_BGR[channel].cols, image_BGR[channel].depth());
        cv::equalizeHist(image_BGR[channel], image_new);
        image_BGR[channel] = image_new;
    }

    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

cv::Mat clahe_BGR(cv::Mat image){
    // built-in cpp function for each channel
    std::vector<cv::Mat> image_BGR;
    cv::split(image, image_BGR);

    for (int channel = 0; channel < image_BGR.size(); ++channel){
        auto clahe = cv::createCLAHE();
        cv::Mat image_new;
        clahe->apply(image_BGR[channel], image_new);
        image_BGR[channel] = image_new;
    }

    cv::Mat image_new;
    cv::merge(image_BGR, image_new);
    return image_new;


    return image;
}

int main(){
    // loading the image
    std::string image_path = "photo_2024-02-26_16-07-29.jpg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    
    // checking if the image was properly loaded
    if (image.empty()){
        std::cout << "Couldn't read the image " << image_path << std::endl;
        std::exit(1);
    }
    
    // saving original image and its histogram
    cv::imwrite("original.jpg", image);
    cv::imwrite("hist_original.jpg", make_hist(image));

    // saving shifted image and its histogram
    cv::Mat image_new;
    image.copyTo(image_new);
    image_new = hist_shift(image_new, 50);
    cv::imwrite("shifted.jpg", image_new);
    cv::imwrite("hist_shifted.jpg", make_hist(image_new));

    // TODO: скажи, что из-за того, что на оригинальной гистограмме был всплеск "справа"(то есть присутствуют яркие области фотографии), при сдвиге мы
    // потеряли информацию о цветах этих пикселей, поэтому получились дефекты

    // saving extended image and its histogram
    image.copyTo(image_new);
    image_new = hist_extension(image_new, 0.5);
    cv::imwrite("extended.jpg", image_new);
    cv::imwrite("hist_extended.jpg", make_hist(image_new));

    // saving unform transformed image and its histogram
    image.copyTo(image_new);
    image_new = uniform_transformation(image_new);
    cv::imwrite("uniform_transformed.jpg", image_new);
    cv::imwrite("hist_uniform_transformed.jpg", make_hist(image_new));
    
    // saving exponencial transformed image and its histogram
    image.copyTo(image_new);
    image_new = exp_transform(image_new, 0.05);
    cv::imwrite("exponencial_transformed_dark.jpg", image_new);
    cv::imwrite("hist_exponencial_transformed_dark.jpg", make_hist(image_new));
    
    // TODO: сдвигаем гистограмму, потому что очень темное изображение
    image.copyTo(image_new);
    image_new = exp_transform(image_new, 0.05);
    image_new = hist_shift(image_new, 80);
    cv::imwrite("exponencial_transformed.jpg", image_new);
    cv::imwrite("hist_exponencial_transformed.jpg", make_hist(image_new));

    // saving Relay transformed image and its histogram
    image.copyTo(image_new);
    image_new = Relay_transform(image_new, 5);
    cv::imwrite("Relay_transformed_dark.jpg", image_new);
    cv::imwrite("hist_Relay_transformed_dark.jpg", make_hist(image_new));


    // TODO: сдвигаем гистограмму, потому что очень темное изображение и растягиваем попутно динамический диапазон
    image.copyTo(image_new);
    image_new = Relay_transform(image_new, 5);
    image_new = hist_shift(image_new, 50);
    image_new = hist_extension(image_new);
    cv::imwrite("Relay_transformed.jpg", image_new);
    cv::imwrite("hist_Relay_transformed.jpg", make_hist(image_new));

    // saving power 2/3 transformed image and its histogram
    image.copyTo(image_new);
    image_new = pow_transform(image_new);
    cv::imwrite("power_transformed.jpg", image_new);
    cv::imwrite("hist_power_transformed.jpg", make_hist(image_new));

    
    // saving hyperbolic transformed image and its histogram
    image.copyTo(image_new);
    image_new = hyp_transform(image_new, 5);
    image_new = hist_shift(image_new, 50);
    image_new = hist_extension(image_new, 0.8);
    cv::imwrite("hyperbolic_transformed.jpg", image_new);
    cv::imwrite("hist_hyperbolic_transformed.jpg", make_hist(image_new));

    // saving equalize transformation from opencv (image and its histogram)
    image.copyTo(image_new);
    image_new = equalize_BGR(image_new);
    cv::imwrite("equalize_transformed.jpg", image_new);
    cv::imwrite("hist_equalize_transformed.jpg", make_hist(image_new));


    // saving CLAHE transformation from opencv (image and its histogram)
    image.copyTo(image_new);
    image_new = clahe_BGR(image_new);
    cv::imwrite("CLAHE_transformed.jpg", image_new);
    cv::imwrite("hist_CLAHE_transformed.jpg", make_hist(image_new));

    return 0;
}