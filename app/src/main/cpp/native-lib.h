#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/opencv.hpp"
using namespace cv;

extern "C" JNIEXPORT jstring

JNICALL
Java_com_developers_meraki_ocv_1one_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

JNIEXPORT void JNICALL Java_com_developers_meraki_opencvtest_OpencvClass_faceDetection
        (JNIEnv *, jclass, jlong);
