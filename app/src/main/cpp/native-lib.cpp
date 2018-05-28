#include "native-lib.h"


JNIEXPORT void JNICALL Java_com_developers_meraki_opencvtest_OpencvClass_faceDetection
        (JNIEnv *, jclass, jlong addrRgba){
    Mat& frame = *(Mat*)addrRgba;

}
