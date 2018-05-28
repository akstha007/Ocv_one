package com.developers.meraki.ocv_one;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.absdiff;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.adaptiveThreshold;
import static org.opencv.imgproc.Imgproc.putText;

public class SimilarityActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    public static final int VIEW_MODE_RGBA = 0;
    public static final int VIEW_MODE_CONTOUR1 = 1;
    public static final int VIEW_MODE_CONTOUR2 = 2;
    public static final int VIEW_MODE_CONTOUR3 = 3;

    private MenuItem mItemPreviewRGBA;
    private MenuItem mItemPreviewContour1;
    private MenuItem mItemPreviewContour2;
    private MenuItem mItemPreviewContour3;
    private CameraBridgeViewBase mOpenCvCameraView;

    private Mat mIntermediateMat;
    private int mHistSizeNum = 25;
    private Mat mSepiaKernel;

    private int[][] rawImages = {
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a11, 11}, {R.raw.a21, 21}, {R.raw.a31, 31}, {R.raw.a41, 41}, {R.raw.a51, 51}, {R.raw.a61, 61}, {R.raw.a71, 71}, {R.raw.a81, 81},
            {R.raw.a112, 112}, {R.raw.a212, 212}, {R.raw.a312, 312}, {R.raw.a412, 412}, {R.raw.a512, 512}, {R.raw.a612, 612}, {R.raw.a712, 712}, {R.raw.a812, 812}
    };

    public static int viewMode = VIEW_MODE_RGBA;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public SimilarityActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.image_manipulations_surface_view);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.image_manipulations_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemPreviewRGBA = menu.add("Preview RGBA");
        mItemPreviewContour1 = menu.add("Contour 1");
        mItemPreviewContour2 = menu.add("Contour 2");
        mItemPreviewContour3 = menu.add("Contour 3");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemPreviewRGBA)
            viewMode = VIEW_MODE_RGBA;
        if (item == mItemPreviewContour1)
            viewMode = VIEW_MODE_CONTOUR1;
        else if (item == mItemPreviewContour2)
            viewMode = VIEW_MODE_CONTOUR2;
        else if (item == mItemPreviewContour3)
            viewMode = VIEW_MODE_CONTOUR3;
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        mIntermediateMat = new Mat();

        // Fill sepia kernel
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
    }

    public void onCameraViewStopped() {
        // Explicitly deallocate Mats
        if (mIntermediateMat != null)
            mIntermediateMat.release();

        mIntermediateMat = null;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        Size sizeRgba = rgba.size();

        Mat rgbaInnerWindow;

        int rows = (int) sizeRgba.height;
        int cols = (int) sizeRgba.width;

        int left = cols / 8;
        int top = rows / 8;

        int width = cols * 3 / 4;
        int height = rows * 3 / 4;

        switch (ImageManipulationsActivity.viewMode) {
            case ImageManipulationsActivity.VIEW_MODE_RGBA:
                //Contours(rgba);
                getContourArea3(rgba);
                break;

            case ImageManipulationsActivity.VIEW_MODE_CONTOUR1:
                //Contours(rgba);
                getContourArea1(rgba);
                break;

            case ImageManipulationsActivity.VIEW_MODE_CONTOUR2:
                rgba = getContourArea2(rgba);
                break;

            case ImageManipulationsActivity.VIEW_MODE_CONTOUR3:
                getContourArea3(rgba);
                break;
        }

        return rgba;
    }

    private static void getContourArea1(Mat bmp) {
        Mat src = new Mat();
        src = bmp;
        //Utils.bitmapToMat(bmp, src);
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
        //adaptiveThreshold(gray, src, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
        //Imgproc.adaptiveThreshold(gray, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

        Imgproc.Canny(gray, gray, 50, 200);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        double maxArea = 2000;
        // find contours:
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            Mat contour = contours.get(contourIdx);
            double contourArea = Imgproc.contourArea(contour);

            if (contourArea > maxArea) {
                //Imgproc.drawContours(src, contours, contourIdx, new Scalar(0, 0, 255), 1);

                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);

            }
        }
        //return src;

    }

    private static Mat getContourArea2(Mat bmp) {
        Mat src = new Mat();
        src = bmp;
        //Utils.bitmapToMat(bmp, src);
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
        //adaptiveThreshold(gray, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);
        //Imgproc.adaptiveThreshold(gray, src, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

        //Imgproc.Canny(gray, gray, 50, 200);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        double minArea = 4000, maxArea = 8000;
        // find contours:
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            Mat contour = contours.get(contourIdx);
            double contourArea = Imgproc.contourArea(contour);

            if (contourArea > minArea && contourArea < maxArea) {
                //Imgproc.drawContours(src, contours, contourIdx, new Scalar(0, 0, 255), 1);

                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);

            }
        }
        return bmp;

    }

    private void getContourArea3(Mat src) {

        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);

        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        double minArea = 4000, maxArea = 8000;

        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            Mat contour = contours.get(contourIdx);
            double contourArea = Imgproc.contourArea(contour);

            if (contourArea > minArea && contourArea < maxArea) {

                Rect rect = Imgproc.boundingRect(contours.get(contourIdx));
                Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);

                Mat imCrop = new Mat(src, rect);
                String sim = compareImages2(imCrop);

                Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);
                putText(src, "" + sim, new Point(rect.x, rect.y),
                        Core.FONT_HERSHEY_PLAIN, 2.0, new Scalar(255, 0, 0));

                /*if (sim > 0) {
                    Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);
                    putText(src, "" + sim, new Point(rect.x, rect.y),
                            Core.FONT_HERSHEY_PLAIN, 2.0, new Scalar(255, 0, 0));
                } else {
                    Imgproc.rectangle(src, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), -1);
                }*/

            }
        }
    }

    //for one-one image comparision
    private int compareImages(Mat img1) {

        String imgName = "a11";
        InputStream imageStream = this.getResources().openRawResource(R.raw.a11);
        Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
        Mat img2 = new Mat();
        Utils.bitmapToMat(bitmap, img2);


        //resize the images
        Mat resizeImg1 = new Mat();
        Mat resizeImg2 = new Mat();
        Size sz = new Size(50, 50);

        Imgproc.resize(img1, resizeImg1, sz);
        Imgproc.resize(img2, resizeImg2, sz);

        //resizeImg1.convertTo(resizeImg1, CvType.CV_32F);
        //resizeImg2.convertTo(resizeImg2, CvType.CV_32F);

        Imgproc.cvtColor(resizeImg1, resizeImg1, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.cvtColor(resizeImg2, resizeImg2, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(resizeImg1, resizeImg1, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);
        adaptiveThreshold(resizeImg2, resizeImg2, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

        Mat s1 = new Mat();
        absdiff(resizeImg1, resizeImg2, s1);       // |I1 - I2|
        s1.convertTo(s1, CvType.CV_32F);  // cannot make a square on 8 bits
        s1 = s1.mul(s1);           // |I1 - I2|^2

        Scalar s = Core.sumElems(s1);        // sum elements per channel
        double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
        //double mse  = sse/ (double)(resizeImg1.channels() * resizeImg1.total());
        double mse = sse / resizeImg1.channels();

        return (int) mse;
    }

    private String compareImages1(Mat img1) {

        //resize the images
        Mat resizeImg1 = new Mat();
        Size sz = new Size(50, 50);
        Imgproc.resize(img1, resizeImg1, sz);
        Imgproc.cvtColor(resizeImg1, resizeImg1, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(resizeImg1, resizeImg1, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

        double minError = 100;
        String imageName = "";

        for (int i = 0; i < rawImages.length; i++) {
            InputStream imageStream = this.getResources().openRawResource(rawImages[i][0]);
            Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
            Mat img2 = new Mat();
            Utils.bitmapToMat(bitmap, img2);

            Mat resizeImg2 = new Mat();
            Imgproc.resize(img2, resizeImg2, sz);
            Imgproc.cvtColor(resizeImg2, resizeImg2, Imgproc.COLOR_RGBA2GRAY);
            adaptiveThreshold(resizeImg2, resizeImg2, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

            Mat s1 = new Mat();
            absdiff(resizeImg1, resizeImg2, s1);       // |I1 - I2|
            s1.convertTo(s1, CvType.CV_32F);  // cannot make a square on 8 bits
            s1 = s1.mul(s1);           // |I1 - I2|^2

            Scalar s = Core.sumElems(s1);        // sum elements per channel
            double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
            double mse = sse / resizeImg1.channels();
            if (minError > mse) {
                minError = mse;
                imageName = "" + rawImages[i][1];
            }
        }

        return imageName;
    }

    private String compareImages2(Mat img1) {

        //resize the images
        Mat resizeImg1 = new Mat();
        Size sz = new Size(50, 50);
        Imgproc.resize(img1, resizeImg1, sz);
        Imgproc.cvtColor(resizeImg1, resizeImg1, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(resizeImg1, resizeImg1, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

        double minError = 200;
        String imageName = "";

        for (int i = 0; i < rawImages.length; i++) {
            InputStream imageStream = this.getResources().openRawResource(rawImages[i][0]);
            Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
            Mat img2 = new Mat();
            Utils.bitmapToMat(bitmap, img2);

            Mat resizeImg2 = new Mat();
            Imgproc.resize(img2, resizeImg2, sz);
            Imgproc.cvtColor(resizeImg2, resizeImg2, Imgproc.COLOR_RGBA2GRAY);
            adaptiveThreshold(resizeImg2, resizeImg2, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 40);

            Mat s1 = new Mat();
            absdiff(resizeImg1, resizeImg2, s1);       // |I1 - I2|
            s1.convertTo(s1, CvType.CV_32F);  // cannot make a square on 8 bits
            s1 = s1.mul(s1);           // |I1 - I2|^2

            Scalar s = Core.sumElems(s1);        // sum elements per channel
            double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
            double mse = sse / resizeImg1.channels();
            if (minError > mse) {
                minError = mse;
                imageName = "" + rawImages[i][1];
            }
        }

        return imageName;
    }


}
