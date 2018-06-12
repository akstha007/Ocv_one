package com.developers.meraki.ocv_one;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera.Size;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
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
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.ListIterator;

import static org.opencv.core.Core.absdiff;
import static org.opencv.imgproc.Imgproc.ADAPTIVE_THRESH_MEAN_C;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY;
import static org.opencv.imgproc.Imgproc.adaptiveThreshold;
import static org.opencv.imgproc.Imgproc.putText;

public class Tutorial3Activity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "OCVSample::Activity";

    private Tutorial3View mOpenCvCameraView;
    private List<Size> mResolutionList;
    private MenuItem[] mEffectMenuItems;
    private SubMenu mColorEffectsMenu;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;

    private int[][] rawImages = {
            {R.raw.a, 1}, {R.raw.b, 2}, {R.raw.c, 3}, {R.raw.d, 4}, {R.raw.e, 5}, {R.raw.f, 6},
            {R.raw.g, 7}, {R.raw.h, 8}, {R.raw.i, 9}, {R.raw.j, 10}, {R.raw.k, 11}, {R.raw.l, 12},
            {R.raw.m, 13}, {R.raw.n, 14}, {R.raw.o, 15}, {R.raw.p, 16}, {R.raw.q, 17}, {R.raw.r, 18},
            {R.raw.s, 19}, {R.raw.t, 20}, {R.raw.u, 21}, {R.raw.v, 22}, {R.raw.w, 23}, {R.raw.x, 24}
    };

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(Tutorial3Activity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public Tutorial3Activity() {
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

        setContentView(R.layout.tutorial3_surface_view);

        mOpenCvCameraView = (Tutorial3View) findViewById(R.id.tutorial3_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

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

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        getContourArea3(rgba);
        return rgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        List<String> effects = mOpenCvCameraView.getEffectList();

        if (effects == null) {
            Log.e(TAG, "Color effects are not supported by device!");
            return true;
        }

        mColorEffectsMenu = menu.addSubMenu("Color Effect");
        mEffectMenuItems = new MenuItem[effects.size()];

        int idx = 0;
        ListIterator<String> effectItr = effects.listIterator();
        while (effectItr.hasNext()) {
            String element = effectItr.next();
            mEffectMenuItems[idx] = mColorEffectsMenu.add(1, idx, Menu.NONE, element);
            idx++;
        }

        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];

        ListIterator<Size> resolutionItr = mResolutionList.listIterator();
        idx = 0;
        while (resolutionItr.hasNext()) {
            Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }

        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item.getGroupId() == 1) {
            mOpenCvCameraView.setEffect((String) item.getTitle());
            Toast.makeText(this, mOpenCvCameraView.getEffect(), Toast.LENGTH_SHORT).show();
        } else if (item.getGroupId() == 2) {
            int id = item.getItemId();
            Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution);
            resolution = mOpenCvCameraView.getResolution();
            String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
            Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        }

        return true;
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
                Mat imCrop = new Mat(src, rect);
                String sim = compareImages2(imCrop);

                Imgproc.rectangle(gray, rect.tl(), rect.br(), new Scalar(255, 0, 0, .8), 2);

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

    private String compareImages2(Mat img1) {

        //resize the images
        Mat resizeImg1 = new Mat();
        org.opencv.core.Size sz = new org.opencv.core.Size(50, 50);
        Imgproc.resize(img1, resizeImg1, sz);
        Imgproc.cvtColor(resizeImg1, resizeImg1, Imgproc.COLOR_RGBA2GRAY);
        adaptiveThreshold(resizeImg1, resizeImg1, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);

        double minError = 200;
        String imageName = "";

        for (int i = 0; i < rawImages.length; i++) {
            InputStream imageStream = this.getResources().openRawResource(rawImages[i][0]);
            Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
            Mat img2 = new Mat();
            Utils.bitmapToMat(bitmap, img2);
            Imgproc.resize(img2, img2, sz);

            Mat resizeImg2 = new Mat();
            Imgproc.resize(img2, resizeImg2, sz);
            Imgproc.cvtColor(resizeImg2, resizeImg2, Imgproc.COLOR_RGBA2GRAY);
            adaptiveThreshold(resizeImg2, resizeImg2, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);

            Mat s1 = new Mat();
            absdiff(resizeImg1, resizeImg2, s1);       // |I1 - I2|
            s1.convertTo(s1, CvType.CV_32F);  // cannot make a square on 8 bits
            s1 = s1.mul(s1);           // |I1 - I2|^2

            Scalar s = Core.sumElems(s1);        // sum elements per channel
            double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
            double mse = 0;
            if (sse <= 1e-10) // for small values return zero
                mse = 0;
            else {
                mse = sse / (double) (resizeImg1.channels() * resizeImg1.total());
                //mse = 10.0 * Math.log10((255 * 255) / mse);
            }
            if (minError > mse) {
                minError = mse;
                imageName = "" + (char) (rawImages[i][1] + 'A' - 1);
            }
        }

        return imageName;
    }

    @SuppressLint("SimpleDateFormat")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.i(TAG, "onTouch event");
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());

        File imagesFolder = new File(Environment.getExternalStorageDirectory().getPath(), "OCV_one");
        imagesFolder.mkdirs();

        String fileName = Environment.getExternalStorageDirectory().getPath() + "/OCV_one/" + currentDateandTime + ".jpg";

        File image = new File(fileName);
        try {
            image.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }


        mOpenCvCameraView.takePicture(fileName);
        Toast.makeText(this, fileName + " saved", Toast.LENGTH_SHORT).show();
        return false;
    }
}
