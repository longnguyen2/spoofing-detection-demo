package org.pytorch.helloworld;

import android.content.Context;
import android.util.Log;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FaceDetection {
    private final String TAG = getClass().getSimpleName();

    private Mat mRgba;
    private Mat mGray;
    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private DetectionBasedTracker mNativeDetector;

    private Context context;
    private BaseLoaderCallback mLoaderCallback;
    public FaceDetection(Context _context) {
        this.context = _context;
        mLoaderCallback = new BaseLoaderCallback(context) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.i(TAG, "OpenCV loaded successfully");

                        // Load native library after(!) OpenCV initialization
                        System.loadLibrary("detection_based_tracker");

                        try {
                            // load cascade file from application resources
                            InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
                            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
                            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                            FileOutputStream os = new FileOutputStream(mCascadeFile);

                            byte[] buffer = new byte[4096];
                            int bytesRead;
                            while ((bytesRead = is.read(buffer)) != -1) {
                                os.write(buffer, 0, bytesRead);
                            }
                            is.close();
                            os.close();

                            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                            if (mJavaDetector.empty()) {
                                Log.e(TAG, "Failed to load cascade classifier");
                                mJavaDetector = null;
                            } else {
                                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                            }

                            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
                            mNativeDetector.setMinFaceSize(Constants.minFaceSize);

                            cascadeDir.delete();

                        } catch (IOException e) {
                            e.printStackTrace();
                            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                        }
                        break;
                    default:
                        super.onManagerConnected(status);
                }
            }
        };
    }

    public void init() {
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, context, mLoaderCallback);
            mGray = new Mat();
            mRgba = new Mat();
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void release() {
        if (mGray != null) mGray.release();
        if (mRgba != null) mRgba.release();
    }

    public Rect[] detect(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        MatOfRect faces = new MatOfRect();

        mNativeDetector.detect(mGray, faces);

        return faces.toArray();
    }
}
