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
import org.opencv.core.Size;
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

    private Context context;
    private BaseLoaderCallback mLoaderCallback;
    public FaceDetection(Context _context, final CameraBridgeViewBase camera) {
        this.context = _context;
        mLoaderCallback = new BaseLoaderCallback(context) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                        Log.i(TAG, "OpenCV loaded successfully");

                        // Load native library after(!) OpenCV initialization

                        try {
                            // load cascade file from application resources
                            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface);
                            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
                            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface.xml");
                            FileOutputStream os = new FileOutputStream(mCascadeFile);

                            byte[] buffer = new byte[4096];
                            int bytesRead;
                            while ((bytesRead = is.read(buffer)) != -1) {
                                os.write(buffer, 0, bytesRead);
                            }
                            is.close();
                            os.close();

                            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());

                            cascadeDir.delete();

                        } catch (IOException e) {
                            e.printStackTrace();
                            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                        }
                        camera.enableView();
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

    public Rect detect(Mat mGray) {
        MatOfRect faces = new MatOfRect();

        mJavaDetector.detectMultiScale(mGray, faces, 1.1, 4, 2, new Size(Constants.minFaceSize, Constants.minFaceSize), new Size());

        if (faces.toArray().length == 0) return null;
        Rect face = faces.toArray()[0];
        face = scale(face, Constants.scale, mGray.width(), mGray.height());
        return face;
    }

    private Rect scale(Rect rect, float factor, int width, int height) {
        int centerX = rect.x + (rect.width / 2);
        int centerY = rect.y + (rect.height / 2);
        rect.x = centerX - ((int) (rect.width * factor) / 2);
        rect.y = centerY - ((int) ((rect.height * factor) / 2));
        rect.height = (int) (rect.height * factor);
        rect.width = (int) (rect.width * factor);
        if (rect.x < 0 || rect.y < 0 || rect.width > width || rect.height > height) return null;
        else return rect;
    }
}
