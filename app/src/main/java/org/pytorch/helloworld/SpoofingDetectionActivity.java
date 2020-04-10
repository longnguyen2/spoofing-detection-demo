package org.pytorch.helloworld;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class SpoofingDetectionActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private CameraBridgeViewBase cameraBridgeViewBase;
    private FaceDetection faceDetection;
    private Detnet59 detnet59;
    private TextView result;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, 0);
        } else {
            init();
        }
    }

    private int cameraId = CameraBridgeViewBase.CAMERA_ID_BACK;
    private void init() {
        setContentView(R.layout.activity_spoofing_detection);
        result = findViewById(R.id.result);
        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.camera);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setCameraIndex(cameraId);
//        cameraBridgeViewBase.setRotationX(cameraId == 0 ? -90 : 90);
        cameraBridgeViewBase.enableView();
        faceDetection = new FaceDetection(this, cameraBridgeViewBase);
        faceDetection.init();
        detnet59 = new Detnet59(this);
        detnet59.loadModule();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            init();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(cameraBridgeViewBase != null) {
            cameraBridgeViewBase.enableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        faceDetection.release();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        /*
         r = max(w, h) / 2
         centerx = x + w / 2
         centery = y + h / 2
         nx = int(centerx - r)
         ny = int(centery - r)
         nr = int(r * 2)

         faceimg = img[ny:ny+nr, nx:nx+nr]
         lastimg = cv2.resize(faceimg, (224, 224))
        */
        Mat rgba = inputFrame.rgba();
        Mat gray = inputFrame.gray();
        Rect face = faceDetection.detect(gray);
        if (face != null) {
            Mat cropped;
            try {
                cropped = new Mat(rgba, face);
            } catch (Exception e) {
                result.setText("");
                return rgba;
            }
            Mat resize = new Mat();
            Imgproc.resize(cropped, resize, new Size(Constants.minFaceSize, Constants.minFaceSize));
            Bitmap bmp = Bitmap.createBitmap(resize.cols(), resize.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(resize, bmp);
            cropped.release();
            resize.release();
            double prob = detnet59.detect(bmp);
            result.setText(prob > 0.25d ? "Fake: " + prob : "Real: " + prob);
            Imgproc.rectangle(rgba, new Point(face.x, face.y), new Point(face.x + face.width, face.y + face.height), new Scalar(255, 255, 255), 2);
        } else {
            result.setText("");
        }
        return rgba;
    }
}
