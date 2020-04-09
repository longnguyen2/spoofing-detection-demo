package org.pytorch.helloworld;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;

public class SpoofingDetectionActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private CameraBridgeViewBase cameraBridgeViewBase;
    private FaceDetection faceDetection;
    private Detnet59 detnet59;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_spoofing_detection);

        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, 0);
        } else {
            init();
        }
    }

    private void init() {
        detnet59 = new Detnet59(this);
        detnet59.loadModule();
        faceDetection = new FaceDetection(this);
        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.camera);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            init();
            faceDetection.init();
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
        if (faceDetection != null) faceDetection.init();
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
        Rect[] faces = faceDetection.detect(inputFrame);
        if (faces.length == 0) {
            return inputFrame.rgba();
        } else {
            Mat cropped = new Mat(inputFrame.rgba(), faces[0]);
            Log.d("Detection", detnet59.detect(cropped) + "");
            return inputFrame.rgba();
        }
    }
}
