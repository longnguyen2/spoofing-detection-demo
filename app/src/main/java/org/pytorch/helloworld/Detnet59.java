package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Detnet59 {
    private Module module;
    private Context context;

    public Detnet59(Context context) {
        this.context = context;
    }

    public void loadModule() {
        try {
            module = Module.load(assetFilePath(context, "detnet59.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double detect(Mat rgb) {
        Bitmap bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgb, bmp);
        return detect(bmp);
    }

    public double detect(Bitmap bmp) {
        Tensor data = TensorImageUtils.bitmapToFloat32Tensor(bmp, new float[]{0f, 0f, 0f}, new float[]{1f/255.0f, 1f/255.0f, 1f/255.0f});
        Tensor output = module.forward(IValue.from(data)).toTensor();
        float[] outputs = output.getDataAsFloatArray();
//        Log.d("Detection", outputs[0] + " " + outputs[1]);
        float[] values = getValue(outputs);
//        Log.d("Detection", values[0] + " " + values[1]);
        return values[1];
    }

    public float[] getValue(float[] params) {
        float sum = 0;

        for (int i=0; i<params.length; i++) {
            params[i] = (float) Math.exp(params[i]);
            sum += params[i];
        }

        if (Double.isNaN(sum) || sum < 0) {
            for (int i=0; i<params.length; i++) {
                params[i] = 1.0f / params.length;
            }
        } else {
            for (int i=0; i<params.length; i++) {
                params[i] = params[i] / sum;
            }
        }

        return params;
    }

    private String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            Log.d("model", file.getAbsolutePath());
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            Log.d("model", file.getAbsolutePath());
            return file.getAbsolutePath();
        }
    }
}
