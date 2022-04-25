package pbartkow.food_detection_classification

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import pbartkow.food_detection_classification.ml.Model
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {

    val REQUEST_IMAGE_CAPTURE = 420

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val takePhotoButton = findViewById<Button>(R.id.takePhotoButton)
        takePhotoButton.setOnClickListener {
            val imageCaptureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(imageCaptureIntent, REQUEST_IMAGE_CAPTURE)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if(requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // Load image
            var imageBitmap = data!!.extras!!.get("data") as Bitmap

            // Rescale
            var rescaledImageBitmap = Bitmap.createScaledBitmap(
                imageBitmap,
                224,
                224,
                false
            )

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(rescaledImageBitmap)

            val model = Model.newInstance(this)

            // Create inputs for reference.
            val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
            inputFeature.loadBuffer(tensorImage.buffer)

            // Run model inference and gets result.
            val outputs = model.process(inputFeature)
            val outputFeature = outputs.outputFeature0AsTensorBuffer

            // Release model resources if no longer used.
            model.close()
        }
    }
}