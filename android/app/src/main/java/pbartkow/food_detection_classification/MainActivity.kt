package pbartkow.food_detection_classification

import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import pbartkow.food_detection_classification.ml.Model
import java.io.File
import kotlin.math.exp


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
            val imageBitmap = data!!.extras!!.get("data") as Bitmap

            // Rescale
            val rescaledImageBitmap = Bitmap.createScaledBitmap(
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
            val outputFloats= outputFeature.floatArray

            val classes = assets.open("classes.txt").bufferedReader().use{it.readLines()}

            for(x in 1 until outputFeature.shape[1]) {
                for(y in 1 until outputFeature.shape[2]) {

                    var maxIndex = -1
                    var maxValue = -1.0f
                    var classSum = 0.0f

                    for(classNumber in 5 until outputFeature.shape[4]) {
                        val classValue = exp(outputFloats[x * y * classNumber])

                        if(classValue > maxValue) {
                            maxIndex = classNumber - 5
                            maxValue = classValue
                        }

                        classSum += classValue
                    }

                    println("PREDICTION: " + classes[maxIndex] + " PROBABILITY: " + maxValue / classSum)
                }
            }

            // Release model resources if no longer used.
            model.close()
        }
    }
}