import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='slr/model/slr_model.tflite',
        num_threads=1,
    ):
        #: Initializing tensor interpreter
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        self.interpreter.allocate_tensors()

        #: Input Output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        result_index, _ = self.predict(landmark_list)
        return result_index

    def predict(self, landmark_list, confidence_threshold=0.5):

        input_details_tensor_index = self.input_details[0]['index']

        #: Feeding landmarks to the tensor interpreter
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32)
        )

        #: Invoking interpreter for prediction
        self.interpreter.invoke()

        #: Getting tensor index from output details
        output_details_tensor_index = self.output_details[0]['index']
        # print(output_details_tensor_index)

        #: Getting all the prediction percentage
        result = self.interpreter.get_tensor(output_details_tensor_index)
        
        confidence = float(max(np.squeeze(result)))

        if confidence > confidence_threshold:
            #: Getting index of maximum accurate label
            result_index = int(np.argmax(np.squeeze(result)))
            
            return result_index, confidence
        else:
            return 25, confidence
            
