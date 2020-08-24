import tensoflow_datasets as tfds
import tensorflow as tf
from flask import Flask, jsonify, make_response, request


app = Flask(__name__)
padding_size = 1000
model = tf.keras.models.load_model('sentiment_analysis.hdf5')
text_encoder = tfds.features.text.TokenTextEncoder.load_from_file("sa_encoder.vocab")



'''
Function crestes padding to the vector such that each batch will have vectors of
maximum size.

args: vector, size of vector

returns: padded vector
'''
def pad_to_size(vec, size):
  zeroes = [0] * (size - len(vec))
  vec.extend(zeroes)

  return vec

'''
This function taked in the review and encodes the text.

args: review text, padding size

returns: predictions
'''
def predict_fn(pred_text, pad_size):
  encoded_pred_text = text_encoder.encode(pred_text)
  encoded_pred_text = pad_to_size(encoded_pred_text, pad_size)
  encoded_pred_text = tf.cast(encoded_pred_text, tf.int64)
  prediction = model.predict(tf.expand_dims(encoded_pred_text, 0))

  return (prediction.tolist())

@app.route('/seclassifier', methods = ['POST'])

def predict_sentiment():
    text = request.get_json()['text']
    print(text)
    predictions = predict_fn(text, padding_size)
    sentiment = 'positive' if float(''.join(map(str, predictions[0]))) > 0 else 'Negative'
    return  jsonify({'predictions ': predictions, 'sentiment ': sentiment})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000')