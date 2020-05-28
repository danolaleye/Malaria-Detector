from flask import Flask, render_template, request
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from base64 import b64encode
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField


# Initialize the flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sjhef803or0hnc3'

bootstrap = Bootstrap(app)

saved_model = load_model('mobileNet_malaria_detector.h5')
# saved_model.predict()

class UploadForm(FlaskForm):
	"""Upload image form"""
	photo = FileField('Upload an image', validators=[FileAllowed(['jpg', 'png', 'jpeg'], 
	u'Image only!'), FileRequired(u'File was empty!')])
	submit = SubmitField(u'Predict!')

def preprocess(img):
	#preprocess image to the desired shape
	width, height = img.shape[0], img.shape[1]
	img = image.array_to_img(img, scale=False)

	desired_width, desired_height = 130, 130

	if width < desired_width:
		desired_width = width
	start_x = np.maximum(0, int((width-desired_width)/2))

	img = img.crop((start_x, np.maximum(0, height-desired_height), 
		start_x+desired_width, height))
	img = img.resize((130, 130))

	img = image.img_to_array(img)
	return img / 255


@app.route('/', methods = ['GET', 'POST'])
def predict():
	form = UploadForm()
	if form.validate_on_submit():
		print(form.photo.data)
		image_stream = form.photo.data.stream
		original_img = Image.open(image_stream)
		img = image.img_to_array(original_img)
		img = preprocess(img)
		img = np.expand_dims(img, axis=0)
		prediction = saved_model.predict(img)

		prediction = list(zip(*prediction))[1]
		prediction = np.array(prediction) > 0.5

		if prediction:
			result = 'uninfected'
		else:
			result = 'parasitized'

		# if (prediction[0][1] == 0):
		# 	result = 'uninfected'
		# else:
		# 	result = 'parasitized'

		byteIO = BytesIO()
		original_img.save(byteIO, format = original_img.format)
		byte_arr = byteIO.getvalue()
		encoded = b64encode(byte_arr)


		return render_template('result.html', result=result,
			encoded_photo=encoded.decode('ascii'))

	return render_template('index.html', form=form)


if __name__=='__main__':
	app.run(debug=True)
