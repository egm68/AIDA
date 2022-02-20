from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
 
# extract features from each photo using VGG16 trained on ImageNet
def extract_features(filename):
	model = VGG16()
	model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	features = model.predict(image, verbose=0)
	return features
 
# map an integer to a word
def mapped_word(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate a description for an image
def img_description(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = mapped_word(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text
 
# main method for calling everything, generates description for ONE image
# img = image path (e.g. 'example.jpeg')
# tokenizerpath = tokenizer path ('tokenizer.pkl')
# modelpath = model path ('model_18.h5')
def main(img, tokenizerpath, modelpath):
	tokenizer = load(open(tokenizerpath, 'rb'))
	max_length = 34
	model = load_model(modelpath)
	pic = extract_features(img)
	description = img_description(model, tokenizer, pic, max_length)
	description = description[9:-7]
	return description