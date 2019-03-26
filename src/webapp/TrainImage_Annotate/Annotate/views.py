import datetime
import json

import xlwt as xlwt
from django.db.models import Q
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import os
import sys
import zipfile
from io import BytesIO  # for python 3
import numpy as np
import cv2
import shutil
from django.conf import settings
import tensorflow as tf
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, Dropout
from keras.layers.core import Lambda, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import time
from PIL import Image
from Annotate.models import *
import pandas as pd
from django.contrib import messages
from django import forms
from keras.applications.xception import Xception
import keras.regularizers as regularizers
import keras.optimizers as optimizers
#from ldap3.core.exceptions import LDAPException, LDAPOperationsErrorResult, LDAPOperationResult
#from ldap3 import Server, Connection, SIMPLE, SYNC, ALL, SASL, NTLM, ANONYMOUS, SUBTREE, SCHEMA, ALL_ATTRIBUTES, ALL_OPERATIONAL_ATTRIBUTES 


K._LEARNING_PHASE = tf.constant(0)
K.clear_session()
img_width = 512
img_height = 512
adam_10_3 = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


# modelB = ModelB()


@csrf_exempt
def login(request):
	return render(request, 'login.html', {})


def logout(request):
	return render(request, 'login.html', {})


@csrf_exempt
def home(request):
	return render(request, 'homepage.html', {})


@csrf_exempt
def checklogin(request):

	username = request.POST['login']
	password = request.POST['password']

	user = None
	# connection, message =is_valid_Sidra_Email(username,password)

	# email = username+'@sidra.org'
	# print(message)
	# if not connection:
	# 	messages.success(request, ('Please check your username and password'))
	# 	return HttpResponseRedirect("/ai/login/")
	return render(request, 'homepage.html', {})

def is_valid_Sidra_Email (username=None,password=None):
		if password == None or password == '':
		  return  None, 'Password cannot be empty'
		try:
			s = Server('smrc.sidra.org', get_info=ALL )
			c = Connection(s, user='SMRC\\'+username , password=password)
			if c.bind():
				if c.result['description'].lower() == "success".lower():
					return c, 'Success'
			else:
				
				if c.result['description'].lower() == "invalidCredentials".lower():
					return None, 'Wrong Sidra username or password'
		except:
			return  None, 'LDAP Connection error'



def baseline_model(img_size = 299):
	base_model =Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
	x = Dropout(0.5)(x)
	predictions = Dense(2, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)
	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all layers of the based model that is already pre-trained.
	#model.load_weights (old_best_weights)
	for layer in base_model.layers:
		layer.trainable = False
	
	return model


# # Save the default graph for use in another thread.
# # This is a tensorflow quirk.
graph = tf.get_default_graph()
model = None

@csrf_exempt
def modelTest(request):
	return render(request, 'download.html', {'evalBool': False})


@csrf_exempt
def modelTestFolder(request):
	return render(request, 'download_folder.html', {'evalBool': False})


@csrf_exempt
def delete_all_folders():
	if os.path.exists('Annotate/static/input_files/'):
		shutil.rmtree("Annotate/static/input_files/")

	if os.path.exists('Annotate/static/output_files/'):
		shutil.rmtree("Annotate/static/output_files/")


def create_folders():
	if not os.path.exists('Annotate/static/input_files/'):
		os.mkdir('Annotate/static/input_files/')

	if not os.path.exists('Annotate/static/output_files/'):
		os.mkdir('Annotate/static/output_files')


def save_images_locally(files, isFolder):
	from PIL import Image
	from scipy import ndimage, misc

	if isFolder:
		file_list = files.getlist('folder_pic')
	else:
		file_list = files.getlist('pic')

	for file in file_list:
		try:
			with open('Annotate/static/input_files/input_image_' + str(file), 'wb+') as destination:
				for chunk in file.chunks():
					destination.write(chunk)
		except IOError:
			print("not an image file")


'''
	Set the image order for the user based on their username.
	Choose a custom filter, query the database and append it to the 
	list to make sure the image ordered is maintained.
'''


def updateImageOrder(request):
	from django.db import transaction

	# Filter_1 : Ground_truth == 0 and Models_Prediction == 1 and Type= 'Validation' # 340 Files
	# Filter_2 : Ground_truth == 0 and Models_Prediction == 1 and Type= 'Test' # 290 Files
	# Filter_3 : Ground_truth == 1 and Models_Prediction == 0 and Type= 'Validation' # 289 Files
	# Filter_4 : Ground_truth == 1 and Models_Prediction == 0 and Type= 'Test' # 204 Files

	email_to_update = "dkau12ra@sidra.org"
	# some custom filters
	FILTER_1 = {'type_folder': 'Validation', 'ground_truth': 0, 'model_truth': 1}
	FILTER_2 = {'type_folder': 'Test', 'ground_truth': 0, 'model_truth': 1}
	FILTER_3 = {'type_folder': 'Validation', 'ground_truth': 1, 'model_truth': 0}
	FILTER_4 = {'type_folder': 'Test', 'ground_truth': 1, 'model_truth': 0}
	FILTER_5 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b05', 'ground_truth': 1, 'model_truth': 0}
	FILTER_6 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b06', 'ground_truth': 1, 'model_truth': 0}
	FILTER_7 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b01', 'ground_truth': 0, 'model_truth': 0}
	FILTER_8 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b01', 'ground_truth': 0, 'model_truth': 0}
	FILTER_9 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b01', 'ground_truth': 0, 'model_truth': 0}
	FILTER_10 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b01', 'batch_number': 'b02', 'ground_truth': 0,
				 'model_truth': 0}
	FILTER_11 = {'type_folder': 'PA_1_ALL', 'batch_number': 'b01', 'batch_number': 'b04', 'ground_truth': 0,
				 'model_truth': 0}

	image_filter3 = ImageModel.objects.filter(**FILTER_3)
	batch_3 = list(image_filter3)

	user = UserModel.objects.filter(Q(username=email_to_update))

	current_user_id = ''
	for temp in user:
		current_user_id = temp.id

	if user:
		# query the database to get the images based on the filter
		image_filter4 = ImageModel.objects.filter(**FILTER_4)
		image_filter3 = ImageModel.objects.filter(**FILTER_3)
		image_filter2 = ImageModel.objects.filter(**FILTER_2)
		image_filter1 = ImageModel.objects.filter(**FILTER_1)
		idx = 0
		combined = list(image_filter3) + list(image_filter2) + list(image_filter1) + list(image_filter4)


		# set all the image index for the selected user to be null
		if current_user_id == 1:
			ImageOrder.objects.all().update(reviewer1=None)
		elif current_user_id == 2:
			ImageOrder.objects.all().update(reviewer2=None)
		elif current_user_id == 3:
			ImageOrder.objects.all().update(reviewer3=None)
		elif current_user_id == 4:
			ImageOrder.objects.all().update(reviewer4=None)
		elif current_user_id == 5:
			ImageOrder.objects.all().update(reviewer5=None)

		# autocommit does not happen in every save. One big commit after all images
		# are required image indexes are updated
		with transaction.atomic():
			for objs in combined:

				if current_user_id == 1:
					ImageOrder.objects.filter(img_id=objs.id).update(reviewer1=idx)
				elif current_user_id == 2:
					ImageOrder.objects.filter(img_id=objs.id).update(reviewer2=idx)
				elif current_user_id == 3:
					ImageOrder.objects.filter(img_id=objs.id).update(reviewer3=idx)
				elif current_user_id == 4:
					ImageOrder.objects.filter(img_id=objs.id).update(reviewer4=idx)
				elif current_user_id == 5:
					ImageOrder.objects.filter(img_id=objs.id).update(reviewer5=idx)
				idx = idx + 1
	return render(request, "update_order.html", {})


'''
	Predict images from the uploaded folder using the model
'''


def evaluateFolderImg(file, request):

	global graph
	with graph.as_default():

		filenames = []
		prediction = []
		prediction_class = []
		image_path_out = ""

		# delete any previous stored images from the folder
		delete_all_folders()
		create_folders()
		# save the folder images to a folder on the server for prediction 
		# purposes
		save_images_locally(request, True)

		# folder where the input images are stored
		imagePaths = "Annotate/static/input_files/"
		temp_filenames = []
		img_array = []

		# load the images and convert the to an array
		for imagePath in os.listdir(imagePaths):
			temp_filenames.append(imagePath)
			input_path = os.path.join(imagePaths, imagePath)
			img = load_img (input_path , target_size=(img_width,img_height))
			# img.resize(img_width, img_height)
			img_array_single = img_to_array(img)
			img_array_single /=255 
			img_array.append(img_array_single)

		# create a folder to store the predicted images
		if not os.path.exists('Annotate/static/output_files/Normal'):
			os.mkdir('Annotate/static/output_files/Normal')

		if not os.path.exists('Annotate/static/output_files/Abnormal'):
			os.mkdir('Annotate/static/output_files/Abnormal')

		img_array = np.array(img_array)

		img_array.reshape(len(img_array), 512,512,3)
		# get the set of images
		res = model.predict(img_array)
	
		total_count = 0
		total_normal = 0
		total_abnormal = 0
		normal_path = 'Annotate/static/output_files/Normal/'
		abnormal_path = 'Annotate/static/output_files/Abnormal/'
		file_list = request.getlist('folder_pic')
		index = 0

		# move the files to specific folder ased on their respective predicted threshold
		for file in file_list:
			
			total_count = total_count + 1
			if res[index][1] < 0.29434050:
				image_path_out = 'Annotate/static/output_files/Normal/output_file_'+str(temp_filenames[index])
				total_normal = total_normal + 1
				prediction_class.append("Normal")
				
				with open('Annotate/static/output_files/Normal/output_file_'+str(temp_filenames[index]), 'wb+') as destination:
					for chunk in file.chunks():
						destination.write(chunk)
			else:
				image_path_out = 'Annotate/static/output_files/Abnormal/output_file_'+str(temp_filenames[index])
				total_abnormal = total_abnormal + 1
				prediction_class.append("Abnormal")
				
				with open('Annotate/static/output_files/Abnormal/output_file_'+str(temp_filenames[index]), 'wb+') as destination:
					for chunk in file.chunks():
						destination.write(chunk)

			
			filenames.append(image_path_out)
			prediction.append(res[index][0])
			index = index + 1

		return total_count,total_normal,total_abnormal, normal_path, abnormal_path


'''
	Load the images when the user clicks on a the Normal button
	after folder prediction
'''


def loadNormalImages(request):
	path = 'Annotate/static/output_files/Normal/'
	analytics_list = os.listdir(path)
	return render(request, 'list_files.html', {'analytics': analytics_list})


'''
	Load the images when the user clicks on a the Abnormal button
	after folder prediction
'''


def loadAbnormalImages(request):
	path = 'Annotate/static/output_files/Abnormal/'
	analytics_list = os.listdir(path)
	return render(request, 'list_files_abnormal.html', {'analytics': analytics_list})


'''
	Run the model on a single image
'''


def evaluateSingleImg(file, request):
	
	global graph

	# K.set_session(session1)
	with graph.as_default():
		# tf.initialize_all_variables()
		filenames = []
		prediction = []
		prediction_class = []

		# delete the previous iamge in the folder, if any.
		delete_all_folders()
		create_folders()
		#  save the image to the local folder on the server
		save_images_locally(request, False)
		
		
		Image_path = 'Annotate/static/input_files/input_image_' + str(file)
		Image_path_out = '/static/input_files/input_image_' + str(file)
		Image_path_cam = '/static/input_files/input_image_cam_' + str(file)
		# load the image and convert it to an array
		filenames.append(Image_path_out)
		filenames.append(Image_path_cam)
		full_size_img = load_img(Image_path)
		img = load_img (Image_path, target_size=(img_width,img_height))
		img_array = img_to_array(img)

		# get the last layer from the convnet. this is to generate
		# the Class Activation Mapping(CAM) image of the X-ray

		
		Last_conv_layer_name = model.layers[131].name
		cam_img, confidence = get_cam (Image_path_cam, img_array,full_size_img,model,Last_conv_layer_name,2048)

			 
		y = confidence
		# set the confidence level and the label basedd on the threshold
		if y < 0.29434050:
			z = (1-y)*100
			prediction.append(round(z,2))
			prediction_class.append("Normal")
		else:
			z = y * 100
			prediction.append(round(z,2))
			prediction_class.append("Abnormal")

		return prediction, prediction_class, filenames


@csrf_exempt
def imgEval(request):
	
	model_type = request.POST['type_text']

	if model_type == 'boneage':
		gender = request.POST['genradio']
		age = request.POST['age']
	# else:
	# 	K.clear_session()
	# 	K.set_learning_phase(False)
	# 	model = baseline_model(128)
	# 	model.load_weights('../../../data/110_xception_softmax_04_D1_1515_512_temp_9141_8828_9454.hdf5')
	# 	model.compile(optimizer=adam_10_3, loss='binary_crossentropy',metrics=['accuracy'])
	# 	model._make_predict_function()
	# 	graph = tf.get_default_graph()

	if "folder_pic" in request.FILES:
		total_count, total_normal, total_abnormal, normal_path, abnormal_path = evaluateFolderImg(
			request.FILES['folder_pic'], request.FILES)
		return render(request, 'upload.html',
					  {'total_count': total_count, 'total_normal': total_normal, 'total_abnormal': total_abnormal,
					   'evalBool': True, 'folder': True, 'normal_path': normal_path, 'abnormal_path': abnormal_path, 'type':model_type})
	else:
		prediction, prediction_class, filenames = evaluateSingleImg(request.FILES['pic'], request.FILES)
		return render(request, 'upload.html',
					  {'prediction_class': prediction_class, 'predictions': prediction, 'filenames': filenames,
					   'evalBool': True, 'folder': False, 'type': model_type})
	
	return render(request, 'upload.html',
				  {'prediction_class': [], 'predictions': [], 'filenames': [], 'evalBool': False, 'folder': False, 'type':model_type})


@csrf_exempt
def uploadType(request, type):
	global model, graph
	if type == 'xray':
		K.clear_session()
		K.set_learning_phase(False)
		model = baseline_model(128)
		model.load_weights('../../../data/110_xception_softmax_04_D1_1515_512_temp_9141_8828_9454.hdf5')
		model.compile(optimizer=adam_10_3, loss='binary_crossentropy',metrics=['accuracy'])
		model._make_predict_function()
		graph = tf.get_default_graph()
	return render(request, 'upload.html', {'type':type})


def get_cam(Image_path_cam, x ,org_img, model , layer, layer_size ):
	
	x1 = np.expand_dims(x, axis=0)
	x1 /=255
	preds = model.predict(x1)
	preds_index = np.argmax(preds[0])
	model_prediction = preds[0][1]
	#print (preds_index)
	image_output = model.output[:, 1]
	#print(model.output[:, 1])
	# The is the output feature map of the last convolutional layer in VGG16
	last_conv_layer = model.get_layer(layer)
	#print(last_conv_layer.output[0])

	# This is the gradient of the "xchest" class with regard to
	# the output feature map of last convolutional layer
	grads = K.gradients(image_output, last_conv_layer.output)[0]

	# This is a vector of shape (512,), where each entry
	# is the mean intensity of the gradient over a specific feature map channel
	pooled_grads = K.mean(grads, axis=(0, 1, 2))

	# This function allows us to access the values of the quantities we just defined:
	# `pooled_grads` and the output feature map of `block5_conv3`,
	# given a sample image
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

	# These are the values of these two quantities, as Numpy arrays,
	# given our sample image of two elephants
	
	pooled_grads_value, conv_layer_output_value = iterate([x1])

	# We multiply each channel in the feature map array
	# by "how important this channel is" with regard to the elephant class
	for i in range(layer_size):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

	# The channel-wise mean of the resulting feature map
	# is our heatmap of class activation
	heatmap = np.mean(conv_layer_output_value, axis=-1)
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)

	heatmap = cv2.resize(heatmap, (org_img.size[0], org_img.size[1]))
	# We convert the heatmap to RGB
	heatmap = np.uint8(255 * heatmap)

	# We apply the heatmap to the original image
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	# 0.4 here is a heatmap intensity factor
	#plt.imshow(x)
	#plt.show()
	superimposed_img = heatmap * 0.6 * model_prediction + image.img_to_array(org_img)
	#superimposed_img = heatmap * 0.4 + image.img_to_array(org_img)
	# Save the image to disk
	# os.remove('Annotate/static/input_files/input_image_superimposed_img.jpg')
	cv2.imwrite('Annotate'+Image_path_cam, superimposed_img)
	img = image.load_img('Annotate'+Image_path_cam)

	return img , model_prediction


'''
	admin function to load all the images into the DB
'''
@csrf_exempt
def loadAllImagesDB(request):
	from itertools import islice
	from django.db import transaction
	# read the csv file that contains the list of images
	data = pd.read_csv('/home/baljit92/Annotation_Phase_01.csv')
	image_list = []

	ImageModel.objects.all().delete()
	with transaction.atomic():
		# # get the type of PA image
		for index, row in data.iterrows():
			image_type = 0

			if row['File_name'].find("pa.jpg") != -1:
				image_type = 1
			elif row['File_name'].find("pa_2.jpg") != -1:
				image_type = 2
			elif row['File_name'].find("pa_3.jpg") != -1:
				image_type = 3
			elif row['File_name'].find("pa_4.jpg") != -1:
				image_type = 4
			elif row['File_name'].find("pa_5.jpg") != -1:
				image_type = 5

			# store the image
			image_data = ImageModel(
				type_folder=str(row['Type']),
				ground_truth=int(row['Ground_truth']),
				model_truth=int(row['Models_Prediction']),
				model_max=float(row['Models_Max']),
				model_min=float(row['Models_Min']),
				model_avg=float(row['Models_Average']),
				file_name=str(row['File_name']),
				image_type=image_type
			)
			image_list.append(image_data)

		

			image_data.save()

	# this method creates indices with image ids (required)
	img_idx_list = []
	ImageOrder.objects.all().delete()
	# image_order_items = ImageOrder.objects.all()
	image_order_items = ImageModel.objects.all()
	idx = 0
	with transaction.atomic():
		if image_order_items:
			for obj in image_order_items:

				if obj.id is not None:
					image_data = ImageOrder(
						img_id=obj.id,
						reviewer1=idx,
						reviewer2=idx,
						reviewer3=idx,
						reviewer4=idx,
						reviewer5=idx
					)
					# obj.save()
					img_idx_list.append(image_data)
					idx = idx + 1

					image_data.save()

	return render(request, "update_order.html", {})


def saveUser(request):
	# ---- SAVE USER TO USERMODEL ----
	user1 = UserModel(
		username="amro@1qbit.com",
		img_array=json.dumps([])
	)

	user1.save()

	return render(request, "update_order.html", {})


@csrf_exempt
def annotationData(request):
	return render(request, 'annotation_data.html')


''' 
	Helper method to dind the nth occurence of a 
	character in a string.
'''


def findnth(haystack, needle, n):
	parts = haystack.split(needle, n + 1)
	if len(parts) <= n + 1:
		return -1
	return len(haystack) - len(parts[-1]) - len(needle)


'''
	Save the annotated result in the database

'''

@csrf_exempt
def annotationDataPost(request):
	message = 'error'
	result = False
	# email = request.POST['username'] + '@sidra.org'
	email = 'bsingh@sidra.org'
	if request.POST['file_name'] and request.POST['username']:
		#  save the result in the AnnotationData table.
		#  TODO: Annotation should only be saved in the ImageModel table
		annotation = AnnotationData.objects.filter(Q(file_name=request.POST['file_name']), Q(username=email))
		if annotation:
			for an in annotation:
				an.annotation_result = int(request.POST['annotation_result'])
				
				cords = json.loads(request.POST['cords'])
				an.poly_cords.clear()
				for cord in cords:
					polygon = PolygonCoordinates(
						username=request.POST['username'],
						cords=json.dumps(cord)
					)
					polygon.save()
					an.poly_cords.add(polygon)


				if len(cords) == 0:
					an.poly_cords.clear()
				an.save()

		else:
			annotation = AnnotationData(
				username=email,
				file_name=request.POST['file_name'],
				annotation_result=int(request.POST['annotation_result']),
			)
			annotation.save()

			for cord in json.loads(request.POST['cords']):
				polygon = PolygonCoordinates(
					username=request.POST['username'],
					cords=json.dumps(cord)
				)
				polygon.save()
				annotation.poly_cords.add(polygon)

			annotation.save()


		user = UserModel.objects.filter(Q(username=email))
		current_user_id = 0
		if user:
			for temp in user:
				current_user_id = temp.id

		filename_start = findnth(request.POST['file_name'], "/", 1)
		filename = request.POST['file_name'][(filename_start + 1):]

		# save the annotation result in the reviewer column for
		# the given image
		img_metadata = ImageModel.objects.filter(file_name=filename)

		for ob in img_metadata:
			if img_metadata:
				if current_user_id == 1:
					ob.reviewer1 = int(request.POST['annotation_result'])
				elif current_user_id == 2:
					ob.reviewer2 = int(request.POST['annotation_result'])
				elif current_user_id == 3:
					ob.reviewer3 = int(request.POST['annotation_result'])
				elif current_user_id == 4:
					ob.reviewer4 = int(request.POST['annotation_result'])
				elif current_user_id == 5:
					ob.reviewer5 = int(request.POST['annotation_result'])
				ob.save()

		message = 'saved'

	response = HttpResponse(json.dumps({'message': message}),
							content_type='application/json')
	response.status_code = 200
	return response


'''
	When the annotation page loads the first time, 
	this function is called to load the first image
	for the signed ni user that is not annotated
'''


@csrf_exempt
def getNextPath(request):
	global graph
	with graph.as_default():
		path = ''
		cords = []
		i = 0
		# get the user 
		# email = request.GET['username']+'@sidra.org'
		email = 'bsingh@sidra.org'
		user = UserModel.objects.filter(Q(username=email))

		current_user = ''
		current_user_id = 0

		if user:
			for temp in user:
				current_user = temp.username
				current_user_id = temp.id

			file_name_list = []
			user_image_index_list = []

			file_name_json = None
			image_order_objects = None
			
			# exclude all the image indices that are NULL. Images not to be shown
			if current_user_id == 1:
				image_order_objects = ImageOrder.objects.order_by('reviewer1').all().exclude(reviewer1__isnull=True)
			elif current_user_id == 2:
				image_order_objects = ImageOrder.objects.order_by('reviewer2').all().exclude(reviewer2__isnull=True)
			elif current_user_id == 3:
				image_order_objects = ImageOrder.objects.order_by('reviewer3').all().exclude(reviewer3__isnull=True)
			elif current_user_id == 4:
				image_order_objects = ImageOrder.objects.order_by('reviewer4').all().exclude(reviewer4__isnull=True)
			elif current_user_id == 5:
				image_order_objects = ImageOrder.objects.order_by('reviewer5').all().exclude(reviewer5__isnull=True)


			reviewer = None
			if image_order_objects:
				for obj in image_order_objects:
					# set the reviewer
					if reviewer is None:
						if current_user_id == 1:
							reviewer = obj.reviewer1
						elif current_user_id == 2:
							reviewer = obj.reviewer2
						elif current_user_id == 3:
							reviewer = obj.reviewer3
						elif current_user_id == 4:
							reviewer = obj.reviewer4
						elif current_user_id == 5:
							reviewer = obj.reviewer5
					# create a list of full file names that are to be annotated
					if reviewer is not None:
						img_file = ImageModel.objects.filter(id=obj.img_id)

						if img_file:
							for idx in img_file:
								img_full_path = idx.file_name
								file_name_list.append(img_full_path)

				# convert the list to a string to be stored in the database
				file_name_json = json.dumps(file_name_list)
				for temp in user:
					temp.img_array = file_name_json
					temp.save()

				#  convert the string to an array
				files = json.loads(file_name_json)

				# get the first image that is not annotated
				for file in files:
					i = i + 1 
					exist = AnnotationData.objects.filter(Q(file_name=file), Q(username=email))
					if not exist:
						path = '/static/media/' + str(file)
						annotations = AnnotationData.objects.all().filter(file_name=file)
						if annotations:
							for annotation in annotations:
								for cord in annotation.poly_cords.all():
									cords.append(json.loads(cord.cords))

						break;
				if path:
					
					img_file_str = str(file)
					image_name = '/static/input_files/input_image_cam_'+img_file_str[(img_file_str.index("/")+1):]
					

					Image_path = "Annotate"+path
					full_size_img = load_img(Image_path)
					img = load_img (Image_path, target_size=(img_width,img_height))
					img_array = img_to_array(img)
					
					
					if not os.path.exists('Annotate'+image_name) :
						# 	# get the last layer from the convnet. this is to generate
					# 	# the Class Activation Mapping(CAM) image of the X-ray
						Last_conv_layer_name = model.layers[131].name
						cam_img, confidence = get_cam (image_name, img_array,full_size_img,model,Last_conv_layer_name,2048)

					response = HttpResponse(json.dumps({'path': path, 'file_name':file, 'current_index':str(i), 'total_files':str(len(files)), 'username':request.GET['username'], 'cords':cords}),
											content_type='application/json')
				else:
					response = HttpResponse(json.dumps({'path': '', 'file_name':'', 'current_index':str(i), 'total_files':str(len(files)), 'username':request.GET['username']}),
											content_type='application/json')
				response.status_code = 200
			else:
				response = HttpResponse("Failure",
											content_type='application/json')
				response.status_code = 400
			return response



'''
	This method allows the user to scroll the images
	by adding the functionality of going to the previous
	image
'''


@csrf_exempt
def getPrevPath(request):
	global graph
	with graph.as_default():
		path = ''
		prev_img = ''
		cords = []
		current_img = request.GET['current_img'];
		email = 'bsingh@sidra.org'
		# email = request.GET['username'] + '@sidra.org'
		user = UserModel.objects.filter(Q(username=email))

		current_user_img = ""
		for temp in user:
			current_user_img = temp.img_array

		files = json.loads(current_user_img)

		i = 0;
		
		for file in files:
			if current_img == file:
				break;
			else:
				i = i + 1
				prev_img = file
				path = '/static/media/' + str(file)

		# only load the previous image is it has already been annotated
		if path:
			exist = AnnotationData.objects.filter(Q(file_name=prev_img), Q(username=email))
		
			annotations = AnnotationData.objects.all().filter(Q(file_name=prev_img) , Q(username=email))
			if annotations:
				for annotation in annotations:
					for cord in annotation.poly_cords.all():
						cords.append(json.loads(cord.cords))

			if exist:
				for annotation in exist:

				
					# img_metadata = ImageModel.objects.filter(file_name=annotation.file_name).values()
					# check  = AnnotationData.objects.filter(Q(file_name=prev_img), Q(username=email))
					# print(check)

					# for temp in img_metadata:

					#     poly_coords = PolygonCoordinates.objects.filter(id = temp.id)

					img_file_str = str(prev_img)
					image_name = '/static/input_files/input_image_cam_' + img_file_str[(img_file_str.index("/") + 1):]

					Image_path = "Annotate/static/media/" + prev_img
					full_size_img = load_img(Image_path)
					img = load_img(Image_path, target_size=(img_width, img_height))
					img_array = img_to_array(img)
					
					if not os.path.exists('Annotate'+image_name) :
						# 	# get the last layer from the convnet. this is to generate
					# 	# the Class Activation Mapping(CAM) image of the X-ray
						Last_conv_layer_name = model.layers[131].name
						cam_img, confidence = get_cam (image_name, img_array,full_size_img,model,Last_conv_layer_name,2048)
				
					response = HttpResponse(json.dumps(
					  {'path': path, 'cords': cords, 'file_name': prev_img, 'current_index': str(i), 'total_files': str(len(files)),
						'username': request.GET['username'], 'annotation':str(annotation.annotation_result)}),
							content_type='application/json')

			else:
				response = HttpResponse(json.dumps(
					{'path': '', 'file_name': ' ', 'current_index': str(i), 'total_files': str(len(files)),
					 'username': request.GET['username'], 'annotation': ''}),
										content_type='application/json')

		else:
			response = HttpResponse(json.dumps(
				{'path': '', 'file_name': '', 'username': '', 'current_index': str(i), 'total_files': str(len(files)),
				 'annotation': ''}),
									content_type='application/json')
		response.status_code = 200
		return response


'''
	This method allows the user to scroll the images
	by adding the functionality of going to the next
	image
'''


@csrf_exempt
def getForwardPath(request):
	global graph
	with graph.as_default():
		path = ''
		next_img = ''
		current_img = request.GET['current_img'];
		cords = []
		email = 'bsingh@sidra.org'
		# email = request.GET['username']+'@sidra.org'
		user = UserModel.objects.filter(Q(username=email))

		current_user_img = ""
		for temp in user:
			current_user_img = temp.img_array

		files = json.loads(current_user_img)

		i = 0
		for idx in range(0, len(files)):
			
			if files[idx] == current_img:
				if (idx+1) < len(files):
					next_img = files[(idx+1)]
					i = (idx + 1)   
					idx = len(files)
					path =  '/static/media/' + str(next_img)
					
				else:
					i = idx
					idx = len(files)
					path = ''
		

		if path:
			exist = AnnotationData.objects.filter(Q(file_name=current_img), Q(username=email))
			next_img_exist = AnnotationData.objects.filter(Q(file_name=next_img), Q(username=email))
			annotations = AnnotationData.objects.all().filter(Q(file_name=next_img), Q(username=email))
			if annotations:
				for annotation in annotations:
					for cord in annotation.poly_cords.all():
						cords.append(json.loads(cord.cords))
			
			if exist and next_img_exist:
				# load the image and convert it to an array
					
				for annotation in next_img_exist:

					img_file_str = str(next_img)
					image_name = '/static/input_files/input_image_cam_'+img_file_str[(img_file_str.index("/")+1):]

					Image_path = "Annotate/static/media/"+next_img
					full_size_img = load_img(Image_path)
					img = load_img (Image_path, target_size=(img_width,img_height))
					img_array = img_to_array(img)

					if not os.path.exists('Annotate'+image_name) :
						# 	# get the last layer from the convnet. this is to generate
						# the Class Activation Mapping(CAM) image of the X-ray
						Last_conv_layer_name = model.layers[131].name
						cam_img, confidence = get_cam (image_name, img_array,full_size_img,model,Last_conv_layer_name,2048)

					response = HttpResponse(json.dumps({'path': path,'cords':cords, 'file_name':next_img, 'current_index':str(i+1), 'total_files':str(len(files)), 'username':request.GET['username'], 'annotation':str(annotation.annotation_result)}),
									content_type='application/json')
			elif exist: 
				img_file_str = str(next_img)
				image_name = '/static/input_files/input_image_cam_'+img_file_str[(img_file_str.index("/")+1):]

				Image_path = "Annotate/static/media/"+next_img
				full_size_img = load_img(Image_path)
				img = load_img (Image_path, target_size=(img_width,img_height))
				img_array = img_to_array(img)

				if not os.path.exists('Annotate'+image_name) :
						# 	# get the last layer from the convnet. this is to generate
						# the Class Activation Mapping(CAM) image of the X-ray
						Last_conv_layer_name = model.layers[131].name
						cam_img, confidence = get_cam (image_name, img_array,full_size_img,model,Last_conv_layer_name,2048)

				response = HttpResponse(json.dumps({'path': path, 'cords':cords, 'file_name':next_img, 'current_index':str(i+1), 'total_files':str(len(files)), 'username':request.GET['username'], 'annotation':''}),
									content_type='application/json')
			else: 
				if i < (len(files) - 1):
					response = HttpResponse(json.dumps({'path': '', 'file_name':'', 'current_index':str(i), 'total_files':str(len(files)), 'username':request.GET['username'], 'annotation':''}),
									content_type='application/json')
				else:
					response = HttpResponse(json.dumps({'path': '', 'file_name':'', 'current_index':str(i), 'total_files':str(len(files)), 'username':request.GET['username'], 'annotation':''}),
									content_type='application/json')

		else:
			response = HttpResponse(json.dumps({'path': '', 'file_name':'', 'username':'', 'current_index':str(i+1), 'total_files':str(len(files)), 'annotation':''}),
									content_type='application/json')
		response.status_code = 200
		return response


'''
	Export the annotation file
	TODO: This URL is not required

'''


def exportAnnotation(request):
	http_status = 400
	status = False
	data = {}
	message = ""

	annotations = AnnotationData.objects.all()

	# xl_file_obj = StringIO.StringIO() # for python 2.7
	xl_file_obj = BytesIO()  # for python 3
	book = xlwt.Workbook()

	details_sheet = book.add_sheet("Details")

	details_sheet.write(0, 0, "ID")
	details_sheet.write(0, 1, "Username")
	details_sheet.write(0, 2, "Date")
	details_sheet.write(0, 3, "File Name")
	details_sheet.write(0, 4, "Annotation Result")

	row_i = 1
	for annotation in annotations:
		details_sheet.write(row_i, 0, str(annotation.id))

		details_sheet.write(row_i, 1, str(annotation.username))

		details_sheet.write(row_i, 2, str(annotation.date))

		details_sheet.write(row_i, 3, str(annotation.file_name))

		details_sheet.write(row_i, 4, str(annotation.annotation_result))

		row_i += 1

	book.save(xl_file_obj)

	xl_file_obj.seek(0)

	response = HttpResponse(xl_file_obj.read(), content_type="application/ms-excel")
	response['Content-Disposition'] = "attachment; filename=annotation_export_%s.xls" % (
		str(datetime.datetime.now().date()))

	return response
