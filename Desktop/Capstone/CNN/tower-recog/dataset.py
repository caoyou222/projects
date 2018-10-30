import os
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_SIZE = 64
input_path = 'sample-img-color'
output_path = 'image_data'
classes = {'Turtle', 'Box', 'Fiver'}

def uniform_img():  
	for index, name in enumerate(classes):  
		class_path = input_path +"/"+ name +"/"
		count = 1
		for img_name in os.listdir(class_path):  
			if(img_name == '.DS_Store'):
				continue
			img_path = class_path + img_name  
			img = Image.open(img_path)  
			img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
			img = img.convert("L")
			out_name = name+str(count)+'.PNG'
			count = count+1
			img.save(output_path + '/' + name + '/' + out_name )

def rotate(degree):
	for index, name in enumerate(classes):  
		class_path = output_path +"/"+ name +"/"  
		for img_name in os.listdir(class_path):  
			if(img_name == '.DS_Store'):
				continue
			img_path = class_path + img_name
			rotate_name = img_name[:-4] + '_rotate' + str(degree) + '.PNG'
			#print(rotate_name)
			img = Image.open(img_path)
			im2 = img.convert('RGBA')
			im2 = im2.rotate(degree, expand=1)
			fff = Image.new('RGBA', (64,64), color='white')
			img = Image.composite(im2,fff,im2)
			img = img.convert('L')
			img.save(output_path + '/' + name + '/' + rotate_name)

def scaling(scale):
	for index, name in enumerate(classes):
		class_path = output_path +"/"+ name +"/"
		for img_name in os.listdir(class_path):  
			if(img_name == '.DS_Store'):
				continue
			img_path = class_path + img_name
			new_name = img_name[:-4] + '_scale' + str(scale) + '.PNG'
			img = Image.open(img_path) 
			new_size = int(round(IMAGE_SIZE*scale))
			margin = int(round((new_size-IMAGE_SIZE)/2))
			img = img.resize((new_size,new_size))
			img = img.crop((margin, margin, margin+IMAGE_SIZE, margin+IMAGE_SIZE))
			img.save(output_path + '/' + name + '/' + new_name)

def create_record():
	writer = tf.python_io.TFRecordWriter("tower.tfrecords")
	for index, name in enumerate(classes):  
		class_path = output_path +"/"+ name +"/"
		for img_name in os.listdir(class_path):  
			if(img_name == '.DS_Store'):
				continue
			img_path = class_path + img_name  
			img = Image.open(img_path)  
			img_raw = img.tobytes()
			example = tf.train.Example(
				features = tf.train.Features(
					feature = {
						"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),#0 = Turtle, 1 = Box, 2 = Fiver
						'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
					}))
			writer.write(example.SerializeToString())
	writer.close()

if __name__ == '__main__':  
    uniform_img()
    scaling(1.2)
    rotate(30)
    rotate(90)
    rotate(15)
    create_record()
    
