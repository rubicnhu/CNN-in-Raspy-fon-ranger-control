#python3  Classify.py \--model move4_clas.tflite \
import cv2
import csv
import argparse
import io
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import tensorflow as tf
from PIL import Image
from aurigapy.aurigapy import *
import time
from datetime import datetime
from hiloCamara import hiloCamara
import sys

def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def on_reading(value, timeout):
    # TODO: it has to fail to acquire image at 33 miliseconds
    print("%r > %r (%r)" % (timestamp(), value, timeout))

    

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  return output


def main(ap):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  args = parser.parse_args()

    
  
  interpreter = tf.lite.Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  move=['recto rapido','derecha','izquierda','recto lento']
  cabeceras=['Imagen','Tiempo preprocesado','Tiempo regresion','Tiempo ejecucion robot','Tiempo total entre bucles','Angulo']
  
  vs = hiloCamara().start()
  time.sleep(2.0)
  
  myFile=open('resultadosRegresion/time.csv','w')
  writer=csv.writer(myFile)
  writer.writerow(cabeceras)
  old=None
  for i in range(20000):
    t0 = time.time()  
    image = vs.read()
    imgs=np.array(image)/255.
    t1=time.time()
    results = classify_image(interpreter, imgs)
    t2=time.time()
    angle=int(results*180)
    ap.set_command_Regre( angles=angle,speed=100,offset=5, callback=on_reading)
    t3=time.time()
   # print(angle)
    if old is not None:
      time1=round((t1-t0)*1000)
      time2=round((t2 - t1) * 1000)
      time3=round((time.time() - t2) * 1000)
      final_time=round((t0 - old) * 1000)
      tdate=datetime.now()
      filename='resultadosRegresion/imgs/image_'+ str(tdate.date())+"_%05d_%05d_%05d_%06d_%05d_orig.jpg" % (tdate.hour ,tdate.minute, tdate.second, tdate.microsecond,i)
      cv2.imwrite(filename,image)
      data=[str(filename),str(time1),str(time2),str(time3),str(final_time),angle]
      writer.writerow(data)
      #print('tiempo imagen ',time1)
      #print('tiempo clasificacion ',time2)
      #print('tiempo robot ',time3)
      #print('tiempo final entre bucles ',final_time)
    old=t0 
   




if __name__ == '__main__':
  try:
        #robot_start()
    ap = AurigaPy(debug=False)
    bluetooth = "/dev/tty.Makeblock-ELETSPP"
    usb = "/dev/ttyUSB0"

    print(" Conectando..." )
    ap.connect(usb)
    print(" Conectado!" )
    time.sleep(0.2)
    ap.set_command(command="forward", speed=0, callback=on_reading) 
    main(ap)
  except KeyboardInterrupt:
    print('parado')
    ap.set_command(command='forward',speed=0, callback=on_reading)
    ap.reset_robot()
    ap.close()
