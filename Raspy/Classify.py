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

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


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
  cabeceras=['Imagen','Tiempo preprocesado','Tiempo clasificacion','Tiempo ejecucion robot','Tiempo total entre bucles','Comando']
  
  vs = hiloCamara().start()
  time.sleep(2.0)
    
  myFile=open('resultados/time.csv','w')
  writer=csv.writer(myFile)
  writer.writerow(cabeceras)
  i=0;
  old=None
  for i in range(50000):
    t0 = time.time()
    image = vs.read()
    #img=cv2.resize(image,(width,height))
    t1=time.time()
    results = classify_image(interpreter, image)
    t2=time.time()
    label_id,prob = results[0]
   # print(move[label_id])
    if move[label_id] == "recto rapido":
            ap.sset_command_Clasi(command="forward", speed=55,callback=on_reading)
    if move[label_id] == "recto lento":
            ap.set_command_Clasi(command="forward", speed=40,callback=on_reading)
    if move[label_id] == "izquierda":
            ap.set_command_Clasi(command="left", speed=20,callback=on_reading)
    if move[label_id] == "derecha":
            ap.set_command_Clasi(command="right", speed=20,callback=on_reading)
    t3=time.time()
    if old is not None:
      time1=round((t1-t0)*1000)
      time2=round((t2 - t1) * 1000)
      time3=round((time.time() - t2) * 1000)
      final_time=round((t0 - old) * 1000)
      tdate=datetime.now()
      filename='resultados/imgs/image_'+ str(tdate.date())+"_%05d_%05d_%05d_%06d_%05d_orig.jpg" % (tdate.hour ,tdate.minute, tdate.second, tdate.microsecond,i)
      cv2.imwrite(filename,image)
      data=[str(filename),str(time1),str(time2),str(time3),str(final_time),move[label_id]]
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
