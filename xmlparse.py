from xml.sax.handler import ContentHandler
from xml.sax import parse
import os
import cv2

xmlpath = '/media/ws/000F9A5700006688/TDDOWNLOAD/SUN2012pascalformat/SUN2012pascalformat/Annotations/sun_aaaatuxrpwrbvtuv.xml'
class TestHandler (ContentHandler):
#    in_headline = False
#    xmin = False
#    bndbox = False
#    ymin = False
#    xmax = False
#    ymax = False
 #   i = 0
 #   in_name = False
 #   label = ''
 #   in_right_name =False
 #   if_target = False # if we find  the right object
    in_the_right_box = False  # if we find the right box
    type_name = ''            #get object type which we want  ex. chair bed
    def __init__(self, headlines):
  #      self.in_name = in_name
        self.data = []
        self.all_data = []
#       ContentHandler.__init__(self)
#       self.headlines = headlines
#       self.data = [] def startElement(self, name, attrs):
#       if name == 'name':
#           self.in_headline = True
#       if in_headline == True and name == 'bndbox':
#           self.bnbbox =True
#       if in_headline == True and bnbbox == Tree and name = 'xmin':
#            self.xmin = True
    def set_object_type(self, type_name):
        self.type_name = type_name

    def startElement(self, name, attr):pass

       # if name == 'name' :
      #      print "====="+self.label+"========"
      #      self.in_name = True
   #         print "im here"
        #print "start element" + name
    # ad d the
    def endElement(self, name):
        # see if we are leaving the right bndbox tag or just leaving a regular
        # one
        if name == 'bndbox' and self.in_the_right_box == True:
            self.in_the_right_box = False
          #   print self.data.__len__()
  #          print self.data
       #     print 'name: '+''.join(self.data[0])
       #     print 'xmin: '+''.join(self.data[1])
       #     print 'ymin: '+''.join(self.data[2])
       #     print 'xmax: '+''.join(self.data[3])
       #     print 'ymax: '+''.join(self.data[4])
            self.all_data.append(self.data)
            self.data = []
  #      print "11"
     #   if name == 'name':
     #       print self.label
  #          print 'im also here'
         #   print ''.join(self.data)
    #        self.data = []
   #         print "22"
    #        self.in_name =False
#        print "end Element" + name
#      if name == 'name' and ''.join(self.data) == 'speaker':
#          text = ''.join(self.data)
#          self.data = []
#          self.headlines.append(text)
#          self.headlines = False
#      if xmin == True:
#          self.data = []
#          self.headlines.append(text)
#          self.xmin = False

    def get_data(self):
      #  temp_data = self.all_data
      #  self.all_data = []
      #  return temp_data
        return self.all_data
    def characters(self, string):
    #k    print 'bb'
 #       print self.in_name
#     if we find string variable contains substring "speak" , then we know
#     that we have found the right box
        self.label = string
        self.label = self.label.strip()
        self.label = self.label.strip("\n")
        self.label = self.label.strip("\r")
        if self.type_name in self.label :
           self.in_the_right_box = True
        if self.in_the_right_box:
            if self.label != '':
                self.data.append(self.label)
     #      self.data.append(string)
#        if self.in_name:
#
#            self.label = string
#            self.label = self.label.strip()
#            self.label = self.label.strip("\n")
#            print self.label
#            if self.label == 'speaker':
#                self.if_target = True;
#                print "find the speaker"
#            string = string.strip("\n")
#            self.data.append(string)
 #           print string+"aa"
#
#    if self.in_headline:
#        self.data.append(string)
#        print string
      #  print str(self.i)+ "start"+string.strip('\n')+"end"
      #  self.i += 1


xmltest = "/home/ws/test.xml"

headlines = []
#bndboxes = {}
# parse (xmltest, TestHandler(headlines))
#parse(xmlpath,TestHandler(headlines))
######t =  TestHandler(headlines)
######parse(xmlpath,t)
######my_data =  t.get_data()
#for h in headlines:
#   print h

root_path = '/media/ws/000F9A5700006688/TDDOWNLOAD/SUN2012pascalformat/SUN2012pascalformat/'
#all_files_path = '/media/ws/000F9A5700006688/TDDOWNLOAD/SUN2012pascalformat/SUN2012pascalformat/Annotations'
anno_path = os.path.join(root_path, "Annotations")
img_path = os.path.join(root_path, "JPEGImages")
#files = os.listdir(all_files_path)
#for f in files:pass
  ##  print f
#jpeg_path = root_path + 'JPEGImages/'

#img =cv2.imread(jpeg_path+'sun_aaaatuxrpwrbvtuv.jpg')
#roi = img [300:400,500:600]
#cv2.imshow("ss",img)
#cv2.imshow("aa",roi)
#cv2.waitKey(0)
num = 0
def get_object_from_pic(data, img_path, img_name, object_name):
#    print data
    global num
    for bndboxes in data:
        num += 1
        #img = cv2.imread(jpeg_path + img_name)
       # print jpeg_path + img_name
  #      cv2.imshow("aa",img)
  #      cv2.waitKey(0)
 #       print bndboxes
        img = cv2.imread(os.path.join(img_path, img_name))
        #print "======="
        #print "xmin: "+bndboxes[1]
        #print "ymin: "+bndboxes[2]
        #print "xmax: "+bndboxes[3]
        #print "ymax: "+bndboxes[4]
        roi = img[int(bndboxes[2]):int(bndboxes[4]),int(bndboxes[1]):int(bndboxes[3])]
        num = "{num}".format(num="%05d"%num)
       # print root_path+"speaker/"+"person"+num+".jpg"
        if os.path.exists(os.path.join(root_path, object_name)) == False:
            os.mkdir(os.path.join(root_path, object_name))
        cv2.imwrite(os.path.join(root_path, object_name, object_name+num+".jpg"), roi)
        num = int(num)
    #    for bndbox in bndboxes:
    #        print bndbox
    #        img = cv2.imread(jpeg_path + img_name)
    #        print bndbox[1]
    #        print bndbox[3]
    #        print bndbox[2]
    #        print bndbox[4]
    #        roi = img[int(bndbox[1]):int(bndbox[3]), int(bndbox[2]):int(bndbox[4])]
    #        num = "{num}".format(num="%04d"%num)
    #        cv2.imwrite(root_path+"person"+num+".png")
##################get_object_from_pic (my_data, jpeg_path, 'sun_aaaatuxrpwrbvtuv.jpg', 'wall')

#img = cv2.imread(jpeg_path + 'sun_aaaatuxrpwrbvtuv.jpg' )
#roi = img[100:200,100:1000]
#cv2.imshow("aaa", roi)
#cv2.waitKey(0)

def  get_all_xmlfiles(anno_path):
   xml_filenames =  os.listdir(anno_path)
   return xml_filenames


def   get_pic_by_xml_name(xmlname):
    pic_name = xmlname.replace('.xml', '.jpg')
    return pic_name


def  result_show(datas):
    for data in datas:
        print  "name: "+ data[0]
        print  "xmin: "+ data[1]
        print  "ymin: "+ data[2]
        print  "xmax: "+ data[3]
        print  "ymax: "+ data[4]

def  get_final_result(type_name):
    # get all the xml files

    file_names = get_all_xmlfiles(anno_path)

    # get target object from each xmlfile
  #  testHandler =  TestHandler(headlines)
  #  print type_name
   # testHandler.set_object_type(type_name)
    i = 0
    for file_name in file_names:
        i += 1
 #       print "=========="
        testHandler =  TestHandler(headlines)

        testHandler.set_object_type(type_name)
       # print file_name
        parse(anno_path+"/"+file_name, testHandler)
        # get the pixel of postion information from each xml
        pos_data = testHandler.get_data()
    #    print pos_data

 #       print file_name
       # print pic_name
        if len(pos_data) != 0:
        # get the pic from pos_data
            pic_name = get_pic_by_xml_name(file_name)

            get_object_from_pic(pos_data, img_path, pic_name, type_name)

  #          result_show(pos_data)
       # if i == 15:
       #     return



get_final_result('shelf')




#xml_filenames = get_all_xml_path(anno_path)
#for xml_filename in xml_filenames:
#    print get_pic_by_xml_name(xml_filename)
