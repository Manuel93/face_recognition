import face_recognition as fr
from PIL import Image, ImageDraw
import os
import cv2

from tkinter import *
import tkinter.font as tkFont
from tkinter import filedialog

from PIL import ImageTk,Image

import time
import random
import json

import numpy as np

class FileManagement(object):

    def get_file_list(self,path):
    
        files = []
        file_paths = []
        file_sizes = []
        file_created = []
        file_changed = []

        file_list = []


        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                file_list.append([])

                files.append(file)
                file_list[-1].append(file)

                file_path = os.path.join(r, file).replace("\\","/")
                file_paths.append(file_path)
                file_list[-1].append(file_path)

                size = os.path.getsize(file_path)
                file_sizes.append(size)
                file_list[-1].append(size)

                time_created = os.path.getctime(file_path)
                time_created_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_created))
                file_created.append(time_created_str)
                file_list[-1].append(time_created_str)

                time_changed = os.path.getmtime(file_path)
                time_changed_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_changed))
                file_changed.append(time_changed_str)
                file_list[-1].append(time_changed_str)
                
        self.file_list = file_list
                    
        return file_list

    def get_media_file_list_small_file_size(self,file_list_small):
        """
        get media file list 

        parameters:
            file_list_small: file list with only file name and file size as array values

        returns:
            media file list with only file name and file size as array values

        """

        image_formats = ["jpg","jpeg","png"]
        video_formats = ["mp4","m4p","mpg","mpeg","avi","mov","wmv","gif"]
        formats = image_formats # + video_formats

        media_file_list_small = []

        for file in file_list_small:

            for f in formats:
                if file[0][-len(f):].lower() == f and file[1] > 10000:
                    media_file_list_small.append([file[0],file[1]])
                    
        return media_file_list_small

    def get_complete_list(self,file_list_small,file_list):
        """
        get complete file list

        description:
            assembles the complete file with array values file_name,file_path,file_size,file_created,file_modified 
            from the small list with only file name and file size

        parameters:
            file_list_small: file list with only file name and file size as array values
            file_list: file list with all five array values

        returns:
            media file list with all five array values

        """

        media_file_set = {(file[0],file[1]) for file in file_list_small}

        media_file_list = []

        for file in file_list:

            if tuple((file[0],file[2])) in media_file_set:
                media_file_list.append([file[0],file[1],file[2],file[3],file[4]])
                
        return media_file_list    

    def get_media_file_list_from_file_list(self,file_list):
        """
        get media file list 

        parameters:
            file_list: file list with array values file_name,file_path,file_size,file_created,file_modified.

        returns:
            media file list with all five array values

        """


        image_formats = ["jpg","jpeg","png"]
        video_formats = ["mp4","m4p","mpg","mpeg","avi","mov","wmv","gif"]
        formats = image_formats # + video_formats

        media_file_list = []

        for file in file_list:

            for f in formats:
                if file[0][-len(f):].lower() == f and file[2] > 10000:
                    media_file_list.append(file)
                    
        return media_file_list

    def get_media_file_list(self,path):
        file_list = self.get_file_list(path)
        media_file_list = self.get_media_file_list_from_file_list(file_list)

        return media_file_list



class Application(Frame):

    def get_file_list(self):
    
        self.file_list = self.file_mgmt.get_media_file_list(self.path_img.get())

        self.status.set("Click 'Next' to start")
                    
        return self.file_list
    

    def next(self):

        self.iterator = 0

        for i in range(len(self.face_names)):
            if "Unknown" in self.face_names[i] and self.person_name.get()=="":
                self.status.set("Please type in the name of " + self.face_names[i] + " and click 'Next'")
                break
            elif "Unknown" in self.face_names[i] and self.person_name.get()!="":
                self.face_rcg.change_face_name(self.face_names[i],self.person_name.get())
                self.face_names[i] = self.person_name.get()
                self.person_name.set("")
                self.status.set("Click 'Next'")
                break
        else:

            for i in range(len(self.file_list)):

                # first delete old image
                if self.file_path != "":
                    try:
                        os.remove(self.file_path)
                    except Exception as e:
                        print("Could not delete old file. Error: {0}".format(e))

                print("file list: {}".format(self.file_list))
                print("file: {}".format(self.file_list[self.iterator][1]))

                # display new image
                self.face_names, self.file_path = self.face_rcg.recognize(self.file_list[self.iterator][1])

                print(self.face_names)

                # show image with faces
                size = (1000, 1000)
                try:
                    self.im = Image.open(self.file_path)
                    self.im.thumbnail(size, Image.ANTIALIAS)
                    self.iterator += 1
                except IOError:
                    print("cannot create thumbnail for {0}".format(self.file_list[self.iterator]))

            
                self.img2 = ImageTk.PhotoImage(self.im) 

                self.canvas.configure(image=self.img2)
                self.canvas.image = self.img2

                self.status.set("Click 'Next'")


        # show the next image
        """size = (1000, 1000)
        try:
            self.im = Image.open(self.file_list[self.iterator][1])
            self.im.thumbnail(size, Image.ANTIALIAS)
            self.iterator += 1
        except IOError:
            print("cannot create thumbnail for {0}".format(self.file_list[self.iterator]))

    
        self.img2 = ImageTk.PhotoImage(self.im) 

        self.canvas.configure(image=self.img2)
        self.canvas.image = self.img2"""

    def load_files(self):
        self.get_file_list()

    def read_enc(self):
        self.face_rcg.read_known_faces_from_json_file("/home/manuel/Documents/MachineLearning/known_faces.json")

    def write_enc(self):
        self.face_rcg.write_known_faces_to_json("/home/manuel/Documents/MachineLearning/known_faces.json")

    def createWidgets(self):
       
        self.winfo_toplevel().title("Face Recognition")

        fontStyle = tkFont.Font(size=20)
        self.headline = Label(text="Face Recognition", font=fontStyle, anchor="w").grid(row=0,column=0)

        fontStyle = tkFont.Font(size=12)

        self.im = Image.new('RGB', (1000, 1000),(255,255,255))
        self.img = ImageTk.PhotoImage(self.im) 
        self.canvas = Label(image=self.img, width = 1000, height = 1000)
        self.canvas.grid(row=2,column=2, columnspan = 2, rowspan = 20, padx = 5, pady = 5)

        # pfad images modell
        self.path_img_label = Label(text="Path images:", width=40, font=fontStyle, anchor="w").grid(row=7,column=0,sticky=W,columnspan=2)
        self.path_img = StringVar()
        self.path_img.set("/home/manuel/Pictures/test_face_recognition")
        self.entry_path_img = Entry(width=40,textvariable=self.path_img).grid(row=8,column=0,sticky=W,columnspan=2) 

        # get file list button
        self.files_button_text = StringVar()
        self.files_button_text.set("Get file list")
        self.files_button = Button(textvariable=self.files_button_text,command=self.load_files,width=40)
        self.files_button.grid(row=10,column=0,sticky=W,columnspan=2)

        # Name person
        self.person_name_label = Label(text="Person name:", width=40, font=fontStyle, anchor="w").grid(row=12,column=0,sticky=W,columnspan=2)
        self.person_name = StringVar()
        self.entry_person_name = Entry(width=40,textvariable=self.person_name).grid(row=13,column=0,sticky=W,columnspan=2) 

        # next button
        self.next_button_text = StringVar()
        self.next_button_text.set("Next")
        self.next_button = Button(textvariable=self.next_button_text,command=self.next,width=40)
        self.next_button.grid(row=16,column=0,sticky=W,columnspan=2)

        # read known encodings
        self.read_encodings_text = StringVar()
        self.read_encodings_text.set("Read encodings")
        self.read_encodings = Button(textvariable=self.read_encodings_text,command=self.read_enc,width=40)
        self.read_encodings.grid(row=18,column=0,sticky=W,columnspan=2)

        # write known encodings
        self.write_encodings_text = StringVar()
        self.write_encodings_text.set("Write encodings")
        self.write_encodings = Button(textvariable=self.write_encodings_text,command=self.write_enc,width=40)
        self.write_encodings.grid(row=20,column=0,sticky=W,columnspan=2)

        self.status = StringVar()
        self.status.set("Waiting")
        self.status_headline = Label(textvariable=self.status, width=40, font=fontStyle, anchor="w").grid(row=27,column=0,sticky=W,columnspan=2)
        

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.iterator = 0
        self.face_names = []
        self.file_path = ""
        self.file_mgmt = FileManagement()
        self.face_rcg = FaceRecognition()
        self.grid()
        self.createWidgets()



class FaceRecognition(object):

    def recognize(self,path):

        face_locations = []
        face_encodings = []
        face_names = []


        img_full = cv2.imread(path) #, cv2.IMREAD_UNCHANGED)
        #small_frame = cv2.resize(img,(0,0),fx=0.25, fy=0.25)
        #rgb_small_frame = small_frame[:, :, ::-1]

        scale_percent = 1000/img_full.shape[1] # percent of original size
        width = int(img_full.shape[1] * scale_percent )
        height = int(img_full.shape[0] * scale_percent )
        dim = (width, height)
        # resize image
        img_bgr = cv2.resize(img_full, dim)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        #img = img_bgr[:, :, ::-1]
    

        face_locations = fr.face_locations(img)
        face_encodings = fr.face_encodings(img,face_locations)

        for face_encoding in face_encodings:

            # See if the face is a match for the known face(s)
            #matches = fr.compare_faces(self.known_face_encodings, face_encoding)
            #matches = []

            # If a match was found in known_face_encodings, just use the first one.
            """if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            else:
                name = "Unknown_" + str(round(random.random()*1000000000000000)) """

            # Or instead, use the known face with the smallest distance to the new face
            #face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            #best_match_index = np.argmin(face_distances)
            #if matches[best_match_index]:
            #    name = self.known_face_names[best_match_index]
            #else:
            #    name = "Unknown_" + str(round(random.random()*1000000000000000)) 

            name = "Unknown_" + str(round(random.random()*1000000000000000)) 

            face_names.append(name)

        self.known_face_encodings.extend(face_encodings)
        self.known_face_names.extend(face_names)

        print("face locations: {0}".format(face_locations))
        print("face names: {0}".format(face_names))

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            print(name)

            # Draw a box around the face
            cv2.rectangle(img_bgr, (left, top), (right, bottom), (255, 255, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img_bgr, (left, bottom), (right+150, bottom+35), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_bgr, name, (left + 6, bottom + 30), font, 0.8, (0, 0, 0), 2)

        path_folder = path[:-len(path.split('/')[-1])]
        file_path = path_folder + "image_faces_" + str(round(random.random()*1000000000000000)) + ".jpg"
        cv2.imwrite(file_path,img_bgr)

        return face_names, file_path

    def write_json_file(self,path,data):
        with open(path, "w") as write_file:
            json.dump(data, write_file)

    def change_face_name(self,old_face_name,new_face_name):
        print(self.known_face_names)
        self.known_face_names = [new_face_name if x==old_face_name else x for x in self.known_face_names]
        print(self.known_face_names)

    def write_known_faces_to_json(self,path):
        print(self.known_face_names)
        data_faces = dict()
        data_faces["faces"] = [{"name_person":self.known_face_names[i],"face_encoding":self.known_face_encodings[i].tolist()} for i in range(len(self.known_face_encodings))]
        self.write_json_file(path,data_faces)

    def read_known_faces_from_json_file(self,path):
    
        with open(path, "r") as read_file:
            data = json.load(read_file)

            face_names = [file["name_person"] for file in data["faces"]]
            face_encodings = [np.array(file["face_encoding"]) for file in data["faces"]]

        self.known_face_encodings = face_encodings
        self.known_face_names = face_names

        return face_names,face_encodings

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []




if __name__ == "__main__":

    root = Tk()
    app = Application(master=root)
    app.mainloop()
    root.destroy()
