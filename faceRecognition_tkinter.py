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

from collections import Counter

import string

from sklearn.cluster import DBSCAN
from imutils import build_montages

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

        self.path_img.set("")

        self.new_file_list = True

        self.status.set("Click 'Next' to start")
                    
        return self.file_list
    

    def next(self):
        """Combination of methods encode and cluster. This method first encodes the faces and then clusters them. 
        Finally one face of every cluster is shown to name the face. 

        """

        for i in range(len(self.face_names)):

            if self.face_names[i][0] == "":

                display_size_image = 600
                display_scale_image = display_size_image / 1000

                img_full = cv2.imread(self.face_rcg.get_known_face_files()[self.face_rcg.get_known_face_ids().index(self.face_ids[i][0])]) #, cv2.IMREAD_UNCHANGED)
                scale_percent = display_scale_image*1000/img_full.shape[1] # percent of original size
                width = int(img_full.shape[1] * scale_percent )
                height = int(img_full.shape[0] * scale_percent )
                dim = (width, height)
                # resize image
                img_bgr = cv2.resize(img_full, dim)

                (top, right, bottom, left) = self.face_rcg.get_known_face_locations()[self.face_rcg.get_known_face_ids().index(self.face_ids[i][0])]

                top = int(display_scale_image*top)
                right = int(display_scale_image*right)
                bottom = int(display_scale_image*bottom)
                left = int(display_scale_image*left)

                # Draw a box around the face
                cv2.rectangle(img_bgr, (left, top), (right, bottom), (255, 255, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img_bgr, (left, bottom), (right+120, bottom+30), (255, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img_bgr, self.face_ids[i][0], (left + 6, bottom + 25), font, 0.6, (0, 0, 0), 2)

                # change color channel
                b,g,r = cv2.split(img_bgr)
                img = cv2.merge((r,g,b))

                # Convert the Image object into a TkPhoto object
                im = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im) 
                self.canvas.configure(image=imgtk)
                self.canvas.image = imgtk


                self.status.set("Please type in the name of " + self.face_ids[i][0] + " and click 'Save' and 'Next'")
                break

        else:

            if self.new_file_list:

                self.new_file_list = False

                self.face_rcg.encode([file[1] for file in self.file_list])

                self.face_clusters = self.face_rcg.cluster_faces_dbscan()
                #self.face_clusters = self.face_rcg.cluster_faces()
                self.face_ids,self.face_names,self.face_cluster_id = self.face_rcg.get_processed_clusters(self.face_clusters)

                self.status.set("Faces recognized. Click 'Next'")

            """else:

                self.face_rcg.write_images_with_names("/home/manuel/Pictures/test_face_recognition/known/")
                self.status.set("Finished. Type path and click 'Get file list' to start new.")"""


    def encode(self):
        """Encodes the images of the file list

        """

        if self.new_file_list:

            self.new_file_list = False

            self.face_rcg.encode([file[1] for file in self.file_list])

            self.status.set("Faces encoded")

    def cluster(self):
        """Clusters the images which were already encoded. Then one face of every cluster is shown to name the face. 

        """

        self.face_clusters = self.face_rcg.cluster_faces_dbscan()
        #self.face_clusters = self.face_rcg.cluster_faces()
        self.face_ids,self.face_names,self.face_cluster_id = self.face_rcg.get_processed_clusters(self.face_clusters)

        self.status.set("Faces clustered. Click 'Next'")


    def write_images(self):

        self.face_rcg.write_images_with_all_names("/home/manuel/Pictures/test_face_recognition/known/")
        self.status.set("Finished. Type path and click 'Get file list' to start new.")


    def save(self):
        for i in range(len(self.face_names)):
            """if "Unknown" in self.face_names[i][0] and self.person_name.get()=="":
                self.status.set("Please type in the name of " + self.face_names[i][0] + " and click 'Next'")
                break"""
            if self.face_names[i][0] == "":

                for k,face in enumerate(self.face_names[i]):
                    self.face_rcg.change_face_name(self.face_ids[i][k],self.person_name.get())
                    self.face_rcg.change_face_cluster_id(self.face_ids[i][k],self.face_cluster_id[i][k])
                    self.face_names[i][k] = self.person_name.get()

                self.face_rcg.change_cluster_name(self.face_cluster_id[i][0],self.person_name.get())
                
                self.person_name.set("")

                break

    def load_files(self):
        self.get_file_list()

    def read_enc(self):
        cwd = os.getcwd()
        #self.face_rcg.read_known_faces_from_json_file("/home/manuel/Documents/MachineLearning/known_faces.json")
        path = cwd + "/known_faces.json"
        self.face_rcg.read_known_faces_from_json_file(path)

    def write_enc(self):
        cwd = os.getcwd()
        #self.face_rcg.write_known_faces_to_json("/home/manuel/Documents/MachineLearning/known_faces.json")
        path = cwd + "/known_faces.json"
        self.face_rcg.write_known_faces_to_json(path)

    def createWidgets(self):
       
        self.winfo_toplevel().title("Face Recognition")

        fontStyle = tkFont.Font(size=20)
        self.headline = Label(text="Face Recognition", font=fontStyle, anchor="w").grid(row=0,column=0)

        fontStyle = tkFont.Font(size=12)

        self.im = Image.new('RGB', (600, 600),(255,255,255))
        self.img = ImageTk.PhotoImage(self.im) 
        self.canvas = Label(image=self.img, width = 600, height = 600)
        self.canvas.grid(row=2,column=2, columnspan = 2, rowspan = 20, padx = 5, pady = 5)

        # pfad images modell
        self.path_img_label = Label(text="Path images:", width=40, font=fontStyle, anchor="w").grid(row=7,column=0,sticky=W,columnspan=2)
        self.path_img = StringVar()
        self.path_img.set("/home/manuel/Pictures/test_face_recognition/unknown")
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
        self.next_button.grid(row=14,column=0,sticky=W,columnspan=2)

        # encode button
        self.encode_button_text = StringVar()
        self.encode_button_text.set("Encode")
        self.encode_button = Button(textvariable=self.encode_button_text,command=self.encode,width=40)
        self.encode_button.grid(row=15,column=0,sticky=W,columnspan=2)

        # cluster button
        self.cluster_button_text = StringVar()
        self.cluster_button_text.set("Cluster")
        self.cluster_button = Button(textvariable=self.cluster_button_text,command=self.cluster,width=40)
        self.cluster_button.grid(row=16,column=0,sticky=W,columnspan=2)

        # write images button
        self.write_images_button_text = StringVar()
        self.write_images_button_text.set("Write images")
        self.write_images_button = Button(textvariable=self.write_images_button_text,command=self.write_images,width=40)
        self.write_images_button.grid(row=17,column=0,sticky=W,columnspan=2)

        # save name button
        self.save_button_text = StringVar()
        self.save_button_text.set("Save name")
        self.save_button = Button(textvariable=self.save_button_text,command=self.save,width=40)
        self.save_button.grid(row=19,column=0,sticky=W,columnspan=2)

        # read known encodings
        self.read_encodings_text = StringVar()
        self.read_encodings_text.set("Read encodings")
        self.read_encodings = Button(textvariable=self.read_encodings_text,command=self.read_enc,width=40)
        self.read_encodings.grid(row=21,column=0,sticky=W,columnspan=2)

        # write known encodings
        self.write_encodings_text = StringVar()
        self.write_encodings_text.set("Write encodings")
        self.write_encodings = Button(textvariable=self.write_encodings_text,command=self.write_enc,width=40)
        self.write_encodings.grid(row=23,column=0,sticky=W,columnspan=2)

        self.status = StringVar()
        self.status.set("Type in path and click 'Get file list'")
        self.status_headline = Label(textvariable=self.status, width=40, font=fontStyle, anchor="w").grid(row=30,column=0,sticky=W,columnspan=2)
        

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.iterator = 0
        self.face_names = []
        self.face_ids = []
        self.face_cluster_id = []
        self.face_clusters = []
        self.file_path = ""
        self.new_file_list = False
        self.file_mgmt = FileManagement()
        self.face_rcg = FaceRecognition()
        self.grid()
        self.createWidgets()



class FaceRecognition(object):

    def id_generator(self,size=12, chars=string.ascii_letters + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def encode(self,paths):
        """Encode all images 

        Parameters:
        array paths: paths of the images to encode

        Returns:


        """

        for path in paths:

            face_locations = []
            face_encodings = []
            face_names = []
            face_ids = []


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

                name = ""
                face_id = self.id_generator()

                face_names.append(name)
                face_ids.append(face_id)

            self.known_face_encodings.extend(face_encodings)
            self.known_face_names.extend(face_names)
            self.known_face_ids.extend(face_ids)
            self.face_cluster_id.extend(["" for i in range(len(face_ids))])
            self.known_face_locations.extend(face_locations)
            self.known_face_files.extend([path for i in range(len(face_names))])

            print("face names: {0}".format(face_names))


        return 

    def build_cluster(self,ids,encodings,names,face_cluster_id):
        """Recursive function to build clusters of same faces based on compare_faces method of face_recognition

        Parameters:
        array ids: ids of faces
        array encodings: encodings of faces
        array names: names of faces 
        array face_cluster_id: id of the cluster the face belongs to 

        Returns:
        array: recognized cluster

        """

        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(encodings[1:], encodings[0],tolerance=0.58)
        face_ids_cluster = []
        face_encodings_cluster = []
        face_names_cluster = []
        face_cluster_id_cluster = []
        face_ids_unclustered = []
        face_encodings_unclustered = []
        face_names_unclustered = []
        face_cluster_id_unclustered = []

        clusters = []

        if True in matches:
            # find the indexes of all matched faces 
            matchedIdxs = [1+i for (i, b) in enumerate(matches) if b]

            face_ids_cluster.append(ids[0])
            face_ids_cluster.extend([ids[i] for i in matchedIdxs])
            face_ids_unclustered.extend([ids[i] for i in range(1,len(ids)) if i not in matchedIdxs])
            face_names_cluster.append(names[0])
            face_names_cluster.extend([names[i] for i in matchedIdxs])
            face_names_unclustered.extend([names[i] for i in range(1,len(names)) if i not in matchedIdxs])
            face_cluster_id_cluster.append(face_cluster_id[0])
            face_cluster_id_cluster.extend([face_cluster_id[i] for i in matchedIdxs])
            face_cluster_id_unclustered.extend([face_cluster_id[i] for i in range(1,len(face_cluster_id)) if i not in matchedIdxs])
            face_encodings_cluster.append(encodings[0])
            face_encodings_cluster.extend([encodings[i] for i in matchedIdxs])
            face_encodings_unclustered.extend([encodings[i] for i in range(1,len(encodings)) if i not in matchedIdxs])
        else:
            face_ids_cluster.append(ids[0])
            face_ids_unclustered.extend([ids[i] for i in range(1,len(ids))])
            face_names_cluster.append(names[0])
            face_names_unclustered.extend([names[i] for i in range(1,len(names))])
            face_cluster_id_cluster.append(face_cluster_id[0])
            face_cluster_id_unclustered.extend([face_cluster_id[i] for i in range(1,len(face_cluster_id))])
            face_encodings_cluster.append(encodings[0])
            face_encodings_unclustered.extend([encodings[i] for i in range(1,len(encodings))])

        clusters.append([face_ids_cluster,face_encodings_cluster,face_names_cluster,face_cluster_id_cluster])

        if len(face_encodings_unclustered) > 1:
            clusters.extend(self.build_cluster(face_ids_unclustered,face_encodings_unclustered,face_names_unclustered,face_cluster_id_unclustered))
        elif len(face_encodings_unclustered) == 1:
            clusters.append([face_ids_unclustered,face_encodings_unclustered,face_names_unclustered,face_cluster_id_unclustered])

        return clusters

    def cluster_faces(self):
        """Loops through all face encodings and returns clusters of unknown same faces. 
        Uses the compare_faces method of face_recognition to find faces which are similar

        Parameters:
        
        Returns:
        array: array of clusters

        """

        clusters = []

        face_encodings = self.known_face_encodings.copy()
        face_names = self.known_face_names.copy()
        face_ids = self.known_face_ids.copy()
        face_cluster_id = self.face_cluster_id.copy()

        clusters = self.build_cluster(face_ids,face_encodings,face_names,face_cluster_id)

        return clusters

    def cluster_faces_dbscan(self):
        """Returns clusters of same faces. Uses the DBSCAN algorithm to cluster faces. 

        Parameters:

        Returns:
        array: array of clusters 

        """

        clusters = []

        face_encodings = self.known_face_encodings.copy()
        face_names = self.known_face_names.copy()
        face_ids = self.known_face_ids.copy()
        face_cluster_id = self.face_cluster_id.copy()

        clt = DBSCAN(eps=0.56,metric='euclidean')
        clt.fit(face_encodings)

        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])

        for labelID in labelIDs:
            # find all indexes into the `data` array that belong to the
	        # current label ID
            idxs = np.where(clt.labels_ == labelID)[0]

            face_ids_cluster = []
            face_encodings_cluster = []
            face_names_cluster = []
            face_cluster_id_cluster = []

            for i in idxs:
                face_ids_cluster.append(face_ids[i])
                face_encodings_cluster.append(face_encodings[i])
                face_names_cluster.append(face_names[i])
                face_cluster_id_cluster.append(face_cluster_id[i])

            clusters.append([face_ids_cluster,face_encodings_cluster,face_names_cluster,face_cluster_id_cluster])


        return clusters
        

    def get_processed_clusters(self,clusters):
        """Assign cluster IDs to unknown faces. Generate new cluster ID for unknown clusters 

        Parameters:
        array clusters: clusters of known and unknown faces

        Returns: 
        array face_ids: ids of faces
        array face_names: names of faces
        array face_cluster_id: cluster IDs of faces

        """

        face_ids = [cluster[0] for cluster in clusters]
        face_names = [cluster[2] for cluster in clusters]
        face_cluster_id = [cluster[3] for cluster in clusters]

        for i in range(len(face_ids)):
            face_cluster_id_cache = sorted(face_cluster_id[i],reverse=True)
            if "" in face_cluster_id_cache and face_cluster_id_cache[0]:

                cluster_ids_sorted = [cluster_id for cluster_id in face_cluster_id[i] if cluster_id]
                cluster_names_sorted = [name for name in face_names[i] if name]
                cluster_ids_sorted = sorted(cluster_ids_sorted, key=face_cluster_id[i].count,reverse=True)
                cluster_names_sorted = sorted(cluster_names_sorted, key=face_cluster_id[i].count,reverse=True)

                # then assign the new faces the most common cluster
                face_cluster_id[i] = [cluster_ids_sorted[0] for i in range(len(face_cluster_id_cache))]
                face_names[i] = [cluster_names_sorted[0] for i in range(len(face_cluster_id_cache))]

                # change names and cluster IDs of already known faces
                for k in range(len(face_ids[i])):
                    idx = self.known_face_ids.index(face_ids[i][k])
                    self.change_face_name(face_ids[i][k],face_names[i][0])
                    self.change_face_cluster_id(face_ids[i][k],face_cluster_id[i][0])
                

            elif "" in face_cluster_id_cache:
                # cluster is not known -> generate cluster id and assign it
                new_cluster_id = self.id_generator()
                face_cluster_id[i] = [new_cluster_id for i in range(len(face_cluster_id_cache))]
                self.cluster_ids.append(new_cluster_id)
                self.cluster_names.append("")


        return face_ids,face_names,face_cluster_id


    def write_images_with_names(self,write_path):
        """Write all known images. Every image contains just one name. 

        Parameters:
        string write_path: path to write the images to

        Returns:


        """

        paths = self.known_face_files
        face_locations = self.known_face_locations
        face_encodings = self.known_face_encodings
        face_names = self.known_face_names
        face_ids = self.known_face_ids

        for i,path in enumerate(paths):

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

            (top,right,bottom,left) = face_locations[i]
            name = face_names[i]

            # Draw a box around the face
            cv2.rectangle(img_bgr, (left, top), (right, bottom), (255, 255, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img_bgr, (left, bottom), (right+150, bottom+35), (255, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_bgr, name, (left + 6, bottom + 30), font, 0.8, (0, 0, 0), 2)
            
            #path_folder = path[:-len(path.split('/')[-1])]
            file_name = path[-len(path.split('/')[-1]):-4]
            file_path = write_path + file_name + "_" + name + "_" + str(round(random.random()*1000000000000000)) + ".jpg"
            cv2.imwrite(file_path,img_bgr)

        return 

    def write_images_with_all_names(self,write_path):
        """Write all known images. Every image contains all names in it. 

        Parameters:
        string write_path: path to write the images to

        Returns:


        """

        paths = self.known_face_files
        face_locations = self.known_face_locations
        face_encodings = self.known_face_encodings
        face_names = self.known_face_names
        face_ids = self.known_face_ids

        uniquePaths = np.unique(paths)
        np_paths = np.array(paths)

        for path in uniquePaths:

            idxs = np.where(np_paths == path)[0]

            img_full = cv2.imread(path) 

            scale_percent = 1000/img_full.shape[1] # percent of original size
            width = int(img_full.shape[1] * scale_percent )
            height = int(img_full.shape[0] * scale_percent )
            dim = (width, height)

            # resize image
            img_bgr = cv2.resize(img_full, dim)
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            for i in idxs:
                (top,right,bottom,left) = face_locations[i]
                name = face_names[i]

                # Draw a box around the face
                cv2.rectangle(img_bgr, (left, top), (right, bottom), (255, 255, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img_bgr, (left, bottom), (right+150, bottom+35), (255, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img_bgr, name, (left + 6, bottom + 30), font, 0.8, (0, 0, 0), 2)
            
            #path_folder = path[:-len(path.split('/')[-1])]
            file_name = path[-len(path.split('/')[-1]):-4]
            file_path = write_path + file_name + "_known_faces" + ".jpg"
            cv2.imwrite(file_path,img_bgr)

        return 


    def write_json_file(self,path,data):
        with open(path, "w") as write_file:
            json.dump(data, write_file)

    def change_face_name(self,face_id,new_face_name):
        print(self.known_face_names)
        #self.known_face_names = [new_face_name if value==face_id else self.known_face_names[count] for count,value in enumerate(self.known_face_ids)]
        idx = self.known_face_ids.index(face_id)
        self.known_face_names[idx] = new_face_name
        print(self.known_face_names)

    def change_face_cluster_id(self,face_id,new_cluster_id):
        idx = self.known_face_ids.index(face_id)
        self.face_cluster_id[idx] = new_cluster_id

    def change_cluster_name(self,cluster_id,new_cluster_name):
        idx = self.cluster_ids.index(cluster_id)
        self.cluster_names[idx] = new_cluster_name

    def write_known_faces_to_json(self,path):
        print(self.known_face_names)
        data_faces = dict()
        data_faces["faces"] = [{"face_id":self.known_face_ids[i],"name_person":self.known_face_names[i],"face_cluster_id":self.face_cluster_id[i],"face_location":self.known_face_locations[i],"face_file":self.known_face_files[i],"face_encoding":self.known_face_encodings[i].tolist()} for i in range(len(self.known_face_encodings))]
        self.write_json_file(path,data_faces)

    def read_known_faces_from_json_file(self,path):
    
        with open(path, "r") as read_file:
            data = json.load(read_file)

            face_ids = [file["face_id"] for file in data["faces"]]
            face_names = [file["name_person"] for file in data["faces"]]
            face_cluster_id = [file["face_cluster_id"] for file in data["faces"]]
            face_encodings = [np.array(file["face_encoding"]) for file in data["faces"]]
            face_locations = [file["face_location"] for file in data["faces"]]
            face_files = [file["face_file"] for file in data["faces"]]

        self.known_face_ids = face_ids
        self.known_face_encodings = face_encodings
        self.known_face_names = face_names
        self.face_cluster_id = face_cluster_id
        self.known_face_locations = face_locations
        self.known_face_files = face_files

        return face_ids,face_names,face_cluster_id,face_encodings,face_locations,face_files

    def get_known_face_names(self):
        return self.known_face_names

    def get_known_face_ids(self):
        return self.known_face_ids

    def get_known_face_files(self):
        return self.known_face_files

    def get_known_face_locations(self):
        return self.known_face_locations

    def __init__(self):
        self.known_face_encodings = [] # Encodings of all recognized faces
        self.known_face_names = [] # Given face names for all recognized faces
        self.known_face_ids = [] # IDs of all recognized faces
        self.face_cluster_id = [] # ID of cluster each face belongs to (1 cluster = 1 person) 
        self.known_face_locations = []
        self.known_face_files = []

        self.cluster_ids = [] # available cluster ids
        self.cluster_names = [] # names of clusters




if __name__ == "__main__":

    root = Tk()
    app = Application(master=root)
    app.mainloop()
    root.destroy()

