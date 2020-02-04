import face_recognition

bill_image = face_recognition.load_image_file("person/Bill_Gates01.jpg")
elon_image = face_recognition.load_image_file("person/Elon_Musk01.jpg")
steve_image = face_recognition.load_image_file("person/Steve_Jobs01.jpg")
unknown_image = face_recognition.load_image_file("testphoto/bill_elon_steve2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    bill_face_encoding = face_recognition.face_encodings(bill_image)[0]
    elon_image_face_encoding = face_recognition.face_encodings(elon_image)[0]
    steve_image_face_encoding = face_recognition.face_encodings(steve_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    bill_face_encoding,
    elon_image_face_encoding,
    steve_image_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding,0.8)
distance = face_recognition.face_distance(known_faces,unknown_face_encoding)
print(results)
print(distance)
print("Is the unknown face a picture of Bill? {}".format(results[0]))
print("Is the unknown face a picture of Elon? {}".format(results[1]))
print("Is the unknown face a picture of Steve? {}".format(results[2]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
