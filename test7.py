import face_recognition

known_bill_image = face_recognition.load_image_file("person/Bill_Gates01.jpg")
known_elon_image = face_recognition.load_image_file("person/Elon_Musk01.jpg")
known_steve_image = face_recognition.load_image_file("person/Steve_Jobs01.jpg")

bill_face_encoding = face_recognition.face_encodings(known_bill_image)[0]
elon_face_encoding = face_recognition.face_encodings(known_elon_image)[0]
steve_face_encoding = face_recognition.face_encodings(known_steve_image)[0]

known_encodings = [
    bill_face_encoding,
    elon_face_encoding,
    steve_face_encoding
]

image_to_test = face_recognition.load_image_file("testphoto/bill_elon_steve3.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

# See how far apart the test image is from the known faces
face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

for i, face_distance in enumerate(face_distances):
    print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
    print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
    print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(
        face_distance < 0.5))
    print()
