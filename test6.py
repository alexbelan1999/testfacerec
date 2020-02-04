import face_recognition
from PIL import Image, ImageDraw
import numpy as np

bill_image = face_recognition.load_image_file("person/Bill_Gates01.jpg")
elon_image = face_recognition.load_image_file("person/Elon_Musk01.jpg")
steve_image = face_recognition.load_image_file("person/Steve_Jobs01.jpg")

try:
    bill_face_encoding = face_recognition.face_encodings(bill_image)[0]
    elon_image_face_encoding = face_recognition.face_encodings(elon_image)[0]
    steve_image_face_encoding = face_recognition.face_encodings(steve_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()


known_face_encodings = [
    bill_face_encoding,
    elon_image_face_encoding,
    steve_image_face_encoding
]
known_face_names = [
    "Bill",
    "Elon",
    "Steve"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("testphoto/bill_elon_steve3.jpg")


face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)


pil_image = Image.fromarray(unknown_image)

draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


del draw


pil_image.show()

pil_image.save("savephoto/image_with_boxes.jpg")