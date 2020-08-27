from tensorflow.keras.models import model_from_json

def init():
    json_file = open('model3.json', 'r')
    loaded_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_json)
    model.load_weights("model3.h5")
    print("Loaded the model!")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model