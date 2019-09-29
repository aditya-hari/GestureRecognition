from keras.models import load_model
from keras.preprocessing import image

def predictor(): 
    import numpy as np

    classifier = load_model('Trained_model.h5')

    test_image = image.load_img('cap.jpg', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #trsaining_set.class_indices
    if result[0][0] == 1:
        return('A')
    elif result[0][1] == 1:
        return('C')
    elif result[0][2] == 1:
        return('F')
    elif result[0][3] == 1:
        return('I')
    elif result[0][4] == 1:
        return('L')
    elif result[0][5] == 1:
        return('U')
    elif result[0][6] == 1:
        return('W')
