from cv2 import imread,imshow,waitKey
import tensorflow as tf
import numpy as np

def mod_load(TF_CNNModel,signet):
    model_weight_path = 'prediction/models/signet.pkl'
    model = TF_CNNModel(signet, model_weight_path)
    return model

def ses_init():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess

# Use the CNN to extract features
def euc_distance(sess,real,test,model,preprocess_signature,canvas_size = (952, 1360)):
    print(real)
    real_i=imread(real, 0)
    real_i=preprocess_signature(real_i, canvas_size)
    test_i=imread(test, 0)
    test_i=preprocess_signature(test_i, canvas_size)    
    freal = model.get_feature_vector(sess, real_i)
    ftest = model.get_feature_vector(sess, test_i)
    return np.linalg.norm(freal-ftest)

