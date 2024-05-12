# Importing pandas for preprocessing
import pandas as pd

# Importing joblib to dump and load embeddings df
import joblib

# Importing cv2 to read images
import cv2

# Importing cosine_similarity to find similarity between images
from sklearn.metrics.pairwise import cosine_similarity

# Importing the below libraries for our model building
from pandas.core.common import flatten

# import torch
import torch

# import cv models
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

# import image
from PIL import Image

import warnings

import swifter

warnings.filterwarnings("ignore")

# Using error_bad_lines and warn_bad_lines parameters to avoid reading bad lines in this dataset
df = pd.read_csv('./archive/styles.csv', on_bad_lines='skip')

# top 10 rows
df.head(10)

# Creating a new column called as image which will store name of the image corresponding to that item id
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)

# reseting the index
df = df.reset_index(drop=True)


# image path
def image_location(img):
    return './archive/images/' + img

# function to load image
def import_img(image):
    image = cv2.imread(image_location(image))
    return image

# Defining the input shape
width = 224
height = 224

# Loading the pretrained model
resnetmodel = models.resnet18(pretrained=True)

# Use the model object to select the desired layer
layer = resnetmodel._modules.get('avgpool')

# scaling the data
s_data = transforms.Resize((224, 224))

# normalizing
standardize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

# converting to tensor
convert_tensor = transforms.ToTensor()

# missing image object
missing_img = []

# function to get embeddings

def vector_extraction(resnetmodel, image_id):

    # Using concept of exception handling to ignore missing images
    try:
        # 1. Load the image with Pillow library
        img = Image.open(image_location(image_id)).convert('RGB')

        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))

        # 3. Create a vector of zeros that will hold our feature vector
        # The 'avgpool' layer has an output size of 512
        embeddings = torch.zeros(512)

        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            embeddings.copy_(o.data.reshape(o.data.size(1)))

        # 5. Attach that function to our selected layer
        hlayer = layer.register_forward_hook(copy_data)

        # 6. Run the model on our transformed image
        resnetmodel(t_img)

        # 7. Detach our copy function from the layer
        hlayer.remove()

        # 8. Return the feature vector
        return embeddings

    # If file not found
    except FileNotFoundError:
        # Store the index of such entries in missing_img list and drop them later
        missed_img = df[df['image'] == image_id].index
        print(missed_img)

# Testing if our vector_extraction function works well on sample image
# sample_embedding_0 = vector_extraction(resnetmodel, df.iloc[0].image)
#
#
# # Applying embeddings on subset of this huge dataset
# df_embeddings = df[:5000]  # We can apply on entire df, like: df_embeddings = df
#
# # looping through images to get embeddings
# map_embeddings = df_embeddings['image'].swifter.apply(lambda img: vector_extraction(resnetmodel, img))
#
# # convert to series
# df_embs = map_embeddings.apply(pd.Series)
# df_embs.head()
#
# # export the embeddings
# df_embs.to_csv('df_embs.csv')
#
# # importing the embeddings
# df_embs = pd.read_csv('df_embs.csv')
# df_embs.drop(['Unnamed: 0'], axis=1, inplace=True)
# df_embs.dropna(inplace=True)
#
# # exporting as pkl
# joblib.dump(df_embs, 'df_embs.pkl', 9)

# importing the pkl
df_embs = joblib.load('df_embs.pkl')

def recommend_image_output(image_id):

    # Loading image and reshaping it
    img = Image.open('./archive/images/' + image_id).convert('RGB')
    print(img)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(standardize(convert_tensor(s_data(img))).unsqueeze(0))
    
    # 3. Create a vector of zeros that will hold our feature vector
    # The 'avgpool' layer has an output size of 512
    embeddings = torch.zeros(512)
    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        embeddings.copy_(o.data.reshape(o.data.size(1)))

    # 5. Attach that function to our selected layer
    hlayer = layer.register_forward_hook(copy_data)
    
    # 6. Run the model on our transformed image
    resnetmodel(t_img)
    
    # 7. Detach our copy function from the layer
    hlayer.remove()
    emb = embeddings
    
    
    # Calculating Cosine Similarity
    cs = cosine_similarity(emb.unsqueeze(0),df_embs)
    cs_list = list(flatten(cs))
    cs_df = pd.DataFrame(cs_list,columns=['Score'])
    cs_df = cs_df.sort_values(by=['Score'],ascending=False)
    
    # Printing Cosine Similarity
    print(cs_df['Score'][:4])
    
    # Extracting the index of top 10 similar items/images
    top4 = cs_df[:4].index
    top4 = list(flatten(top4))
    images_list = []

    for i in top4:
        image_id = df[df.index==i]['image']
        images_list.append(image_id)

    images_list = list(flatten(images_list))

    return images_list