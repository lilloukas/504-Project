import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from timeit import default_timer as timer
from tqdm import tqdm
import random
from mpl_toolkits.axes_grid1 import ImageGrid
TOP_K = 5
def input_image(path):
    img = Image.open(path)
    img = np.array(img.resize((32,32))).reshape(1,-1)
    return img

# Hashing the embeding with random vector projection
def get_hash(vector_represenatation,hyperplanes):
    # Take the dot product of the input vector with each of the n hyperplanes
    projection = np.dot(np.array(vector_represenatation),hyperplanes.T)
    # Classify as a 1 or zero if the value is positive or negative
    projection = projection>0
    
    hashed_values = []
    for p in projection:
        j = int_converter(p)
        hashed_values.append(j)
    return hashed_values




def int_converter(boolean_array):
    bool_as_int = 0
    for bit in boolean_array:
        bool_as_int = (bool_as_int<<1)|bit
    return bool_as_int



def create_hyperplanes(number_of_hashes,vector_dimension):
    hyperplanes = np.random.randn(number_of_hashes,vector_dimension)
    return hyperplanes


'''
Class to hash values into boolean array, which is viewed as a bit array and 
converted to its corresponding integer value by int_converter
'''

class Hash_Table:
    
    def __init__(self, number_of_hashes, vector_dimension):
        self. number_of_hashes = number_of_hashes
        self.hyperplanes = create_hyperplanes(number_of_hashes, vector_dimension)
        self.hash_table = {} #initializing empty dictionary 
        
    def insert_hashes(self, file_number,embeddings):
        
        
        #compute the hashed representation of the embeddings
        
        hashed_embeddings = get_hash(embeddings,self.hyperplanes)
        
        for idx,hashes in enumerate(hashed_embeddings):

            if hashes in self.hash_table:
                self.hash_table[hashes].extend([file_number[idx]])
            else:
                self.hash_table[hashes] = [file_number[idx]]

        

    def find_matches(self, input_embedding):

        hashed_embeddings = get_hash(input_embedding,self.hyperplanes)

        # List to store the nearest neighbors from the location the 
        #input_embedding is hashed to
        nearest_neighbors = []
        
        for hashes in hashed_embeddings:
            if hashes in self.hash_table:
                nearest_neighbors.extend(self.hash_table[hashes])

        return nearest_neighbors

'''     
Class to create n internal tables (try to reduce the impact from a single
vector splitting points which are extremly close by)
'''
class Multi_Hash_Table:
    
    def __init__(self, number_of_hashes,vector_dimension,number_of_tables):
        self.number_of_tables = number_of_tables
        self.hash_tables  = []
        
        # Create number_of_tables tables
        for i in range(self.number_of_tables):
            self.hash_tables.append(Hash_Table(number_of_hashes,vector_dimension))
            
        
    def insert_hashes(self, file_number,embeddings):
        for hash_table in self.hash_tables:
            hash_table.insert_hashes(file_number,embeddings)
            
    def find_matches(self,input_embeddings):
        nearest_neighbors = []

        for hash_table in self.hash_tables:
            nearest_neighbors.extend(hash_table.find_matches(input_embeddings))

        
        return nearest_neighbors
            


'''
Class to store a set of images in their hashed formats for comparison against 
other images 
'''
class Projection_LSH:
    
    def __init__(self,number_of_hashes = 5,number_of_tables = 5,vector_dimension = 3072):

        self.number_of_hashes = number_of_hashes
        self.number_of_tables = number_of_tables
        self.vector_dimension = vector_dimension
        self.multi_table = Multi_Hash_Table(self.number_of_hashes,
                                            self.vector_dimension,
                                            self.number_of_tables)

    def create_hash(self,image_array):

        self.multi_table.insert_hashes(file_number = list(range(0,len(image_array))),embeddings = image_array)

    def find_nearest_neighbors(self,image):
        
            
        nearest_neighbor = self.multi_table.find_matches(image)
        print(len(nearest_neighbor))
        most_seen = {}
        
        for i in nearest_neighbor:
            most_seen[i] = most_seen.get(i,0)+1
        
        sorted_answer = dict(sorted(most_seen.items(), key=lambda item: item[1],reverse = True))
        # neighbor_dict = {key:nearest_neighbors.count(key) for key in nearest_neighbors}
        
        return sorted_answer

def get_random_train(image_array):
    idx = random.randint(0,len(image_array))

    random_image = image_array[idx][0].numpy().reshape(1,-1)
    
    return idx,random_image

    
def nearest_neighbors_visual(idx,neighbors,test,train):
    
    count = 0
    images = []
    labels = []
    images.append(test[idx][0].permute(1,2,0))
    labels.append('Original - '+ str(test[idx][1]))
    for key,value in neighbors.items():
        if count>=TOP_K:
            break
        else:
            images.append(train[key][0].permute(1,2,0))
            labels.append(train[key][1])
    
        count+=1
    
        
    fig = plt.figure(figsize=(25., 25.))
    grid = ImageGrid(fig, 111,  
                     nrows_ncols=(1,6), 
                     axes_pad=0.3, 
                     )
    for ax, im,lab in zip(grid, images,labels):
    
        ax.imshow(im)
        ax.set_title(lab)
    plt.show()
    

transform = transforms.Compose([transforms.Resize([32,32]),transforms.ToTensor()])
cifar1 = CIFAR10(download=True,root="./data",transform = transform)
cifar2 = CIFAR10(root="./data",train=False,transform = transform)
NUM_IMGS = 50000
DIMENSION1 = 3
DIMENSION2 = 32
DIMENSION3 = 32
# Array of images from CIFAR100
image_array= np.zeros((NUM_IMGS,DIMENSION1,DIMENSION2,DIMENSION3))
for idx,thing in enumerate(cifar1):
    image_array[idx] = cifar1[idx][0]
image_array= image_array.reshape(image_array.shape[0],-1)
idx,test = get_random_train(cifar2)

start = timer()
LSH = Projection_LSH(number_of_hashes = 10,
                       number_of_tables = 25,
                       vector_dimension=DIMENSION1*DIMENSION2*DIMENSION3)
end = timer()
print('Time to initialize table:', end-start,'seconds')

start = timer()
LSH.create_hash(image_array)
end = timer()
print('Time to hash images:', end-start,'seconds')

start = timer()
nearest_neighbors = LSH.find_nearest_neighbors(test)
end = timer()
print('Time to query:',end-start,'seconds')

nearest_neighbors_visual(idx, nearest_neighbors,cifar2,cifar1)



def l1_distance(val1,val2):
    return np.linalg.norm(val1-val2)

def manual_nn(test,dataset):
    dist = []
    index = []
    
    for idx,vec in enumerate(dataset):
        dist.append(l1_distance(test,vec))
        index.append(idx)
    
    nearest_neighbors = sorted(list(zip(index,dist)),key = lambda x:x[1],reverse=False )
    
    return nearest_neighbors

start = timer()
manual = manual_nn(test,image_array)
end = timer()
print('Time to manually find nearest-neighbor',end-start)



def test_efficiency(num_tables,num_hashes,num_iter,image_array):
    LSH = Projection_LSH(number_of_hashes = num_hashes,
                         number_of_tables = num_tables,
                         vector_dimension=DIMENSION1*DIMENSION2*DIMENSION3)
    print('Creating LSH object with',num_hashes,'hashes and',num_tables,'tables...')
    LSH.create_hash(image_array)
    times_manual = []
    times_lsh = []
    accuracies = {}
    top_1 = 0
    top_3 = 0
    top_5 = 0
        
    for i in tqdm(range(num_iter)):
       
        idx,test = get_random_train(cifar2)
        start = timer()
        manual = manual_nn(test,image_array)
        end = timer()
        times_manual.append(end-start)
        
        actual_top5 = []
        for i in range(5):
            actual_top5.append(manual[i][0])
            
        start = timer()
        nearest_neighbors = LSH.find_nearest_neighbors(test)
        end = timer()
        times_lsh.append(end-start)
        
        try:
            lsh_best = list(nearest_neighbors.keys())[0]
            print('lsh best',lsh_best)
            print('actual best',actual_top5)
            if lsh_best == actual_top5[0]:
                top_1+=1
                top_3+=1
                top_5+=1
            elif (lsh_best == actual_top5[1]) or (lsh_best == actual_top5[2]):
                top_3+=1
                top_5+=1
            elif (lsh_best == actual_top5[1]) or (lsh_best == actual_top5[2]):
                top_5+=1
        except IndexError:
            print('No neighbors found in this table')
            
        

   
    accuracies['Top 1'] = top_1/num_iter
    accuracies['Top 3'] = top_3/num_iter
    accuracies['Top 5'] = top_5/num_iter

    return accuracies,times_manual,times_lsh

acc,man,ltime = test_efficiency(12,25,200,image_array)



LSH = Projection_LSH(number_of_hashes=10,number_of_tables=10)
LSH.create_hash(image_array)
idx,test = get_random_train(cifar1)
j = LSH.find_nearest_neighbors(test)

man = manual_nn(test,image_array)
