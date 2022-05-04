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

## Helper functions ##

# Hashing the embeding with random vector projection
def get_hash(vector_represenatation,hyperplanes):
    '''
    Hashes the input vectors into boolean array based on the input hyperplanes
    
    '''
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
    '''
    Converts the input boolean array into its integer representation for hashing buckets
    
    '''
    bool_as_int = 0
    for bit in boolean_array:
        bool_as_int = (bool_as_int<<1)|bit
    return bool_as_int

def create_hyperplanes(number_of_hashes,vector_dimension):
    hyperplanes = np.random.randn(number_of_hashes,vector_dimension)
    return hyperplanes

# Approximate nearest neighbors 
def lsh_l2(test,indexes,dataset):
    '''
    For input indices (indexes), calculates the distance from input image (test)
    and returns the sorted nearest neighbors 
    '''
    approximate_nn = {}
    for value in indexes:
       approximate_nn[value] = np.linalg.norm(test-dataset[value])
    sorted_answer = dict(sorted(approximate_nn.items(), key=lambda item: item[1],reverse = False))
    return sorted_answer

# Eucliddean distance between vectors
def l2_distance(val1,val2):
    return np.linalg.norm(val1-val2)

# Manually calculating nearest neighbors
def manual_nn(test,dataset):
    '''
    Manually calculate the nearest neighbors of image test in dataset
    '''
    dist = []
    index = []
    
    for idx,vec in enumerate(dataset):
        dist.append(l2_distance(test,vec))
        index.append(idx)
    
    nearest_neighbors = sorted(list(zip(index,dist)),key = lambda x:x[1],reverse=False )
    
    return nearest_neighbors

# Selecting a random image from the training set
def get_random_train(image_array):
    
    '''
    Selects an image randomly from the training set to hash and compute nn
    '''
    idx = random.randint(0,len(image_array))

    random_image = image_array[idx][0].numpy().reshape(1,-1)
    
    return idx,random_image

# Plotting the nearest neighbors 
def nearest_neighbors_visual(idx,neighbors,test,train):
    '''
    Plots the 5 nearest neighbors for image at idx from the training set
    
    '''
    count = 0
    images = []
    labels = []
    images.append(test[idx][0].permute(1,2,0))
    labels.append('Original - '+ str(test[idx][1]))
    if type(neighbors) == dict:
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
    else:
        for i in range(len(neighbors)):
            if count>=TOP_K:
                break
            else:
                images.append(train[neighbors[i][0]][0].permute(1,2,0))
                labels.append(train[neighbors[i][0]][1])
        
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
    
def top_10_recall(pred,true):
    return len(pred.intersection(true))/len(true)
    
'''
Class to store single hash table with specified hashes and hyperplanes
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
vector splitting points which are extremly close)
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
Class to store a set of images in their hashed formats in n tables for 
approximate nearest neighbor searches
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

        return set(nearest_neighbor)

   
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
check_idx = LSH.find_nearest_neighbors(test)
nearest_neighbors = lsh_l2(test, check_idx, image_array)
end = timer()
print('Time to query:',end-start,'seconds')

nearest_neighbors_visual(idx, nearest_neighbors,cifar2,cifar1)



start = timer()
manual = manual_nn(test,image_array)
end = timer()
print('Time to manually find nearest-neighbor',end-start)
nearest_neighbors_visual(idx, manual,cifar2,cifar1)


def test_efficiency(num_tables,num_hashes,num_iter,image_array):
    LSH = Projection_LSH(number_of_hashes = num_hashes,
                         number_of_tables = num_tables,
                         vector_dimension=DIMENSION1*DIMENSION2*DIMENSION3)
    print('\nCreating LSH object with',num_hashes,'hashes and',num_tables,'tables...')
    print('Testing Progress Below:')
    start = timer()
    LSH.create_hash(image_array)
    end = timer()
    hash_time = end-start
    times_manual = []
    times_lsh = []
    accuracies = {}
    top_1 = 0
    top_3 = 0
    top_5 = 0
    recall = []
    for i in tqdm(range(num_iter)):
       
        # Calculating nearest neighbors manually
        idx,test = get_random_train(cifar2)
        start = timer()
        manual = manual_nn(test,image_array)
        end = timer()
        times_manual.append(end-start)
        
        actual_top10 = []
        for i in range(10):
            actual_top10.append(manual[i][0])
        
        # Calculating approximate nearest neighbors
        start = timer()
        check_idx = LSH.find_nearest_neighbors(test)
        nearest_neighbors = lsh_l2(test, check_idx, image_array)
        end = timer()
        times_lsh.append(end-start)
        
        top_10 = []
        for key,_ in nearest_neighbors.items():
            top_10.append(key)
        
        # Prevent error if no neighbors are found
        try:
            lsh_best = list(nearest_neighbors.keys())[0]

            if lsh_best == actual_top10[0]:
                top_1+=1
                top_3+=1
                top_5+=1
            elif (lsh_best == actual_top10[1]) or (lsh_best == actual_top10[2]):
                top_3+=1
                top_5+=1
            elif (lsh_best == actual_top10[1]) or (lsh_best == actual_top10[2]):
                top_5+=1
        except IndexError:
            print('No neighbors found in this table')
            
        recall.append(top_10_recall(set(top_10),set(actual_top10)))
        # nearest_neighbors_visual(idx, nearest_neighbors,cifar2,cifar1) # If you want to see the predicted nearest neighbors
   
    accuracies['Top 1'] = top_1/num_iter
    accuracies['Top 3'] = top_3/num_iter
    accuracies['Top 5'] = top_5/num_iter

    return accuracies,times_manual,times_lsh,recall,hash_time

tests = []
hash_num=  list(range(1,15))
table_num = list(range(1,15))
number_of_tests = 20
for h in hash_num:
    for t in table_num:
        acc,manual_time,lsh_hash_time ,recall,hash_time= test_efficiency(t,h,number_of_tests,image_array)
        results = list([acc,np.mean(manual_time),np.mean(lsh_hash_time),np.mean(recall),hash_time])
        tests.append(results)


LSH = Projection_LSH(number_of_hashes=10,number_of_tables=10)
LSH.create_hash(image_array)
idx,test = get_random_train(cifar1)
j = LSH.find_nearest_neighbors(test)


print('*************')
h = 1;
t = 1;
count = 0
for test in tests:
    
    print('Hash Table with %d tables and %d hashes:\n'%(t,h))
    print('Time to hash all images:',test[4])
    print('Top 1 Accuracy:',test[0]['Top 1'])
    print('Top 3 Accuracy:',test[0]['Top 3'])
    print('Top 5 Accuracy:',test[0]['Top 5'])
    print('Average time to manually calculate NN:',test[1])
    print('Average time to calculate NN with LSH:',test[2])
    print('Top ten recall:',test[3])
   
    
    if t == 14:
        count+=1
        if count<=14:
            h += 1
            t = 1
        else:
            break
    else:
        t+=1
    
    if h > 14:
        break

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    