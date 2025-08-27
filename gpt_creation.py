stoi=itos={}
text="hello my name is abcdefghijklmnopqrstuvwxyz"
chars=sorted(list(set(text)))
for i,ch in enumerate(chars): #enumerate is the index and the character taken sideways
    stoi[ch]=i
    itos[i]=ch
print(stoi)

def encode(s):
    result=[]
    for c in s:
        result.append(stoi[c]) #the unique integer is appended to the result list
    return result

def decode(l):
    result=''
    for i in l:
        result+=itos[i]
    return result #the unique characters are added into the string result

print(encode("hi nnefje"))
print(decode(encode("hi heer")))

#importing torch and subsitaries

import torch
data=torch.tensor(encode(text)) # data converted to tensors after encoding

#splitting up the training sets and the validation/test sets

n=int(0.9*len(data))
train_data=data[:n] #till 90%
val_data=data[n:]

#initializing block size

block_size=8  #8 tokens in each block
train_data[:block_size+1] #checking for 9 examples/tokens

#for example take 1 2 3 4 5 
x=train_data[:block_size] # 1 2 3 4 
y=train_data[1:block_size+1] # 2 3 4 5

for i in range(block_size):
    context=x[:i+1] # 1, 1 2 , 1 2 3
    target=y[i] # 2 3 4 5
    print("when input is",context,"the target is ",target)

#initialize torch seed
#torch seed initialized a random number which can be used to repeating examples
# why randomness? so that the model sees diverse patterns to train the model
torch.manual_seed(1337)
batch_size=4
block_size=8
# for each batch of 4 there will be 8 tokens each for a block
def get_batch(split):
    if split=="train":
        data=train_data  #an input of whether to take train or validation will be taken and according to user input the data will be set to train or validation
    else:
        data=val_data
    indice=[]
    for a in range(batch_size): #0 1 2 3 
        max_start_index=len(data)-block_size #so we dont go out of bounds of the index
        # also max_start_index is the last minimum index u can take
        random_index_tensor=torch.randint(low=0,high=max_start_index,size=(1,))
        #picks a starting index between 0 and max_start_index
        #generalizes better
        random_index=random_index_tensort.item()
        #because it returns a tensor and we just want the number
        indice.append(random_index) #all random indexes are added here

    input_sequence=[]
    target_sequence=[]
    '''
    for the following code take this example
    data=[10,20,30,40,50,60,70,80,90,100]
    block size=4
    i is a random starting index in indice for eg 2
    
    '''
    for i in indice:
        input_seq=target_seq=[]
        for j in range(i,i+block_size): #(2,6)
            input_seq.append(data[j].item()) #[30,40,50,60]
        
        for j in range(i+1,i+block_size+1):#(3,7)
            target_seq.append(data[j].item()) #[40,50,60,70]

    input_sequence.append(input_seq) # the final list is added to the list in form of a nested list
    target_sequence.append(target_seq)

    x=torch.tensor(input_sequence , dtype=torch.long) # converting all the indices to vectors
    y=torch.tensor(target_sequence, dtype=torch.long)

    return x,y

#now generating training data
xt,yt=get_batch("train")
print(xt) #inputs
print(yt) #outputs/target

for i in range(batch_size):
    for j in range(block_size):
        context_tokens=[]
        for a in range(j+1):
            token=xt[i,a].item()
            context_tokens.append(token)
        target_token=yt[i,j].item()

#basically generating the next predicted token


#PART 2- BIGRAM LANGUAGE MODEL

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module): #nn.Module is for all neural network models

    def _init_(self,vocab_size): #init is a method that runs automatically when a class is run
        super().__init__() #nn.Module has its own init method so all init methods are called, super important
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)# lookup table where token->vector(size=vocab_size)
        '''
        num_embeddings=vocab_size (like how many tokens in the vocab)
        embedding_dim=vocab_size  (size of vector per token)

        so thats why both parameters are the same. this line creates a embedding layer
        where vocab_size is how many tokens are there and what is the size

        self is a model like the entire model
        self.token_embedding_table creates the lookup table which stores inside self
        we cant do self(idx) as self is AN ENTIRE MODEL.
        self.token_embedding_table is like a part of the self model
        '''

    #idx is made of (B,T) ie batch size and tokens in each sequence

        '''
        F.cross_entropy():
        input of shape(N,C):
            N is number of samples
            C is number of classes   aka logits
        input of (N,):
            class indices for each N sample  aka target
        logits corresponds to predicted logits for ONE token
        target corresponds to correct class index for that token

        CROSS ENTROPY CHECKS HOW WRONG THE PREDICTION IS COMPARED TO THE TRUE ANSWER
        AND APPLIES SOFTMAX

        CONVERTS LOGITS TO PROBABILITES
        
        '''
        
    def forward(self,idx,targets=None):# forward pass function
        logits=self.token_embedding_table(idx)# embedding layer
        if targets is None:
            loss=None #how can loss be calculated if targets exist
        else: #flattening tensors here
            b,t,c=logits.shape
            logits=logits.view(b*t,c)
            targets=targets.view(b*t)
            loss=F.cross_entropy(logits,targets)
        return logits,loss #returning the raw tokens and loss predicted after converting and all

    def generate(self,idx,max_new_tokens): #max_new_tokens is how many tokens do u want to generate
        for a in range(max_new_tokens):
            logits,loss=self(idx) #the model runs with idx(with b and t) and gets logits and loss
            logits=logits[:,-1,:] #gets the logits of the last token of EACH sequence
            #the commas exist because we are interacting with 3d space tensors
            probs=F.softmax(logits,dim=-1) #convert to probabilites
            idx_next=torch.multinomial(probs,num_samples=1) #uses multinomial sampling to pick the next sample
            idx=torch.cat((idx,idx_next),dim=1) #updating idx by adding newly generated token
        return idx
    model=BigramLanguageModel(vocab_size)
    logits,loss=m(xt,yt)
    print(logits.shape)
    print(loss)

#creating a optimizer
optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)

'''
why we are using AdamW here is because AdamW prevents overfitting better and helps in L2 regularization

1e-3=0.001. this is the learning rate

if the learning rate is too high it might lead to overfitting
if the learning rate is low it might lead to underlearning or underfitting 
'''

batch_size=32 #32 examples, training in batches
for steps in range(100):#more steps more good results
    xb,yb=get_batch("train") #both are set to training data

    logits,loss=m(xb,yb) #the bigram model computes the logits and loss
    optimizer.zero_grad(set_to_none=True) # avoids leftover gradients
    loss.backward() #backpropogration, computes gradiet of loss wrt of every parameter
    optimizer.step() #updates the weights
    
print(loss.item()) #prints the LAST loss item, technically how good or bad the model is doing



    
    
