import numpy as np

#Essa rede neural vai determinar se eu estou feliz
#Hyper feliz = 2, feliz = 1, tisti = 0

#Did I programmed in java tday, yes = 1, no = 0
#Did i programmer in python tday, yes = 1, no = 0
#Did I talked to females today yes = 1 no = 0

#Aqui eu coloco varios casos em especificos, tipo 1,1 programei e falei com femeas
inputs = np.array([
    [1, 1, 1], #Programei em java, py e falei com molieres
    [1, 1, 0], #programei em java e py
    [1, 0, 1], #Proggramei em java e falei com femeas
    [0, 1, 1], #programei em python e falei com molieres
    [0, 0, 1], #Falei com femeas
    [0, 1, 0], #Programei em python
    [1, 0, 0] #programei em java
])

outputs = np.array([[2], [1], [1], [2], [1], [2], [0]])

#This tells how important something is, like wuts more importante python or females
#All tho we make it random cuz the ai need to learn it and shit
weights = np.random.rand(3, 1)

#basically it makes the robot understand that 0 is sad and 2 is hyper sippie happy using a thing called sigmoid
def sigmoid(x):
    return 1/ (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

#Now we make the ai run this shit a lot of fucking times so it fucking learns!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#test 10k times
for _ in range(10000):
    #give the data
    input_layer = inputs #give it the examples
    guesses = sigmoid(np.dot(input_layer, weights)) #Robot make guesses

    #find out how wrong it was
    error = outputs - guesses

    #fix a bit wuts wrong
    adjustments = error * sigmoid_derivative(guesses)
    weights += np.dot(input_layer.T, adjustments)
    #Now repeat 10 fucking thousand times (emoji with black glasses cuz its coollll)

#so thats the data, for example i did java but talked to woman
data = np.array([1, 0, 1])
#calculates how happy i am
result = sigmoid(np.dot(data, weights))
print("How happy i am", result)

#Im so sorry for how brainrot that was and for mixing  english and portuguese, but im glad I made a NEURAL NETWORK *boommmm*