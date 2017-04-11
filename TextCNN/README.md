This script is a practice of applying CNN for **sentence**(i.e. very short docs) polarity classification. The goal for myself is to reproduce [Yoon Kim (2014)](https://arxiv.org/pdf/1408.5882.pdf)'s results on Rotten Tomatoes' movie reviews to get my hands dirty on CNN/TensorFlow. This goal, though, was unexpectedly tough to achieve. The [author's source code](https://github.com/yoonkim/CNN_sentence) is in Keras. [Denny Britz](https://github.com/dennybritz/cnn-text-classification-tf) has a TensorFlow implementation but it is too advanced for beginners. In addition, all the novice-friendly CNN codes are for images, giving me a real hard time translating the image context to text. 

Therefore, another goal of this script is to provide a beginner-friendly TensorFlow implementation of CNN on text. The script mimicked **[Hvass-Labs' CNN tutorial](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb) (here is the [video tutorial](https://www.youtube.com/watch?v=HMcx-zY8JSg&t=1503s); the whole series is HIGHLY RECOMMEND for TensoFlow/DeepLearning absolute beginners)**. If you unfortunately decide to pick up CNN from modeling text, hopefully this script can be of some help to you.