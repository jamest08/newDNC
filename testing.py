import numpy as np

# word = ['hello']
# words = np.array(['one', 'two'])
# words = np.append(words, word, axis=0)
# print(words)

segment = np.array([[1, 2, 3], [1, 2, 3]])
my_array = np.append(segment, [[1, 4, 6]])

print(my_array)