import os
import numpy as np 

fn = 'data/glove.840B.300d.txt'

char_vecSum = {}
with open(fn, 'rb') as f:
	for line in f:
		line_split = line.strip().split(" ")
		word_vec = np.array(line_split[1:], dtype=float)
		word = line_split[0]

		for char in word:
			#subset to normal ASCII range
			if ord(char) < 128:
				if char in char_vecSum:
					char_vecSum[char] = (char_vecSum[char][0] + vec, char_vecSum[char][1] + 1)
				else:
					char_vecSum[char] = (vec,1)

new_file = os.path.splitext(os.path.basename(fn))[0] + '-char_embeds.txt'
with open(new_file, 'wb') as file:
	for char in char_vecSum:
		#get mean of word vectors for each character occurrence in vocabulary
		mean_vec = np.round(char_vecSum[char][0] / char_vecSum[char][1],6).tolist()
		file.write(char + ' ' + ' '.join(str(i) for i in mean_vec) + "\n")
