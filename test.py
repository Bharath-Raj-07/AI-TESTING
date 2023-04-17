import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the sample text
with open('sample.txt', 'r') as f:
    sample_text = f.read()

# load the student files
student_files = ['file1.txt', 'file2.txt', 'file3.txt']
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# vectorize the text
def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()

vectors = vectorize(student_notes + [sample_text])
s_vectors = list(zip(student_files, vectors[:-1]))
sample_vector = vectors[-1]

# calculate the similarity
def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])

plagiarism_results = {}

# check for plagiarism
def check_plagiarism():
    global s_vectors, sample_vector
    for student, text_vector in s_vectors:
        sim_score = similarity(text_vector, sample_vector)[0][1]
        plagiarism_results[student] = sim_score * 100
    return plagiarism_results

# print the results
for student, similarity in check_plagiarism().items():
    print(f"Plagiarism level for {student}: {similarity:.2f}%")