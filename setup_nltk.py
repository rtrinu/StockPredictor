import nltk

resources = [
    'vader_lexicon'
]

for resource in resources:
    nltk.download(resource)

print("NLTK Setup Completed.")