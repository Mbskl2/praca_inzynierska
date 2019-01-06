"""Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html
Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

################ Spelling Corrector 

import re
from collections import Counter

def readWords(text):
    "Read all words from the text"
    return re.findall(r'[a-z]+', text.lower())

"Read the dictionary data base from big.txt"
WORDS = Counter(readWords(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correct(word): 
    "Most probable spelling correction for word."
    if word.isnumeric():
        return word # Numbers won't be corrected
    return max(candidates(word.lower()), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (getKnownWords([word]) or getKnownWords(editOnce(word)) or getKnownWords(editTwice(word)) or [word])

def getKnownWords(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    knownWords = set()
    for word in words:
        if word in WORDS:
            knownWords.add(word)
    return knownWords
            
def getSplits(word):
    splits = []
    for i in range(len(word) + 1):
        splits.append((word[:i], word[i:]))   
    return splits

def getDeletes(splits):
    deletes = []
    for L, R in splits:
        if len(R) >= 1:
            deletes.append(L + R[1:])
    return deletes

def getTransposes(splits):
    transposes = []
    for L, R in splits:
        if len(R) >= 2:
            transposes.append(L + R[1] + R[0] + R[2:])
    return transposes

def getReplaces(splits):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    replaces = []
    for L, R in splits:
        if len(R) >= 1:
            for c in letters:
                replaces.append(L + c + R[1:])
    return replaces

def getInserts(splits):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    inserts = []
    for L, R in splits:
        for c in letters:
            inserts.append(L + c + R)
    return inserts

def replaceDigitsWithSimilarLetters(word):
    words = set([word])
    if '2' in word:
        word = word.replace('2', 'z')
    if '3' in word:
        word = word.replace('3', 'b')
    if '4' in word:
        word = word.replace('4', 'a')
    if '5' in word:
        word = word.replace('5', 's')
    if '7' in word:
        word = word.replace('7', 'j')
    if '8' in word:
        word = word.replace('8', 'b')
    if '1' in word:
        word = word.replace('1', 'i')
        words.add(word)
        words.add(word.replace('1', 'l'))
    if '6' in word:
        word = word.replace('6', 'b')
        words.add(word)
        words.add(word.replace('6', 'h')) 
    if '9' in word:
        word = word.replace('9', 'g')
        words.add(word)
        words.add(word.replace('9', 'q'))  
    if '0' in word:
        word = word.replace('0', 'o')
        words.add(word)
        words.add(word.replace('0', 'q')) 
        
    words.add(word)
    return words.copy()

def editOnce(word):
    "All edits that are one edit away from `word`."
    allProposals = set()
    for oneWord in replaceDigitsWithSimilarLetters(word):
        splits = getSplits(oneWord)
         
        allProposals.update(getDeletes(splits)
                             + getTransposes(splits)
                             + getReplaces(splits) 
                             + getInserts(splits))
    return allProposals

def editTwice(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in editOnce(word) for e2 in editOnce(e1))

if __name__ == '__main__':
    print('Correct way to use:')
    print('import spell')
    print("spell.correct('speling')")
    
