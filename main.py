import letter_segmentation as ls
import classification
import spellingCorrector as sc
import sys

def main(image_path, folder_path):
    ls.segment(image_path, folder_path)
    letters = classification.run(folder_path)
    print('Word: ', letters)
    #print('Word before correction: ', letters)
    #word = sc.correct(letters)
    #print('Word after correction: ', word)
   
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Arguments are missing: script.py image_path path_to_save")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
