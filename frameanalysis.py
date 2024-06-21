# import libraries
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from os import listdir

# define some helper functions
def print_output_console(model_output: list):
    for person in model_output:
        print(f'The age of the person = {person['age']} \n')
        print(f'The dominant gender is {person['dominant_gender']}') 
        print(f'with a likelihood of {person['gender'][person['dominant_gender']]}')
        print(f'The dominant race is {person['dominant_race']}')
        print(f'with a likelihood of {person['race'][person['dominant_race']]}')
        print(f'The dominant emotion is {person['dominant_emotion']}')
        print(f'with a likelihood of {person['emotion'][person['dominant_emotion']]}')

def dictionary_to_histogram(dictionary: dict):
    plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
    plt.xticks(range(len(dictionary)), list(dictionary.keys()))

def plot_likelihood_plots(model_output: list, save_file_name: str =None):
    identifiers = ['gender', 'race', 'emotion']
    plt.figure()
    for person in model_output:
        for index, identifier in enumerate(identifiers, start=1):
            plt.subplot(1,len(identifiers), index)
            dictionary_to_histogram(dictionary=person[identifier])
            plt.xticks(rotation=45, horizontalalignment='right', fontsize=8)
            plt.title((identifier + ' ' +'likelihoods'), fontsize=10)
    plt.tight_layout()
    if save_file_name:
        plt.savefig(save_file_name, bbox_inches='tight')
    plt.show(block=False)

def plot_image_annotates(model_output: list, image_path: str, save_file_name: str =None):
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.set_title('Image under analysis') 
    for person in model_output:
        bbx = person['region']['x']
        bby = person['region']['y']
        bbw = person['region']['w']
        bbh = person['region']['h']
        x = person['region']['x'] - person['region']['w']/2
        y = person['region']['y'] + person['region']['h']/2
        age = person['age']
        gender = person['dominant_gender']
        gender_likelihood = person['gender'][person['dominant_gender']]
        race = person['dominant_race']
        race_likelihood = person['race'][person['dominant_race']]
        emotion = person['dominant_emotion']
        emotion_likelihood = person['emotion'][person['dominant_emotion']]
        s=f'Age = {age}\nGender = {gender},{gender_likelihood:.2f}\nRace = {race},{race_likelihood:.2f}\nEmotion = {emotion},{emotion_likelihood:.2f}'
        ax.text(x, y, s, bbox=dict(fill=False, edgecolor='red', linewidth=1),
                c='g', fontsize='xx-small')
        face_bbox = patches.Rectangle((bbx, bby), bbw, bbh, linewidth=0.5, edgecolor='r', facecolor='none')
        ax.add_patch(face_bbox)
    if save_file_name:
        ax.savefig(save_file_name, bbox_inches='tight')
    ax.show(block=False)

# Specify some code  output parameters
# specifying whether to plot or not
produce_plots = True
save_plots = True
print_to_console = True

for input_image in listdir('images'):
    input_image_path = 'images/' + input_image 
    # calculating the features
    face_analysis = DeepFace.analyze(img_path=input_image_path,
                            actions=['age', 'gender', 'race', 'emotion'], 
                            detector_backend='mediapipe')


    if produce_plots and save_plots:
        save_image_filename = 'results/Annotated_' + input_image.split('.')[0] + '.png' 
        plot_image_annotates(model_output=face_analysis, image_path=input_image_path, save_file_name=save_image_filename)
        
        
        save_likelihood_filename = 'results/LikelihoodPlots_' + input_image.split('.')[0] + '.png'
        plot_likelihood_plots(model_output=face_analysis, save_file_name=save_likelihood_filename)
    elif produce_plots:
        plot_image_annotates(model_output=face_analysis, image_path=input_image_path)
        plot_likelihood_plots(model_output=face_analysis)


    if print_to_console:    
        # printing the results to console
        print_output_console(model_output=face_analysis)
