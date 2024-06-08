# import libraries
from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
    plt.show()

# Specify some code  output parameters
# specifying whether to plot or not
produce_plots = True

# save liklihood plots
save_likelihood_plots = True

# specify whether to print output to console
print_to_console = True

# Specify input parameters
# specifying input image
input_image = 'face1.jpeg'

# calculating the features
face_analysis = DeepFace.analyze(img_path=input_image,
                        actions=['age', 'gender', 'race', 'emotion'])


if produce_plots:
    # displaying the image
    image = mpimg.imread(input_image)
    plt.imshow(image)
    plt.title('Image under analysis') 
    plt.show(block=False)

    if save_likelihood_plots:
        save_file_name = 'LikelihoodPlots_' + input_image.split('.')[0] + '.png'
        # plotting bar charts of likelihoods
        plot_likelihood_plots(model_output=face_analysis, save_file_name=save_file_name)
    else:
        # plotting bar charts of likelihoods
        plot_likelihood_plots(model_output=face_analysis)


if print_to_console:    
    # printing the results to console
    print_output_console(model_output=face_analysis)
