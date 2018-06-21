'''
Code to filter out the relevant images from the MPII human pose dataset
 we are only interested in images of sports
@author: Dominique Cheray
'''

import scipy.io as spio
import matlab.enginge

#load the matlab annotations
annotations = spio.loadmat('mpii_human_pose_v1_u12_1.mat')
# get train/test assginment of images. Since for the test images there are no
# annotations provided we want to filter them out
train_test_asignment = annotations['RELEASE']['img_train'][0][0][0]
images = open("image_names.txt", "w")
labels = open("labels.txt", "w")
# now iterate over all images
for i in range(len(train_test_asignment)):
    # if it is a train image
    if (train_test_asignment[i] == 1):
        # if its category is sports
        try:
            if(annotations['RELEASE']['act'][0][0][i][0][0][0] == 'sports'):
                # write image name into file
                images.write("{} \n".format(annotations['RELEASE']['annolist'][0][0][0][i][0][0][0][0][0]))
                # write activity name into file
                labels.write("{} \n".format(annotations['RELEASE']['act'][0][0][i][0][1][0]))
        except IndexError:
            # some images are marked as train images but they still have no labels
            # so just skip them
            continue
images.close()
labels.close()
            

# to get category name  annotations['RELEASE']['act'][0][0][i][0][0][0]
# to get activity name annotations['RELEASE']['act'][0][0][i][0][1][0]
# to get image name annotations['RELEASE']['annolist'][0][0][0][i][0][0][0][0][0]

# since the classes are fairly unbalanced let's filter out 10 different classes of
# with a similar size

# first read the images and lables previously saved
all_images_file = open('image_names.txt')
all_images = all_images_file.read().splitlines()
all_images_file.close()
all_lables_file = open('labels.txt')
all_lables = all_lables_file.read().splitlines()
all_lables_file.close()

final_images = open("final_images.txt", "w")
final_labels = open("final_labels.txt", "w")
# now iterate over the labels and for all those classes we want to have
# save the image name and an int label for the class
for i in range(len(all_lables)):
    if('basketball' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(0))
    if('horseback riding' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(1))
    if('martial arts' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(2))
    if('paddleball' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(3))
    if('rock climbing' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(4))
    if('rope skipping' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(5))
    if('skateboarding' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(6))
    if('softball' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(7))
    if('tennis' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(8))
    if('golf' in all_lables[i]):
        final_images.write("{} \n".format(all_images[i]))
        final_labels.write("{} \n".format(9))

final_labels.close()
final_images.close()

# and finally save all the images we are keeping in a folder
keep_images_file = open("final_images.txt")
keep_images = keep_images_file.read().splitlines()
keep_images_file.close()
print(len(keep_images))
