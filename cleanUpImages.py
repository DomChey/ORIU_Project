'''
Code to filter out the relevant images from the MPII human pose dataset
 we are only interested in images of sports
@author: Dominique Cheray
'''

import scipy.io as spio
from PIL import Image

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
                images.write("{}\n".format(annotations['RELEASE']['annolist'][0][0][0][i][0][0][0][0][0]))
                # write activity name into file
                labels.write("{}\n".format(annotations['RELEASE']['act'][0][0][i][0][1][0]))
        except IndexError:
            # some images are marked as train images but they still have no labels
            # so just skip them
            continue
images.close()
labels.close()
            

# to get category name  annotations['RELEASE']['act'][0][0][i][0][0][0]
# to get activity name annotations['RELEASE']['act'][0][0][i][0][1][0]
# to get image name annotations['RELEASE']['annolist'][0][0][0][i][0][0][0][0][0]

# since the classes are fairly unbalanced let's filter out 10 different classes
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
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(0))
    elif('horseback riding' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(1))
    elif('martial arts' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(2))
    elif('paddleball' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(3))
    elif('rock climbing' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(4))
    elif('rope skipping' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(5))
    elif('skateboarding' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(6))
    elif('softball' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(7))
    elif('tennis' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(8))
    elif('golf' in all_lables[i]):
        final_images.write("{}\n".format(all_images[i]))
        final_labels.write("{}\n".format(9))

final_labels.close()
final_images.close()

# and finally save all the images we are keeping in a folder
keep_images_file = open("final_images.txt")
keep_images = keep_images_file.read().splitlines()
keep_images_file.close()
for i in range(len(keep_images)):
    img = Image.open("/home/domi/Downloads/images/{}".format(keep_images[i]))
    img.save("images/{}".format(keep_images[i]))


# assign 20% of all images of one class to the test set
keep_images_file = open("final_images.txt")
keep_images = keep_images_file.read().splitlines()
keep_images_file.close()
keep_labels_file = open("final_labels.txt")
keep_labels = keep_labels_file.read().splitlines()
keep_labels_file.close()

# open files to write down the final train test splits
train_images = open("images/train_images.txt", "w")
test_images = open("images/test_images.txt", "w")
train_labels = open("images/train_labels.txt", "w")
test_labels = open("images/test_labels.txt", "w")

# to keep track how many instances of one class are already assigned to test
num_0 = num_1 = num_2 = num_3 = num_4 = num_5 = num_6 = num_7 = num_8 = num_9 = 0
for i in range(len(keep_labels)):
    if(keep_labels[i] == "0"):
        if (num_0 < 35):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_0 = num_0 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "1"):
        if (num_1 < 27):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_1 = num_1 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "2"):
        if (num_2 < 26):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_2 = num_2 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "3"):
        if (num_3 < 25):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_3 = num_3 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "4"):
        if (num_4 < 30):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_4 = num_4 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "5"):
        if (num_5 < 32):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_5 = num_5 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "6"):
        if (num_6 < 36):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_6 = num_6 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "7"):
        if (num_7 < 35):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_7 = num_7 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "8"):
        if (num_8 < 35):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_8 = num_8 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))
    elif(keep_labels[i] == "9"):
        if (num_9 < 29):
            test_images.write("{}\n".format(keep_images[i]))
            test_labels.write("{}\n".format(keep_labels[i]))
            num_9 = num_9 + 1
        else:
            train_images.write("{}\n".format(keep_images[i]))
            train_labels.write("{}\n".format(keep_labels[i]))

train_images.close()
train_labels.close()
test_images.close()
test_labels.close()
