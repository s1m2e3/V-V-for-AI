import subprocess
import itertools


if __name__=="__main__":

    #  2 types of corruptions will be done: change in brightness as means to represent  changes in the lenses and change in the resolution as means to represent changes in the sensor

    corruption = ["lense_crush_directed"]
    # corruption = ["resolution_change_2x","resolution_change_x_2"]
    #corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    corruptions = ["--corruption_name "+corrupted for corrupted in corruption ]
    #severity = ["2","3","4","5"]
    #severity = ["2"]
    # severity = ["--severity "+severities for severities in severity]
    # corrupt_train = ["False"]
    # corrupt_train = ["True","False"]
    # corrupt_train = ["--apply_on_train " + train for train in corrupt_train]
    # corrupt_test = ["False"]
    # corrupt_test = ["--apply_on_test " + test for test in corrupt_test]
    arguments = []
    
    for element in itertools.product(*[corruptions]):
         arguments.append(element)
    name ="python corrupt.py " 
    
    for quad in arguments:
        print("first_call")
        quad = " ".join(quad)
        final = name+quad
        print(final)
        subprocess.call(final)
        print("finish_call")