import cv2
import numpy as np
import sys

def task_01(filename):
    #read the image
    original_image = cv2.imread(filename)

    #Rescale  the image to a size comparable to VGA size
    rescaled_img = rescale_image(original_image)

    rescaled_copy = rescaled_img.copy()

    # Keypoint extraction from SIFT
    keypoints, y_channel, des = extract_sift_descriptors(rescaled_img)

    #Draw markers 
    # Draws circles around the detected keypoints, showing their size and orientation.
    img = cv2.drawKeypoints(rescaled_img, keypoints, outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #draw marker(+) on the detected keypoint and orientation image
    for key_point in keypoints:
        img = cv2.drawMarker(img,(int(key_point.pt[0]), int(key_point.pt[1])), (0, 255, 0),markerType=cv2.MARKER_CROSS,markerSize=6,thickness=1,line_type=8)
    
    #combine images to visualise scaled and image with highlighted keypoints
    combined_image = np.hstack((rescaled_copy, img))

    # Print the count of keypoints
    count_keypoints = len(keypoints)
    print("# of keypoints in {0} is {1}".format(filename,count_keypoints))

    # disply the combined image
    cv2.imshow('Rescaled and highlighted keypoint image', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def task_02(filename):
    all_descriptors = []
    all_key_points = []
    img_arr = []
    descriptors_dict = {}
    for file in filename:
        img_arr.append(file)

        # read img file
        original_image = cv2.imread(file)

        #rescale the image
        rescaled_image= rescale_image(original_image)
        rescaled_copy = rescaled_image.copy()

        #sift feature extraction
        keypoints, y_channel, des = extract_sift_descriptors(rescaled_image)

        # Saves the descriptors and the number of keypoints for each image.
        all_descriptors.append(des)
        all_key_points.append(len(keypoints))

        # map each image file name to its corresponding descriptors.
        descriptors_dict[file] = des

        # Print the number of keypoints for each image
        print('# of keypoints in {0} is {1}'.format(file,len(keypoints)))  

    # Vertically stack the arrays to create a single array (single dataset of descriptors for clustering)
    desc_array = np.vstack(all_descriptors)

    #Define K values and apply k means
    k_values = [0.05, 0.1, 0.2]

    # To stores the results of the K-Means clustering.
    k_means_out_dict = {}

    for k in k_values:
        compactness, label, center = kmeans(desc_array, k)
        k_means_out_dict[k] = (compactness, label, center)

    # Initializes two dictionaries to store image labels and histograms
    image_label_dict = {}
    histogram_dict = {}

    for k in k_values:
        # Assign labels to images based on K-Means clustering results
        image_label_dict[k] = assign_label(all_labels=k_means_out_dict[k][1], descript_dict=all_descriptors)

        # Find the total number of clusters by getting the maximum label
        total_clusters = k_means_out_dict[k][1].max()

        histogram_dict[k] = {}

        for i in range(len(image_label_dict[k])):
            # Create a histogram from the assigned labels for the current image
            histogram_dict[k][i] = compute_normalized_histogram(image_label_dict[k][i], total_clusters)

    # calculate chi squared distance for normalised histograms
    for k in k_values:
        print("K = {0}% * (total number of keypoints) = {1}".format((k * 100),k_means_out_dict[k][1].shape[0]))
        print()

        print("Dissimilarity Matrix")
        chi = np.zeros((len(image_label_dict[k]), len(image_label_dict[k])))

        header = img_arr
        for image in range(len(image_label_dict[k])):
            out_line = [image]
            for img in range(len(image_label_dict[k])):
                chi[image][img] = cal_chi_square_distance(histogram_dict[k][image], histogram_dict[k][img])
        display_matrix_data(header, chi)
        
def rescale_image(image):
    target_width = 600
    target_height = 480
    # Get the original image dimensions
    height, width, _ = image.shape

    # Calculate the scaling factors for width and height while preserving the aspect ratio
    width_ratio = target_width / width
    height_ratio = target_height / height

    # Choose the smaller scaling factor to ensure the entire image fits within the target resolution
    scaling_factor = min(width_ratio, height_ratio)

    # Calculate the new dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image to the new dimensions while preserving the aspect ratio
    rescaled_image = cv2.resize(image, (new_width, new_height))

    return rescaled_image

def extract_sift_descriptors(image):

    # convert image to YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    #extract Y vomponent
    y_channel = yuv[:, :, 0]

    #Initialise sift detector 
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors in the Y channel
    keypoints, des = sift.detectAndCompute(y_channel, None)
    return keypoints, y_channel, des

def display_matrix_data(headers, matrix):
    desired_col_width = 10  # Set the desired column width

    # Calculate maximum width for each column, considering headers
    max_col_widths = []
    for header, col in zip(headers, zip(*matrix)):
        max_width = max(len(str(header)), *(len(str(x)) for x in col))
        max_col_widths.append(max(desired_col_width, max_width))

    # Print the header row
    print(f'{"":<{max_col_widths[0]}}', end='')  # Print an empty cell for the top-left corner
    for header in headers:
        print(f'{header:<{max_col_widths[0] + 3}}', end='')  # Print headers
    print()  # Move to the next line

    #   Print each row of the matrix
    for i, row in enumerate(matrix):
        print(f'{headers[i]:<{max_col_widths[0]}}', end='')  # Print the row header
        for j, val in enumerate(row):
            print(f'{val:<{max_col_widths[j] + 3}}', end='')  # Print the matrix values
        print()  # Move to the next line

def cal_chi_square_distance(his1, his2):

    # compare the histograms with chi-squared distance
    numerator = (his1 - his2) ** 2
    denomenator = (his1 + his2)

    chi_squared = np.divide(numerator,denomenator,out=np.zeros_like(numerator), where=denomenator!=0.0)
    chi_dist = 0.5 * np.nansum(chi_squared)

    return "{:.2f}".format(chi_dist)

def compute_normalized_histogram(labeled_descript, nbins):
    # calculate normalised number of samples per each bin
    histo, edges = np.histogram(labeled_descript, nbins, density=True)
    return np.float32(histo)

def assign_label(all_labels, descript_dict):
    label_img = []
    start_index = 0
    end_index = 0
    for i in descript_dict:
        # Update the end index by adding the length of the current descriptor array
        end_index += len(i)
        
        # Slice the all_labels array to get labels for the current image
        label = all_labels[start_index:end_index]
        
        # Update the start index for the next image
        start_index = end_index
        
        # Append the labels for the current image to the label_img list
        label_img.append(label)

    return label_img

def kmeans(des, k_percent):

    #kmeans function in OpenCV requires the input data to be in float32 format
    des = np.float32(des)

    # get the K values as percentage of the total desc (get how many clusters to use)
    k = int(des.shape[0] * k_percent)

    # define criteria and apply kmeans()
    # algorithm will stop if either: Maximum number of iterations (10) is reached, or Accuracy (0.1) is achieved
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    # Runs K-Means to cluster the descriptors
    compactness, label, center = cv2.kmeans(des, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return compactness, label, center

try:
    # Check if exactly one argument is provided
    if len(sys.argv) == 2:
        file_name = sys.argv[1]  
        task_01(file_name)  

    # Check if more than one argument is provided
    elif len(sys.argv) > 2:
        file_names = sys.argv[1:]  
        task_02(file_names)  
    else:
        print("No files provided. Please provide at least one file name.")

except Exception as e:
    print(f"An error occurred: {e}")
