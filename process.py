import glob
import numpy as np
import cv2
import pickle
import os
import h5py
import random

PROCESS_TEST_IMAGES = True
DO_TRAIN = False
DO_GRID_SEARCH = False

from skimage.feature import hog
import time
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip
from collections import deque
from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

if DO_TRAIN:
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 0 # Can be 0, 1, 2, or "ALL"
    spatial_size = (64, 64) # Spatial binning dimensions
    hist_bins = 64    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    # Read in cars and notcars
    cars = glob.glob('training/vehicles/**/*.png', recursive=True)
    notcars = glob.glob('training/non-vehicles/**/*.png', recursive=True)
    print("cars = {}, notcars = {}".format(len(cars), len(notcars)))
    
    print("extracting car features...")
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print("extracting non-car features...")
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    print("normalizing and splitting...")
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    if DO_GRID_SEARCH:
        # Use a linear SVC 
        svr = svm.SVC()
        # Grid search
        parameters = {'kernel':('linear', 'rbf'), 'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        svc = GridSearchCV(svr, parameters, n_jobs=25, verbose=100)
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
    else:
        svc = svm.SVC(C=1.0, kernel='rbf')
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    with open("parameters.pkl","wb") as param_file:
        pickle.dump(svc, param_file)
        pickle.dump(color_space, param_file)
        pickle.dump(orient, param_file)
        pickle.dump(pix_per_cell, param_file)
        pickle.dump(cell_per_block, param_file)
        pickle.dump(hog_channel, param_file)
        pickle.dump(spatial_size, param_file)
        pickle.dump(hist_bins, param_file)
        pickle.dump(spatial_feat, param_file)
        pickle.dump(hist_feat, param_file)
        pickle.dump(hog_feat, param_file)
        pickle.dump(X_scaler, param_file)
    param_file.close()


# # Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, hog_channel,
                        spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img_tosearch = draw_img[ystart:ystop,xstart:xstop,:]

    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(img_tosearch)      

    if scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    # Define blocks and steps as above
    nxblocks = (feature_image.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (feature_image.shape[0] // pix_per_cell) - cell_per_block + 1 

    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    if hog_feat == True:
        if hog_channel == 'ALL':
            full_hog_features = []
            for channel in range(feature_image.shape[2]):
                full_hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=False))
        else:
            full_hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=False)

    
    found_cars = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            window_features = []

            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))

            if spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                window_features.append(spatial_features)

            if hist_feat == True:
                hist_features = color_hist(subimg, nbins=hist_bins)
                window_features.append(hist_features)

            # Extract HOG for this patch
            if hog_feat == True:
                if hog_channel =='ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(full_hog_features[channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                    hog_features = np.hstack((hog_features))
                else:
                    hog_features = full_hog_features[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

                window_features.append(hog_features)

            window_features = X_scaler.transform(np.hstack((window_features)).reshape(1, -1))
            test_prediction = svc.predict(window_features)

            xbox_left = np.int(xleft*scale) + xstart
            ybox_top = np.int(ytop*scale) + ystart
            win_draw = np.int(window*scale)
            if test_prediction == 1:
                found_cars.append(((xbox_left, ybox_top), (xbox_left+win_draw, ybox_top+win_draw)))
    return found_cars

print("Loading parameters...")
with open("parameters.pkl","rb") as param_file:
    svc = pickle.load(param_file)
    color_space = pickle.load(param_file)
    orient = pickle.load(param_file)
    pix_per_cell = pickle.load(param_file)
    cell_per_block = pickle.load(param_file)
    hog_channel = pickle.load(param_file)
    spatial_size = pickle.load(param_file)
    hist_bins = pickle.load(param_file)
    spatial_feat = pickle.load(param_file)
    hist_feat = pickle.load(param_file)
    hog_feat = pickle.load(param_file)
    X_scaler = pickle.load(param_file)
param_file.close()

print("Using parameters: {}".format(svc.get_params()))

heatmap_threashold = 10
box_queue = deque(maxlen=heatmap_threashold)

def update_heatmap(img):
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    for boxes in box_queue:
        # Iterate through list of bboxes
        for box in boxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_image(img):
    if PROCESS_TEST_IMAGES == False:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    all_cars = []
    xstart = 535
    xstop = 1279
    ystart = 405
    ystop = 485
    scale = 1
    all_cars.extend(find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, hog_channel,
                        spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    xstart = 479
    xstop = 1279
    ystart = 428
    ystop = 553
    scale = 1.5625
    all_cars.extend(find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, hog_channel,
                        spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    xstart = 559
    xstop = 1279
    ystart = 386
    ystop = 566
    scale = 2.25
    all_cars.extend(find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, hog_channel,
                        spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    xstart = 767
    xstop = 1279
    ystart = 374
    ystop = 694
    scale = 4
    all_cars.extend(find_cars(img, color_space, xstart, xstop, ystart, ystop, scale, hog_channel,
                        spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    
    if PROCESS_TEST_IMAGES == False:
        box_queue.append(all_cars)
        heatmap = update_heatmap(img)
        heatmap = apply_threshold(heatmap, heatmap_threashold)
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    else:
        draw_img = np.copy(img)
        for bbox in all_cars:
            cv2.rectangle(draw_img, bbox[0], bbox[1], (255,0,0), 1)

    return draw_img



if PROCESS_TEST_IMAGES:
    for filename in glob.glob('test_images/*.jpg'):
        print(filename)
        img = cv2.imread(filename)
        out_img = process_image(img)
        # out_img = find_cars(img, color_space, ystart, ystop, scale, hog_channel,
        #                 spatial_feat, hist_feat, hog_feat, svc, X_scaler, 
        #                 orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        cv2.imwrite('output_images/{}'.format(os.path.basename(filename)), out_img)
else:
    clip2 = VideoFileClip("project_video.mp4")
    vid_clip = clip2.fl_image(process_image)
    vid_clip.write_videofile("output_videos/project_video.mp4", audio=False)
