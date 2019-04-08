import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance
import math
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib 
from sklearn.svm import SVC
from collections import Counter
import time

def get_data(path):
    assert isinstance(path, str)
    assert 'pickle' or 'pkl' in path
    return pickle.load(open(path,'rb'))

def barcode_to_names(barcode_file):
    assert isinstance(barcode_file, str)
    barcode_dict = {}
    with open(barcode_file,'r') as f:
        data = f.readlines()
        data = [x.split(',') for x in data]
    for x in data:
        barcode_dict[x[0]] = x[1].strip('\n')
    return barcode_dict

def get_name_given_barcode(barcode, barcode_file):
    assert isinstance(barcode_file, str)
    assert isinstance(barcode, str)

    barcode_dict = barcode_to_names(barcode_file)
    return barcode_dict[barcode]

def computer_kmeans(data, k, max_iter=1000, random_state=42):
    assert isinstance(data, np.ndarray)
    assert isinstance(k, int)
    assert k > 0
    assert isinstance(max_iter, int)
    assert isinstance(random_state, int)

    kmeans = KMeans(n_clusters = k, max_iter=1000, random_state=42).fit(train_X)
    return kmeans

def save_model(model, path, model_file_name):
    assert isinstance(path, str)
    assert isinstance(model_file_name, str)
    
    joblib.dump(model, os.path.join(path, model_file_name))
    print(f"Model saved at: {os.path.join(path, model_file_name)}")

def load_model(path):
    assert isinstance(path, str)

    return joblib.load(path)

def get_data_given_class(idx, X, Y):
    assert isinstance(idx, int)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)

    indices = [i for i, x in enumerate(Y) if x == idx]
    return X[indices]

def get_class_wise_data_dict(class_labels, X, Y):
    assert isinstance(class_labels, list)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    
    data_dict = {}
    for label in class_labels:
        label = int(label)
        data_dict[label] = get_data_given_class(label, X, Y)
    return data_dict

def save_class_wise_stats(save_file, num_classes, gt, preds, barcodes_file, barcode_to_names_file):
    assert isinstance(save_file, str)
    assert isinstance(num_classes, int)
    assert isinstance(gt, np.ndarray)
    assert isinstance(preds, np.ndarray)
    assert isinstance(barcodes, str)
    assert isinstance(barcode_to_name_dict, dict)
    
    barcodes = get_data(barcodes_file)
    with open(os.path.join(save_file), 'w') as f:
        header = 'barcode\tclass_name\tnum_clusters\n'
        f.write(header + '\n')
        for i in range(len(num_classes)):
            indices = np.where(gt == i)
            p = preds[indices]
            mode = stats.mode(p)[0][0]
            barcode = barcodes[i]
            name = get_name_given_barcode(barcode, barcodes_to_names_file)
            num_clusters = len(np.unique(p))
            f.write(str(barcode) + '\t' + str(name) + '\t' + str(num_clusters) + '\n')

def save_cluster_wise_stats(save_file, num_clusters, gt, preds, barcode_file, barcode_to_name_file):
    assert isinstance(save_file, str)
    assert isinstance(num_clusters, int)
    assert isinstance(gt, np.ndarray)
    assert isinstance(preds, np.ndarray)
    assert isinstance(barcode_file, str)
    assert isinstance(barcode_to_name_file, str)    

    barcodes = get_data(barcode_file)
    with open(save_file, 'w') as f:
        header = 'cluster_id\tclass_mode(barcode)\tclass_mode(name)\tmode\ttotal\tcluster_purity\tnum_unique\tclass_count'
        f.write(header + '\n')
        cluster_dict = {}
        for i in range(len(num_clusters)):
            indices = np.where(preds == i)
            cluster_dict[i] = gt[indices]
            unique_objects = np.unique(gt[indices])
            class_counters = Counter(gt[indices])
            top_3 = class_counters.most_common(3)
            top_3 = [(get_name_given_barcode(barcodes[x[0]], barcode_to_name_file)
                ,x[1]) for x in top_3]
            mode = stats.mode(preds[indices])[0][0]
            class_name = get_name_given_barcode(barcodes[mode], barcode_to_name_file)
            cluster_purity = class_counters[mode] / preds[indices].shape[0]
            f.write(str(i) + '\t' + str(int(barcodes[mode])) + '\t'+ str(class_name) +'\t'+ 
                str(top_3[0][1]) +'\t' + str(len(gt[indices])) + '\t' + 
                str(round(cluster_purity, 2)) + '\t' + str(num_clusters) + '\t' + str(top_3) + '\n')

def get_mean_vectors(class_wise_data_dict):
    assert isinstance(class_wise_data_dict, dict)

    mean_vectors = []
    for key, value in train_data_dict.items():
        mean_vector = np.mean(value, axis=0)
        mean_vectors.append(mean_vector)
    return np.array(mean_vectors)

def infer_using_mean_vector(feature_vector, mean_vectors):
    assert isinstance(feature_vector, np.ndarray)
    assert isinstance(mean_vectors, np.ndarray)
    dists = []
    for c in mean_vectors:
        dst = distance.euclidean(feature_vector, c)
        dists.append(dst)
    return np.argmin(dists)

def get_all_categories_models(path, num_category):
    assert isinstance(path, str)
    assert isinstance(num_category, int)
    assert num_category > 0

    models = []
    for i in range(num_category):
        model_path = os.path.join(path, 'category' + str(i+1) +'_meanvect_model.pkl')
        models.append(load_model(model_path))
    return models

def get_barcode_labels(path):
    assert isinstance(path, str)

    num_labels = len(os.listdir(path))
    category_labels = []
    for i in range(num_labels):
        labels_file_path = os.path.join(path, 'category_' + str(i+1) + '.txt')
        with open(labels_file_path, 'r') as f:
            labels = f.readlines() 
            labels = [x.strip('\n') for x in labels]
            category_labels.append(labels)
    return category_labels

def predict_single(data_vector, aisle_model, category_models, barcode_labels):
    assert isinstance(data_vector, np.ndarray)
    assert isinstance(category_models, list)
    assert isinstance(barcode_labels, list)

    aisle_pred = infer_using_mean_vector(data_vector, aisle_model)
    aisle_pred = 0
    category_model = category_models[aisle_pred]
    class_pred = infer_using_mean_vector(data_vector, category_model)
    barcode_pred = barcode_labels[aisle_pred][class_pred]
    return barcode_pred

def predict_end_to_end(X, aisle_model, category_models, barcode_labels):
    assert isinstance(X, np.ndarray)
    assert isinstance(category_models, list)
    assert isinstance(barcode_labels, list)

    preds = []
    for i in range(X.shape[0]):
        pred = predict_single(X[i], aisle_model, category_models, barcode_labels)
        preds.append(pred)
    return preds

def convert_labels_to_barcodes(Y, category, barcode_labels):
    assert isinstance(Y, (np.ndarray, list))
    assert isinstance(category, int)
    assert category > 0
    assert isinstance(barcode_labels, list)

    labels = barcode_labels[category - 1]
    
    barcodes = []
    for y in Y:
        barcodes.append(labels[y])
    return barcodes

def evaluate(preds, gt):
    assert isinstance(preds, list)
    assert isinstance(gt, list)
    assert len(preds) == len(gt)

    total = len(gt)
    correct = 0
    for y, pred in zip(gt, preds):
        if y == pred:
            correct += 1
    return correct, (correct / total)

category_names = ['Laundry','Biscuits','Cereals and Tea','Snacks and Kitchen Items','Hair Products', 'Beauty Products', 'Soaps','toothbrush_and_toothpaste']

correct = 0
total = 0
for i in range(len(category_names)):
    category = i + 1
    category_name = category_names[i]

    print(f'{category}: {category_name}')
    train_path = 'data/train_features'
    valid_path = 'data/valid_features'
    logs_root = 'logs/'
    model_save_path = 'models'
    barcode_file_path = 'data/barcodes.txt'
    labels_file = 'data/labels.pkl'
    ncm_model_path = 'ncm_models'
    barcode_labels_path = 'Labels'
    
    train_data_path = os.path.join(train_path, 'category' + str(category) + '_X.pickle')
    train_labels_path = os.path.join(train_path, 'category' + str(category) + '_Y.pickle')

    valid_data_path = os.path.join(valid_path, 'category' + str(category) + '_X.pickle')
    valid_labels_path = os.path.join(valid_path, 'category' + str(category) + '_Y.pickle')
    
    # train_data_path = 'data/features_8_aisle/train_X.pickle'
    # train_labels_path = 'data/features_8_aisle/train_Y.pickle'
    # valid_data_path = 'data/features_8_aisle/valid_X.pickle'
    # valid_labels_path = 'data/features_8_aisle/valid_Y.pickle'
 
    train_X = get_data(train_data_path)
    train_Y = get_data(train_labels_path)

    valid_X = get_data(valid_data_path)
    valid_Y = get_data(valid_labels_path)
    
    unique_labels = list(set(train_Y.tolist()))
    print(f"No. of unique labels: {len(unique_labels)}")
    
    barcode_to_name_dict = barcode_to_names(barcode_file_path)
    barcodes = get_data(labels_file)

    # kmeans = load_model(os.path.join(model_save_path, 'category' + str(category) + '_model.pkl'))
    k = math.ceil(math.log(len(unique_labels), 2))
    start = time.time()
    kmeans = computer_kmeans(train_X, k=k)
    print(f'Training took {time.time()-start}s')
    save_model(kmeans, model_save_path, 'category' + str(category) + '_model.pkl')
    continue
    train_data_dict = get_class_wise_data_dict(unique_labels, train_X, train_Y)
    valid_data_dict = get_class_wise_data_dict(unique_labels, valid_X, valid_Y)

    aisle_model = load_model(os.path.join(ncm_model_path, '8_aisle_meanvect_model.pkl'))
    category_models = get_all_categories_models(ncm_model_path, len(category_names)) 
    barcode_labels = get_barcode_labels(barcode_labels_path)
    train_barcode_Y = convert_labels_to_barcodes(train_Y, category,barcode_labels)

    train_preds = predict_end_to_end(train_X, aisle_model, category_models, barcode_labels)
    correct, accuracy_score = evaluate(train_preds, train_barcode_Y)
    print(correct)
    print(accuracy_score)
    assert False
    continue

    # evaluate_end_to_end(trainX, trainY, aisle_model, category_models, category, barcode_labels)
    # predict_end_to_end(trainX, train_Y, aisle_model, category_models)

    print(f'Training took {time.time() - start} ms')
    # save_model(mean_vectors, 'ncm_models', 'category' + str(category) + '_meanvect_model.pkl')
    # save_model(mean_vectors, 'ncm_models', '8_aisle_' + '_meanvect_model.pkl')

    
    train_correct = 0

    start = time.time()
    for key, value in train_data_dict.items():
        correct = 0
        for v in value:
            pred = infer_using_mean_vector(v, mean_vectors)
            if pred == key:
                correct += 1
        train_correct += correct
    print(f'Evaluation on train data took {(time.time() - start)}s')
    print(f"Correct: {train_correct}, Total:{train_Y.shape[0]}")
    print(f"Train Accuracy: {(train_correct/ train_Y.shape[0])*100}")

    valid_correct = 0
    start = time.time()
    for key, value in valid_data_dict.items():
        correct = 0
        for v in value:
            pred = infer_using_mean_vector(v, mean_vectors)
            if pred == key:
                correct += 1
        valid_correct += correct
    print(f'Evaluation on valid data took {(time.time() - start)}s')
    print(f"Correct: {valid_correct}, Total:{valid_Y.shape[0]}")
    print(f"Valid Accuracy: {(valid_correct / valid_Y.shape[0])*100}")
    continue
    
    # get predictions from clusters
    train_preds = kmeans.predict(train_X)
    cluster_centers = kmeans.cluster_centers_
    
    # save_class_wise_stats(os.path.join(logs_root,'470_items_class_statistics.csv'),
        # len(unique_labels), train_Y, train_preds, barcodes, barcode_to_name_dict)
    

        # dists = []
        # for c in cluster_centers:
        #     dst = distance.euclidean(mean_vector, c)
        #     dists.append(dst)
        # cluster = np.argmin(dists)
        # labels_dict[key] = cluster
    
    train_correct = 0  
    for key, value in train_data_dict.items():
        correct = 0
        for v in value:
            dists = []
            for c in mean_vectors:
                dst = distance.euclidean(v, c)
                dists.append(dst)
            cluster = np.argmin(dists)
            if cluster == key:
                correct += 1
        train_correct += correct

    valid_correct = 0
    for key, value in valid_data_dict.items():
        correct = 0
        for v in value:
            dists = []
            for c in mean_vectors:
                dst = distance.euclidean(v, c)
                dists.append(dst)
            cluster = np.argmin(dists)
            if cluster == key:
                correct += 1
        valid_correct += correct

    train_accuracy = (train_correct / train_Y.shape[0]) * 100
    valid_accuracy = (valid_correct / valid_Y.shape[0]) * 100

    # preds_train = kmeans.predict(train_X)
    # preds_val = kmeans.predict(valid_X)

    print('****Training Statistics*****')
    print(f"Total data points: {train_Y.shape[0]}")
    print(f'Correct: {train_correct}')
    print(f"Training Accuracy: {train_accuracy}%")

    print('\n****Validation Statistics*****')
    print(f"Total data points: {valid_Y.shape[0]}")
    print(f'Correct: {valid_correct}')
    print(f"Validation Accuracy: {valid_accuracy}%\n\n")

    with open(os.path.join(logs_root, 'category' + str(category) + '.txt'), 'w') as f:
        f.write(f'Category ID: {category}\n')
        f.write(f'Category Name: {category_name}\n')
        f.write('***Training Statistics***\n')
        f.write(f'Total : {train_Y.shape[0]}\n')
        f.write(f'Correct : {train_correct}\n')
        f.write(f'Training Accuracy: {train_accuracy}%\n\n\n')

        f.write('***Validation Statistics***\n')
        f.write(f'Total : {valid_Y.shape[0]}\n')
        f.write(f'Correct : {valid_correct}\n')
        f.write(f'Valid Accuracy: {valid_accuracy}%\n')
        f.write(f'Labels -> cluster: {labels_dict}\n\n')
