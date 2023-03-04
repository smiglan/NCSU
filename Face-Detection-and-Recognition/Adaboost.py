
import numpy as np
import pickle
import math
import time
from sklearn.feature_selection import SelectPercentile, f_classif
# from google.colab import drive
from skimage.feature import draw_haar_like_feature
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt

# drive.mount('/content/gdrive', force_remount=True)
# path = '/content/gdrive/My Drive/ECE763P2/'
path = ''
#make sure the training and test data as pickle files is in the same directory
def integral(image):
    integrali = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
          if y-1>=0:
            s[y][x] = s[y-1][x] + image[y][x]
          else:
            s[y][x] = image[y][x]
          if x-1 >= 0:            
            integrali[y][x] = integrali[y][x-1]+s[y][x]
          else:
           integrali[y][x] = s[y][x]
    return integrali

class RR:
  def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

  def compute(self, ii):
    value = ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])
    return value
def train(X, y, features, weights):
      total_pos, total_neg = 0, 0
      for w, label in zip(weights, y):
          if label == 1:
              total_pos = total_pos+ w
          else:
              total_neg = total_neg+ w

      classifiers = []
      total_features = X.shape[0]
      for index, feature in enumerate(X):

          applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
          pos_seen, neg_seen = 0, 0
          pos_weights, neg_weights = 0, 0
          min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
          for w, f, label in applied_feature:
              error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
              if error < min_error:
                  min_error = error
                  best_feature = features[index]
                  best_threshold = f
                  if pos_seen > neg_seen:
                    best_polarity = 1  
                  else:
                    best_polarity = -1

              if label == 1:
                  pos_seen = pos_seen+ 1
                  pos_weights = pos_weights+w
              else:
                  neg_seen = neg_seen+1
                  neg_weights = neg_weights+w
          
          clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
          classifiers.append(clf)
      return classifiers

def select(classifiers, weights, training_data):
      best_clf, best_error, best_accuracy = None, float('inf'), None
      error_total = []
      for clf in classifiers:
          error, accuracy = 0, []
          for data, w in zip(training_data, weights):
              correctness = abs(clf.classify(data[0]) - data[1])
              accuracy.append(correctness)
              error += w * correctness
          error = error / len(training_data)
          error_total.append(error)
          if error < best_error:
              best_clf, best_error, best_accuracy = clf, error, accuracy
      return best_clf, best_error, best_accuracy, error_total, classifiers



def plot(clfss,name):
  i = 0
  for c in clfss:
    i+=1
    pos = c.pos_region
    neg = c.neg_region
    feature = []
    if len(pos) == 1 and len(neg)==1:
      for p in pos:
        feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      for p in neg:
        feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
    elif len(pos) == 2 and len(neg)==1:
      p = pos[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = neg[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = pos[1]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
    elif len(pos) == 1 and len(neg)==2:
      p = neg[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = pos[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = neg[1]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
    elif len(pos) == 2 and len(neg)==2:
      p = pos[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = neg[0]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = pos[1]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])
      p = neg[1]
      feature.append([(p.y, p.x), (p.y+p.height,p.x+p.width)])

    image = training[0][0]
    image = draw_haar_like_feature(image, 0, 0,19, 19, [feature])
    plt.imshow(image)
    plt.savefig(str(name)+'-'+str(i))
def classify(image,alphas,clfs):  
    total = 0
    ii = integral(image)
    for alpha, clf in zip(alphas, clfs):
        total += alpha * clf.classify(ii)
    return 1 if total >= 0.5 * sum(alphas) else 0,total,0.5 * sum(alphas)


class WeakClassifier:
  def __init__(self, pos_region, neg_region, threshold, polarity):
      self.pos_region = pos_region
      self.neg_region = neg_region
      self.threshold = threshold
      self.polarity = polarity
  
  def classify(self, x):
      feature = lambda ii: sum([pos.compute(ii) for pos in self.pos_region]) - sum([neg.compute(ii) for neg in self.neg_region])
      return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

with open(path+"training.pkl", 'rb') as f:
        training = pickle.load(f)

pos_num = 500
neg_num = 1000

Total = 10
alphas = []
clfs = []
weights = np.zeros(len(training))
training_data = []
for x in range(len(training)):
    training_data.append((integral(training[x][0]), training[x][1]))
    if training[x][1] == 1:
        weights[x] = 1.0 / (2 * pos_num)
    else:
        weights[x] = 1.0 / (2 * neg_num)

height, width = training_data[0][0].shape
features = []
for w in range(1, width+1):
    for h in range(1, height+1):
        i = 0
        while i + w < width:
            j = 0
            while j + h < height:
                immediate = RR(i, j, w, h)
                right = RR(i+w, j, w, h)
                if i + 2 * w < width: 
                    features.append(([right], [immediate]))

                bottom = RR(i, j+h, w, h)
                if j + 2 * h < height: 
                    features.append(([immediate], [bottom]))
                
                right_2 = RR(i+2*w, j, w, h)
                if i + 3 * w < width:
                    features.append(([right], [right_2, immediate]))

                bottom_2 = RR(i, j+2*h, w, h)
                if j + 3 * h < height: 
                    features.append(([bottom], [bottom_2, immediate]))

                bottom_right = RR(i+w, j+h, w, h)
                if i + 2 * w < width and j + 2 * h < height:
                    features.append(([right, bottom], [immediate, bottom_right]))

                j += 1
            i += 1
features = np.array(features)
print('total features',features.shape[0])

print('applying features...')
X = np.zeros((len(features), len(training_data)))
y = np.array(list(map(lambda data: data[1], training_data)))
i = 0
for pos_region, neg_region in features:
    feature = lambda ii: sum([pos.compute(ii) for pos in pos_region]) - sum([neg.compute(ii) for neg in neg_region])
    X[i] = list(map(lambda data: feature(data[0]), training_data))  
    i = i+ 1


print('training...')
for t in range(Total):
  weights = weights / np.linalg.norm(weights)
  weak_classifiers = train(X, y, features, weights)
  clf, error, accuracy, total_err, classif = select(weak_classifiers, weights, training_data)
  beforeclfs = []
  if t==0:
    idx_sorted = np.argsort(total_err)
    for idx in range(10):
        beforeclfs.append(classif[idx_sorted[idx]])
    plot(beforeclfs,'beforeboost')
    print('Plotted before boosting top 10')
  beta = error / (1.0 - error)
  for i in range(len(accuracy)):
      weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
  alpha = math.log(1.0/beta)
  alphas.append(alpha)
  clfs.append(clf)
  print('Choosen weak classifier '+str(t+1))


with open(path+"test.pkl", 'rb') as f:
  test = pickle.load(f)
testn = []
for i in range(2000):
  if i > 1600:
    testn.append(test[i+23573-1560])
  else:
    testn.append(test[i])

testn[350][1]

len(testn)

data = testn
correct = 0
all_negatives, all_positives = 0, 0
true_negatives, false_negatives = 0, 0
true_positives, false_positives = 0, 0
classification_time = 0
P_roc = []
labels = []
for x, y in data:
    if y == 1:
        all_positives = all_positives+1
    else:
        all_negatives = all_negatives +1
    prediction,total,weird = classify(x,alphas,clfs)
    P_roc.append(total)
    labels.append(y)
    if prediction == 1 and y == 0:
        false_positives = false_positives+1
    if prediction == 0 and y == 1:

        false_negatives =false_negatives+ 1
    
    if prediction == y:
      correct += 1 
    else: correct += 0


print("False Positive Rate ", false_positives/all_negatives)
print("False Negative Rate ", false_negatives/all_positives)
print("Accuracy ",correct/len(data))



fpr, tpr, thresholds = metrics.roc_curve(labels,P_roc)
roc_auc = roc_auc_score(labels,P_roc)
plt.plot(fpr, tpr,color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plot(clfs,'afterboost')

