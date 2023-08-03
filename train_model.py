import transtab
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve, confusion_matrix, average_precision_score
import os
import sys


def create_exp_dir(path):
    path_split = path.split("/")
    path_i = "."
    for one_path in path_split:
        path_i += "/" + one_path
        if not os.path.exists(path_i):
            os.mkdir(path_i)


train_datas = sys.argv[1:-1]
test_data = sys.argv[-1:]
print("train_datas:{}".format(train_datas))
print("test_data:{}".format(test_data))

allset, trainset, valset, cat_cols, num_cols, bin_cols = transtab.load_data(train_datas)


# build classifier model
model = transtab.build_classifier(cat_cols, num_cols, bin_cols)


# specify training arguments, take validation loss for early stopping
train_basename = "train_"
for train_data in train_datas:
    train_basename_temp = os.path.basename(train_data)
    train_basename += train_basename_temp

output_dir = './checkpoint/' + train_basename

training_arguments = {
    'num_epoch':50,
    'batch_size':64,
    'lr':1e-4,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir': output_dir
    }

transtab.train(model, trainset, valset[0], **training_arguments)


model.load(output_dir)

allset, trainset, valset, cat_cols, num_cols, bin_cols = transtab.load_data(test_data)

x_test, y_test = allset[0]
ypred = transtab.predict(clf=model, x_test=x_test, y_test=y_test, return_loss=False)
AUC = round(roc_auc_score(y_test, ypred), 4)
print("Test AUC: %.2f" % (AUC * 100))


# store results
test_basename = os.path.basename(test_data[0])
file_path = "results_data/{}/{}".format(train_basename,test_basename)
create_exp_dir(file_path)
result_filename = os.path.join(file_path, test_basename + str(".txt"))
rs_fp = open(result_filename,'w')
ypred_list = [str(el) for el in ypred.tolist()]
l = "\n".join(ypred_list)
rs_fp.write(l)
